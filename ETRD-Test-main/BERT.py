# train_eval.py
# Requirements:
# pip install -U transformers datasets accelerate scikit-learn pandas numpy evaluate
# pip install -U bitsandbytes peft   # for 8-bit quant + LoRA
import os, json, argparse, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support  # NEW
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
from dataclasses import replace

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------- Utilities -------------------------
def set_seed_all(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)

def macro_auprc(y_true: np.ndarray, probas: np.ndarray, num_labels: int) -> float:
    y_bin = label_binarize(y_true, classes=np.arange(num_labels))
    ap = []
    for c in range(num_labels):
        ap.append(average_precision_score(y_bin[:, c], probas[:, c]))
    return float(np.mean(ap))

def recall_at_precision(y_true: np.ndarray, probas: np.ndarray, p_thresh: float, num_labels: int):
    """Return (per_class_recalls, macro_recall) at max recall s.t. precision>=p_thresh (one-vs-rest)."""
    y_bin = label_binarize(y_true, classes=np.arange(num_labels))
    per_class = []
    for c in range(num_labels):
        precision, recall, _ = precision_recall_curve(y_bin[:, c], probas[:, c])
        mask = precision >= p_thresh
        r = float(np.max(recall[mask])) if np.any(mask) else 0.0
        per_class.append(r)
    return per_class, float(np.mean(per_class))

def find_thresholds_for_precision(y_true: np.ndarray, probas: np.ndarray, p_thresh: float, num_labels: int):
    """Return per-class probability thresholds that maximize recall while precision>=p_thresh."""
    y_bin = label_binarize(y_true, classes=np.arange(num_labels))
    thr_vec = np.full(num_labels, 1.01, dtype=float)  # default >1 => never predict that class
    for c in range(num_labels):
        precision, recall, thr = precision_recall_curve(y_bin[:, c], probas[:, c])
        best_r, best_t = 0.0, 1.01
        # precision/recall length = len(thr)+1; align thresholds from index 1
        for i in range(1, len(precision)):
            if precision[i] >= p_thresh and recall[i] > best_r:
                best_r, best_t = recall[i], float(thr[i-1])
        thr_vec[c] = best_t
    return thr_vec

# ------------------------- Custom Trainer -------------------------
class WeightedTrainer(Trainer):
    """
    Accepts class_weights in __init__ and uses them in CrossEntropyLoss.
    Casts weights to logits' dtype/device (fp16-safe).
    Also overrides get_train_dataloader to use WeightedRandomSampler.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # torch.Tensor or None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # match dtype/device for fp16/bf16
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)

        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        ds_train = self.train_dataset
        labels = np.array(ds_train["label"])
        bincount = np.bincount(labels, minlength=int(labels.max())+1)
        sample_weights = 1.0 / np.maximum(bincount[labels], 1)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.float),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            ds_train,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()

    # Data
    ap.add_argument("--csv_path", type=str, default="/home/fengjiao2/fjenv_new/FinalData2W20251008.csv")
    ap.add_argument("--text_col", type=str, default="sentence")
    ap.add_argument("--label_col", type=str, default="consensus_category")

    # CV/Test config
    ap.add_argument("--test_size", type=float, default=0.20)   # 20% independent Test
    ap.add_argument("--kfolds", type=int, default=5)           # 5-fold CV on remaining 80% #For test reuslt ----I change here, please change into 5
    ap.add_argument("--p_thresh", type=float, default=0.90)    # target precision threshold

    # Model
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=256) #For test reuslt ----I change here, please change into 256

    # LoRA / Quantization
    ap.add_argument("--use_lora", type=str, choices=["off","on"], default="on")
    ap.add_argument("--lora_r", type=int, default=16) 
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--quantize", type=str, choices=["none","8bit"], default="none")

    # Training
    ap.add_argument("--epochs", type=int, default=50) #For test reuslt ----I change here, please change into 50
    ap.add_argument("--lr", type=float, default=2e-4)          # LoRA-stable default #For test reuslt ----I change here, please change into 2e-4
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", type=str, default="false")

    # Imbalance
    ap.add_argument("--class_weight", type=str, choices=["off","on"], default="on")

    # Early stopping / best metric (aligned to high-confidence objective)
    ap.add_argument("--early_stop", type=str, choices=["off","on"], default="on")
    ap.add_argument("--early_patience", type=int, default=3)  # LoRA-stable default #For test reuslt ----I change here, please change into 3

    # Output
    ap.add_argument("--output_dir", type=str, default="./runs/roberta_cv5_test20")
    ap.add_argument("--save_merged", type=str, choices=["off","on"], default="on")

    args = ap.parse_args()
    set_seed_all(args.seed)
    print(f"DEBUG: Trying to use os at line 161. Current state of 'os': {os}") 
    os.makedirs(args.output_dir, exist_ok=True) # line 162

    use_cuda = torch.cuda.is_available()

    # ------------------------- Load data -------------------------
    df = pd.read_csv(args.csv_path)
    df = df[[args.text_col, args.label_col]].dropna().drop_duplicates()

    # Label mapping
    labels_sorted = sorted(df[args.label_col].unique().tolist())
    cls2id = {l: i for i, l in enumerate(labels_sorted)}
    id2label = {i: str(l) for l, i in cls2id.items()}
    df["labels"] = df[args.label_col].map(cls2id).astype(int)
    df["text"] = df[args.text_col].astype(str)
    num_labels = len(cls2id)

    # ------------------------- Tokenizer & Collator -------------------------
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, model_max_length=args.max_length)
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)
    #keep_cols = ["text", "label"]
    keep_cols = ["label"]
    data_collator = DataCollatorWithPadding(tok)

    # ------------------------- BitsAndBytes (optional) -------------------------
    bnb_cfg = None
    device_map = None
    if args.quantize == "8bit" and use_cuda:
        try:
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            device_map = "auto"
        except Exception:
            print("[WARN] 8-bit quantization failed, proceeding without it.")
            bnb_cfg = None
            device_map = None

    def build_base_model():
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels,
            id2label=id2label, label2id=cls2id,
            quantization_config=bnb_cfg,
            dtype=torch.float16 if (use_cuda and args.fp16.lower()=="true") else torch.float32,
            device_map=device_map
        )
        if bnb_cfg is not None:
            try:
                base_model = prepare_model_for_kbit_training(base_model)
            except Exception:
                pass

        if args.use_lora == "on":
            # For BERT/RoBERTa
            target_modules = ["query","key","value"]
            if "deberta" in args.model_name.lower():
                target_modules = ["query_proj","key_proj","value_proj"]

            lora_cfg = LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="SEQ_CLS", target_modules=target_modules
            )
            return get_peft_model(base_model, lora_cfg)
        else:
            return base_model

    # ------------------------- Class weights -------------------------
    cw = None
    if args.class_weight == "on":
        cw_np = compute_class_weight(class_weight="balanced",
                                     classes=np.arange(num_labels),
                                     y=df["labels"].values)
        cw = torch.tensor(cw_np, dtype=torch.float32, device=("cuda" if use_cuda else "cpu"))

    # ------------------------- Metrics -------------------------
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(p):
        logits = p.predictions
        probas = stable_softmax(logits)
        preds = np.argmax(probas, axis=1)
        y_true = p.label_ids
        macro_f1 = f1.compute(predictions=preds, references=y_true, average="macro")["f1"]
        m_auprc = macro_auprc(y_true, probas, num_labels)
        _, r_pX_macro = recall_at_precision(y_true, probas, args.p_thresh, num_labels)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=y_true)["accuracy"],
            "f1_weighted": f1.compute(predictions=preds, references=y_true, average="weighted")["f1"],
            "macro_f1": macro_f1,
            "macro_auprc": m_auprc,
            f"recall_at_p{int(args.p_thresh*100)}_macro": r_pX_macro,
        }

    # ------------------------- TrainingArguments (high-confidence aligned) -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(2*args.batch_size, 32),
        gradient_accumulation_steps=args.grad_accum,

        # Must be 'evaluation_strategy'
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        # Align best checkpoint & early stopping with high-confidence objective
        load_best_model_at_end=True,
        metric_for_best_model=f"recall_at_p{int(args.p_thresh*100)}_macro",
        greater_is_better=True,

        fp16=(args.fp16.lower()=="true" and use_cuda),
        bf16=False,
        warmup_ratio=0.08,
        weight_decay=0.01,
        dataloader_num_workers=0,
        seed=args.seed,
        ddp_find_unused_parameters=False
    )

    callbacks = []
    if args.early_stop == "on":
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_patience,
            early_stopping_threshold=0.0
        ))

    # ------------------------- Split: 20% independent Test -------------------------
    trainval_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["labels"]
    )

    # ---------- Persist or Reuse the same hold-out split ----------

    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    holdout_path = os.path.join(split_dir, "holdout.json")

    if not os.path.exists(holdout_path):
        # First time: save this test split for future reuse
        json.dump({
            "seed": args.seed,
            "test_size": args.test_size,
            "n_rows": int(len(df)),
            "test_idx": test_df.index.tolist()
        }, open(holdout_path, "w"))
        print(f"[INFO] Saved new hold-out split to {holdout_path}")
    else:
        # Next time: reload existing test indices to ensure the same 20% are used
        saved = json.load(open(holdout_path))
        test_idx = saved["test_idx"]
        test_df  = df.loc[test_idx]
        trainval_df = df.drop(index=test_idx)
        assert len(test_df) + len(trainval_df) == len(df)
        print(f"[INFO] Reusing existing hold-out split from {holdout_path}")


    # Helper: build HF datasets and tokenize
    def build_ds_tok(df_train, df_val, df_test):
        ds_local = DatasetDict({
            "train": Dataset.from_pandas(df_train[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False),
            "validation": Dataset.from_pandas(df_val[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False),
            "test": Dataset.from_pandas(df_test[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False),
        })

        ds_tok_local = ds_local.map(
            tok_fn, batched=True,
            remove_columns=[c for c in ds_local["train"].column_names if c != "label"]
        )
        return ds_local, ds_tok_local

    # ------------------------- K-fold CV on 80% pool -------------------------
    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.seed)
    fold_thresholds = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(trainval_df, trainval_df["labels"])):
        print(f"\n===== Fold {fold+1}/{args.kfolds} =====")
        tr_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        va_df = trainval_df.iloc[va_idx].reset_index(drop=True)

        dsd, ds_tok_fold = build_ds_tok(tr_df, va_df, test_df)

        model_fold = build_base_model()
        trainer_fold = WeightedTrainer(
            model=model_fold,
            args=training_args,
            train_dataset=ds_tok_fold["train"],
            eval_dataset=ds_tok_fold["validation"],
            processing_class=tok,                      # correct arg name
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            class_weights=cw,
        )

        print("Starting training (CV fold)...")
        trainer_fold.train()
        print("Fold training finished.")

        eval_val = trainer_fold.evaluate(ds_tok_fold["validation"])
        print("Fold val metrics:", eval_val)

        # Threshold search on validation
        pred_val = trainer_fold.predict(ds_tok_fold["validation"])
        probas_val = stable_softmax(pred_val.predictions)
        y_val = np.array(dsd["validation"]["label"])
        thr = find_thresholds_for_precision(y_val, probas_val, args.p_thresh, num_labels)
        fold_thresholds.append(thr)

        fold_metrics.append({
            "macro_auprc": float(eval_val.get("eval_macro_auprc", np.nan)),
            f"recall_at_p{int(args.p_thresh*100)}_macro": float(eval_val.get(f"eval_recall_at_p{int(args.p_thresh*100)}_macro", np.nan)),
            "macro_f1": float(eval_val.get("eval_macro_f1", np.nan)),
        })

    thr_avg = np.mean(np.vstack(fold_thresholds), axis=0)
    print("[INFO] Averaged per-class thresholds for P>=%.2f:" % args.p_thresh,
          {id2label[i]: float(t) for i, t in enumerate(thr_avg)})

    # ------------------------- Final training on full 80% (no eval) -------------------------
    print("\n===== Training final model on full 80% and evaluating on 20% Test =====")
    dsd_full = DatasetDict({
        "train": Dataset.from_pandas(trainval_df[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False),
        "validation": Dataset.from_pandas(trainval_df.sample(0)[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False), # empty
        "test": Dataset.from_pandas(test_df[["text","labels"]].rename(columns={"labels":"label"}), preserve_index=False),
    })
    ds_tok_full = dsd_full.map(tok_fn, batched=True,
                               remove_columns=[c for c in dsd_full["train"].column_names if c != "label"])

    final_args = replace(
    training_args,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=False,
    )

    final_model = build_base_model()
    final_trainer = WeightedTrainer(
        model=final_model,
        args=final_args,
        train_dataset=ds_tok_full["train"],
        eval_dataset=None,
        processing_class=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=cw,
    )
    final_trainer.train()

    # ------------------------- Evaluate on independent Test -------------------------
    pred_test = final_trainer.predict(ds_tok_full["test"])
    logits_test = pred_test.predictions
    probas_test = stable_softmax(logits_test)
    y_test = np.array(dsd_full["test"]["label"])
    texts_test = list(dsd_full["test"]["text"])

    # Argmax metrics (reference)
    pred_top1 = np.argmax(probas_test, axis=1)
    m_auprc_test = macro_auprc(y_test, probas_test, num_labels)
    _, r_pX_macro_test = recall_at_precision(y_test, probas_test, args.p_thresh, num_labels)
    m_f1_test = f1.compute(predictions=pred_top1, references=y_test, average="macro")["f1"]
    # ===== NEW: compute extra metrics for pretty printing =====
    from sklearn.metrics import accuracy_score

    acc_test = accuracy_score(y_test, pred_top1)

    # macro / weighted precision & recall & f1 (zero_division=0 to avoid warnings)
    p_macro, r_macro, f1_macro_prfs, _ = precision_recall_fscore_support(
        y_test, pred_top1, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, pred_top1, average="weighted", zero_division=0
    )

    f1_macro = m_f1_test

    # ===== NEW: pretty print =====
    print(
        "\n[TEST] "
        f"Accuracy: {acc_test:.6f} | "
        f"Precision(macro): {p_macro:.6f} | Recall(macro): {r_macro:.6f} | F1(macro): {f1_macro:.6f} | "
        f"Precision(weighted): {p_weighted:.6f} | Recall(weighted): {r_weighted:.6f} | F1(weighted): {f1_weighted:.6f} | "
        f"Macro AUPRC: {m_auprc_test:.6f} | Recall@P>={args.p_thresh:.2f} (macro): {r_pX_macro_test:.6f}"
    )

    print("\n[TEST] Macro AUPRC: %.6f | Recall@P>=%.2f (macro): %.6f | Macro-F1: %.6f"
          % (m_auprc_test, args.p_thresh, r_pX_macro_test, m_f1_test))

    # Thresholded (high-precision) predictions on Test
    pred_bin = (probas_test >= thr_avg[None, :]).astype(int)
    for i in range(len(pred_bin)):
        if pred_bin[i].sum() == 0:
            pred_bin[i, pred_top1[i]] = 1

    # ------------------------- Save artifacts & CSVs -------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_dir = os.path.join(args.output_dir, "metrics_exports")
    os.makedirs(metrics_dir, exist_ok=True)

    # id2label JSON
    id2label_path = os.path.join(args.output_dir, "id2label.json")
    with open(id2label_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, ensure_ascii=False, indent=2)

    # Save CV summary & thresholds
    with open(os.path.join(args.output_dir, "cv_fold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "thresholds_avg.json"), "w", encoding="utf-8") as f:
        json.dump({id2label[i]: float(t) for i, t in enumerate(thr_avg)}, f, ensure_ascii=False, indent=2)

    # Per-class PR curve points on Test
    y_bin_test = label_binarize(y_test, classes=np.arange(num_labels))
    with open(os.path.join(metrics_dir, "per_class_pr_curves_test.csv"), "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "threshold_index", "precision", "recall"])
        for c in range(num_labels):
            precision, recall, _ = precision_recall_curve(y_bin_test[:, c], probas_test[:, c])
            for i in range(len(precision)):
                writer.writerow([c, id2label[c], i, float(precision[i]), float(recall[i])])

    # Per-class AUPRC & Recall@P≥X on Test
    with open(os.path.join(metrics_dir, "per_class_auprc_recall_pX_test.csv"), "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "auprc", f"recall_at_p{int(args.p_thresh*100)}"])
        per_class_recalls_test, _ = recall_at_precision(y_test, probas_test, args.p_thresh, num_labels)
        for c in range(num_labels):
            ap = average_precision_score(y_bin_test[:, c], probas_test[:, c])
            writer.writerow([c, id2label[c], float(ap), float(per_class_recalls_test[c])])

    # Summary macro metrics on Test
    # Summary metrics on Test (expanded)
    with open(os.path.join(metrics_dir, "summary_macro_metrics_test.csv"), "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow([
            "accuracy",
            "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
            "macro_auprc",
            f"recall_at_p{int(args.p_thresh*100)}_macro"
        ])
        writer.writerow([
            acc_test,
            p_macro, r_macro, f1_macro,
            p_weighted, r_weighted, f1_weighted,
            m_auprc_test,
            r_pX_macro_test
        ])

    # Ranked outputs (argmax confidence) on Test
    pred_conf = probas_test[np.arange(len(pred_top1)), pred_top1]
    rows = []
    for i in range(len(texts_test)):
        row = {
            "text": texts_test[i],
            "true_label": id2label[y_test[i]],
            "pred_label": id2label[pred_top1[i]],
            "pred_conf": float(pred_conf[i]),
        }
        for c in range(num_labels):
            row[f"p({id2label[c]})"] = float(probas_test[i, c])
        rows.append(row)
    rows.sort(key=lambda r: r["pred_conf"], reverse=True)
    cols = ["text", "true_label", "pred_label", "pred_conf"] + [f"p({id2label[c]})" for c in range(num_labels)]
    with open(os.path.join(metrics_dir, "test_predictions_ranked.csv"), "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader()
        for r in rows: writer.writerow(r)

    # Thresholded (high-precision-oriented) predictions CSV on Test
    with open(os.path.join(metrics_dir, "test_predictions_thresholded_pX.csv"), "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        hdr = ["text", "true_label", "argmax_label"] + [f"pred_{id2label[c]}(>=thr?)" for c in range(num_labels)] + \
              [f"p({id2label[c]})" for c in range(num_labels)]
        writer.writerow(hdr)
        for i in range(len(texts_test)):
            row = [texts_test[i], id2label[y_test[i]], id2label[pred_top1[i]]] + \
                  pred_bin[i].astype(int).tolist() + list(map(float, probas_test[i]))
            writer.writerow(row)

    # Save PEFT model/tokenizer
    peft_dir = os.path.join(args.output_dir, "model_peft")
    final_trainer.save_model(peft_dir)
    tok.save_pretrained(peft_dir)

    # Optionally merge LoRA for deployment
    if args.save_merged == "on" and args.use_lora == "on":
        try:
            merged = final_trainer.model.merge_and_unload()
            save_dir = os.path.join(args.output_dir, "model_merged")
            os.makedirs(save_dir, exist_ok=True)
            merged.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            pd.Series(id2label).to_json(os.path.join(save_dir, "id2label.json"))
            print(f"Merged model saved to: {save_dir}")
        except Exception as e:
            print("Merge LoRA failed:", e)

    # Summary JSON
    train_counts_py = {int(k): int(v) for k, v in Counter(trainval_df["labels"].tolist()).items()}
    summary = {
        "model": args.model_name,
        "max_length": args.max_length,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "class_weight": args.class_weight,
        "quantize": args.quantize,
        "use_lora": args.use_lora,
        "kfolds": args.kfolds,
        "p_thresh": args.p_thresh,
        "num_labels": num_labels,
        "trainval_counts": train_counts_py,
        "test_size": args.test_size,
        "test_macro_auprc": m_auprc_test,
        f"test_recall_at_p{int(args.p_thresh*100)}_macro": r_pX_macro_test,
        "test_macro_f1": m_f1_test,
        "test_accuracy": acc_test,
        "test_precision_macro": p_macro,
        "test_recall_macro": r_macro,
        "test_f1_macro": f1_macro,
        "test_precision_weighted": p_weighted,
        "test_recall_weighted": r_weighted,
        "test_f1_weighted": f1_weighted,
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
