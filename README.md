# Structural Cues for Automated Cognitive Presence Detection

Code for the paper on automated **Cognitive Presence (CP)** classification using a
transformer classifier augmented with **theory-informed structural cues** and
interpreted with SHAP.

Sentences are classified into five CP categories: Triggering Event, Exploration,
Integration, Resolution, and Non-CP (`Other` in the code/data).

## Files

| File | Description |
| --- | --- |
| `cue_dictionary.py` | The cue dictionary (**42 regex rules → 36 structural tags**) and tag-injection functions. |
| `code/RoBERTa.py`, `code/BERT.py` | Baseline classifiers (no cues). |
| `code/OurRoBERTa.py`, `code/OurBERT.py` | Classifiers **with structural cues** (primary model: `code/OurRoBERTa.py`). |

The cue dictionary and tag-injection logic referenced in the paper are in
`cue_dictionary.py`. Inspect it directly with `python cue_dictionary.py`.

## Setup

Python 3.10+.

```bash
pip install -U torch transformers datasets accelerate peft scikit-learn pandas numpy evaluate
```

## Data

The study data contain student text and are **not redistributed**. To run the
scripts, supply your own CSV with a `sentence` column (one sentence per row) and
a `consensus_category` label column with the values
`Triggering Event, Exploration, Integration, Resolution, Other`.
Different column names can be passed via `--text_col` and `--label_col`.

## Run

```bash
python code/RoBERTa.py    --csv_path data/annotated_data.csv                       # baseline
python code/OurRoBERTa.py --csv_path data/annotated_data.csv --use_struct_tags on  # with cues
```

Use `-h` on any script to see all options. The defaults are the configuration
reported in the paper (seed 42; stratified 80/20 train–test split; five-fold
stratified cross-validation within the training pool; RoBERTa-base with LoRA).

Rows are de-duplicated before splitting to prevent train–test leakage. The
hold-out split is saved to `splits/holdout.json` on the first run and reused on
subsequent runs, so all models are evaluated on the identical test set. Each run
writes its full configuration and test metrics to `summary.json` in the output
directory, along with per-fold metrics (`cv_fold_metrics.json`) and the averaged
per-class decision thresholds (`thresholds_avg.json`).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper:

> Yu, J. H., Tu, F., Chen, H., Ding, J., Hsieh, C.-J., Dong, L., Kim, H., &
> Watson, S. L. (in press). Measuring cognitive presence in online discussions:
> Automated detection and instructional insights from a MOOC context.
> *Educational Technology Research and Development.*

Archived release: v1.0-etrd (Zenodo DOI to be added).
