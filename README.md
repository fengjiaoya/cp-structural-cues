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
| `RoBERTa.py`, `BERT.py` | Baseline classifiers (no cues). |
| `OurRoBERTa.py`, `OurBERT.py` | Classifiers **with structural cues** (primary model: `OurRoBERTa.py`). |

The cue dictionary and tag-injection logic referenced in the paper are in
`cue_dictionary.py`. Inspect it directly with `python cue_dictionary.py`.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Data format

A CSV with a `sentence` column and a `category` label
(`Triggering Event, Exploration, Integration, Resolution, Other`). See
`data/annotated_data.sample.csv`. The study data contain student text and are not
redistributed; supply your own CSV in the same format.

## Run

```bash
python RoBERTa.py    --csv_path data/annotated_data.csv                       # baseline
python OurRoBERTa.py --csv_path data/annotated_data.csv --use_struct_tags on  # with cues
```

Use `-h` on any script to see all options.

## License

MIT — see `LICENSE`.
