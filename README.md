# NLP-Sentiment-Analysis

Binary sentiment analysis on Amazon-style product reviews (1 = negative, 2 = positive) with two models:
- **hashed** — fastText-style hashed-embedding averaging (very fast, scalable)
- **bilstm** — BiLSTM with Keras TextVectorization (higher accuracy, keeps word order)

<img src="examples/bilstm_amazon_pipeline.png" width="700" />

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Training Options](#training-options)
- [Evaluation & Outputs](#evaluation--outputs)
- [Results](#results)
- [Notes & Guidance](#notes--guidance)
- [License](#license)

---

## Dataset

**Format:** CSV without header  
Columns:
1. `label` — binary sentiment  
   - **2** → Positive (4–5 stars)  
   - **1** → Negative (1–2 stars)  
   - Neutral (3-star) rows are removed
2. `review_title`
3. `review_text`

Place files here:
```
data/raw/train.csv
data/raw/test.csv
```

---

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
├── examples/
│   └── bilstm_amazon_pipeline.png
├── notebooks/
│   └── Sentiment_Analysis_Python.ipynb
├── outputs/
│   ├── models/     # exported .keras / .h5
│   ├── metrics/    # JSON/CSV reports
│   └── figs/       # ROC/PR/CM plots
└── src/
    ├── dataset.py          # load/clean/preprocess
    ├── models.py           # build_hashed_avg(), build_bilstm()
    ├── train_sentiment.py  # --model {hashed,bilstm}
    ├── evaluate.py         # eval + plots
    ├── predict.py          # batch inference
    └── utils.py            # metrics, threshold tuning, plotting
```

---

## Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional (Docker):
```bash
docker compose up --build
```

---

## Quickstart

**Train (hashed baseline)**
```bash
python src/train_sentiment.py \
  --model hashed \
  --train data/raw/train.csv \
  --test  data/raw/test.csv
```

**Evaluate**
```bash
python src/evaluate.py \
  --model_path outputs/models/hashed.keras \
  --test data/raw/test.csv
```

**Train (BiLSTM)**
```bash
python src/train_sentiment.py \
  --model bilstm \
  --train data/raw/train.csv \
  --test  data/raw/test.csv
```

---

## Training Options

- `--model {hashed,bilstm}` — select model
- Data is loaded from raw CSV (no header), cleaned, and minimally normalized
- **Hashed** model splits on whitespace and hashes tokens to fixed buckets
- **BiLSTM** uses `TextVectorization` (max vocab & sequence length set in code)

You can adjust batch sizes, epochs, embedding dims, etc. directly in `src/models.py` or by adding simple flags.

---

## Evaluation & Outputs

Running `src/evaluate.py` saves:
- `outputs/metrics/eval.json` — threshold used, classification report, confusion matrix
- `outputs/figs/roc.png` and `outputs/figs/pr.png`

**Batch inference**
```bash
python src/predict.py \
  --model_path outputs/models/bilstm.keras \
  --input_csv data/raw/test.csv \
  --output_csv outputs/predictions.csv
```

---

## Results

### Model A — Hashed-Embedding Deep Averaging (fastText-style baseline)
Chosen for speed, fixed hash buckets (vocab-free), scales to millions.

**Key hyperparameters**
- `HASH_BINS = 2**20 (1,048,576)`
- `EMBED_DIM = 128`
- `HIDDEN_UNITS = 128`
- `BATCH_SIZE = 2048`
- `LR = 1e-3`

**Training regime**
- Trained on a balanced subset of **500,000** reviews for **4** epochs  
- Evaluated on **200,000** test reviews

**Test (@ threshold 0.50)**
- **Accuracy:** 0.8914  
- **Precision (pos):** 0.8894  
- **Recall (pos):** 0.8940  
- **F1 (pos):** 0.8917  

---

### Model B — BiLSTM with Keras TextVectorization
Keeps word order & bidirectional context (helps with “not good”, intensifiers, phrases).

**Vectorization**
- `MAX_TOKENS = 100,000`
- `SEQ_LEN = 200` tokens

**Training**
- `EMBED_DIM = 128`, `LSTM_UNITS = 64`, `DROPOUT = 0.3`  
- `BATCH_SIZE = 1024`, `EPOCHS = 50` with EarlyStopping & ReduceLROnPlateau  
- Converged in ~15 epochs

**Test (default threshold 0.50)**
- **Accuracy:** 0.9334  
- **Negative:** P 0.9357, R 0.9315, F1 0.9336  
- **Positive:** P 0.9311, R 0.9353, F1 0.9332

**Threshold tuned on validation (best F1 at 0.460) → re-scored on test**
- **Accuracy:** 0.9332  
- **Confusion Matrix:**
  ```
  [[93173  7392]
   [ 5972 93463]]
  ```

---

## Notes & Guidance

- **When to use what**
  - **Hashed**: fastest/cheapest, great for large-scale deployments, but loses word order
  - **BiLSTM**: better on nuanced phrases and negations; still compact and easy to ship
- **Outputs** land in `outputs/` (models, metrics, figs)
- **Diagram** lives in `examples/bilstm_amazon_pipeline.png` and is embedded above

---

## License

MIT