# NLP-Sentiment-Analysis

NLP Sentiment Analysis on Amazon Product Review Dataset 

The attached python code is a complete pipeline to perform sentiment analysis on the Amazon Product Review Dataset to classify reviews
as positive or negative.

**Dataset:**

**Amazon Review Dataset**
● Column 1: Binary sentiment labels in the dataset:
Positive: Label 2 (4-5 star reviews)
Negative: Label 1 (1-2 star reviews)
3 Star (Neutral Reviews) have been removed from the dataset
● Column 2: Review Title
● Column 3: Customer Review


**Data Preprocessing:**

○ Analysis of review length distribution, class distribution, and balance. Handle any missing values or malformed entries
○ Text preprocessing (lowercasing, removing stopwords, punctuation, stemming/lemmatization, etc.).
○ Tokenized the processed text
○ Textual Embedding

**Model Training & Evaluation:**

**Model A] Hashed-Embedding Deep Averaging (fastText-style baseline)**
Choose this baseline as it is extremely fast, vocabulary-free (fixed hash buckets), scales to millions of reviews.

Key hyperparameters
HASH_BINS = 2**20 (1,048,576)
EMBED_DIM = 128
HIDDEN_UNITS = 128
BATCH_SIZE = 2048
LR = 1e-3
Training regime: trained on a subset of 500,000 reviews (balanced), 4
epochs, evaluated on 200,000 test reviews.


Test results (@ threshold 0.50)
Accuracy: 0.8914
Precision (pos): 0.8894
Recall (pos): 0.8940
F1 (pos): 0.8917



**Model B] BiLSTM with Keras TextVectorization**
Choose the BiLSTM model because it keeps word order and bidirectional context, which helps with sentiment cues like “not good”, intensifiers, and phrase structure.

Vectorization (input layer)
MAX_TOKENS = 100,000 (vocabulary cap)
SEQ_LEN = 200 tokens (truncate/pad)

Training setup
EMBED_DIM = 128, LSTM_UNITS = 64, DROPOUT = 0.3
BATCH_SIZE = 1024, EPOCHS = 50 (with early stopping)
Callbacks: EarlyStopping(monitor="val_accuracy", patience=5,
restore_best_weights=True) and ReduceLROnPlateau.
Trained for ~15 epochs until early stopping converged.

Test Results -Default threshold 0.50
Accuracy: 0.9334
Negative: P 0.9357, R 0.9315, F1 0.9336
Positive: P 0.9311, R 0.9353, F1 0.9332

Threshold tuned on validation (best F1 at 0.460) → re-scored on test
Accuracy: 0.9332
Confusion Matrix: [[93173 7392]
[ 5972 93463]]


**Conclusion:**
The hashed model is a fast, scalable workhorse that nails clear-cut sentiment but misses compositional nuance; it’s ideal when simplicity and cost dominate. The BiLSTM is a quality first upgrade, cutting both false positives and false negatives significantly by using word order and context, while staying compact and easy to deploy. Use hashed for speed and simplicity; use BiLSTM when correctness and interpretability of contextual cues matter.
