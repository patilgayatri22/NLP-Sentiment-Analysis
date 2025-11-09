import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----- Hashed-average (fastText-style) -----
def build_hashed_avg(hash_bins=2**20, embed_dim=128, hidden_units=128, dropout=0.2, lr=1e-3):
    text_in = keras.Input(shape=(), dtype=tf.string, name="text")
    tokens = tf.strings.split(text_in)
    ids = layers.Hashing(num_bins=hash_bins)(tokens)
    emb = layers.Embedding(hash_bins, embed_dim, name="embed")(ids)
    x = tf.reduce_mean(emb, axis=1)
    x = layers.Dense(hidden_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(text_in, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")])
    return model

# ----- BiLSTM with TextVectorization -----
def build_vectorizer(max_tokens=100_000, seq_len=200):
    return layers.TextVectorization(max_tokens=max_tokens, standardize="lower_and_strip_punctuation",
                                    split="whitespace", output_mode="int", output_sequence_length=seq_len)

def build_bilstm(vocab_size, seq_len=200, embed_dim=128, lstm_units=64, dropout=0.3, lr=1e-3):
    inp = keras.Input(shape=(seq_len,), dtype=tf.int64, name="tokens")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")(inp)
    x = layers.SpatialDropout1D(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")])
    return model
