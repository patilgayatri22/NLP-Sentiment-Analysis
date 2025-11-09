import argparse, os, numpy as np, pandas as pd, tensorflow as tf
from dataset import load_raw, standardize, preprocess_series
from models import build_hashed_avg, build_vectorizer, build_bilstm
from utils import to_binary

def ds_hashed(df, batch, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((df["text"].astype(str).values, (df["label"].astype(int)==2).astype(np.int32).values))
    if shuffle: ds = ds.shuffle(min(len(df), 1_000_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

def ds_bilstm(df, vec, batch, shuffle, seed):
    x = tf.data.Dataset.from_tensor_slices(df["text"].astype(str).values)
    y = tf.data.Dataset.from_tensor_slices((df["label"].astype(int)==2).astype(np.int32).values)
    ds = tf.data.Dataset.zip((x,y))
    if shuffle: ds = ds.shuffle(min(len(df), 1_000_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch).map(lambda tx, ty: (vec(tx), ty)).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hashed","bilstm"], required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outputs, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    train = standardize(load_raw(args.train))
    test  = standardize(load_raw(args.test))

    # quick cleaning mode for ML
    train["text"] = preprocess_series(train["text"], mode="ml_stem")
    test["text"]  = preprocess_series(test["text"], mode="ml_stem")

    if args.model == "hashed":
        model = build_hashed_avg()
        val = test.iloc[:len(test)//2]
        train_ds = ds_hashed(train, 2048, True, args.seed)
        val_ds   = ds_hashed(val,   2048, False, args.seed)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=cbs, verbose=1)

    else:  # bilstm
        vec = build_vectorizer()
        sample = train["text"].astype(str).sample(min(800_000, len(train)), random_state=args.seed).values
        vec.adapt(tf.data.Dataset.from_tensor_slices(sample).batch(4096))
        model = build_bilstm(vec.vocabulary_size())
        val = test.iloc[:len(test)//2]
        train_ds = ds_bilstm(train, vec, 1024, True, args.seed)
        val_ds   = ds_bilstm(val,   vec, 1024, False, args.seed)
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5)
        ]
        hist = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=cbs, verbose=1)

    model_dir = os.path.join(args.outputs, "models"); os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, f"{args.model}.keras"))
