import argparse, os, json, numpy as np, pandas as pd, tensorflow as tf
from utils import to_binary, save_curves, dump_report, tune_threshold

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--fixed_threshold", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.test, header=0 if "label," in open(args.test).readline() else None)
    if df.columns.size==3: df.columns = ["label","review_title","review_text"]; df["text"]=(df["review_title"].fillna("")+". "+df["review_text"].fillna("")).str.strip()
    y_true = (df["label"].astype(int)==2).astype(int).to_numpy()

    model = tf.keras.models.load_model(args.model_path)
    y_prob = model.predict(df["text"].astype(str).to_numpy(), verbose=0).ravel()

    thr = float(args.fixed_threshold) if args.fixed_threshold is not None else tune_threshold(y_true, y_prob)
    y_pred = (y_prob>=thr).astype(int)

    metrics_dir = os.path.join(args.outputs, "metrics")
    figs_dir = os.path.join(args.outputs, "figs")
    os.makedirs(metrics_dir, exist_ok=True)

    dump_report(y_true, y_pred, y_prob, os.path.join(metrics_dir, "eval.json"), thr)
    save_curves(y_true, y_prob, figs_dir)
    print(f"Saved: metrics→{metrics_dir}, figs→{figs_dir}, threshold={thr:.3f}")
