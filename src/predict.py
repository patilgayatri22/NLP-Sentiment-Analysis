import argparse, pandas as pd, tensorflow as tf

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_csv", required=True)   # columns like the raw data (no header ok)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, header=None, names=["label","review_title","review_text"])
    df["text"] = (df["review_title"].fillna("") + ". " + df["review_text"].fillna("")).str.strip()

    model = tf.keras.models.load_model(args.model_path)
    probs = model.predict(df["text"].astype(str).to_numpy(), verbose=0).ravel()
    df_out = df.assign(p_positive=probs, pred=(probs>=0.5).astype(int))
    df_out.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
