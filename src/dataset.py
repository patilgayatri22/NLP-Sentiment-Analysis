from pathlib import Path
import pandas as pd
import re
from tqdm.auto import tqdm
from nltk.stem import SnowballStemmer

COLS = {"label":"label","title":"review_title","text":"review_text"}
NEGATIONS = {"no","not","nor","never","n't"}
_url  = re.compile(r"https?://\S+|www\.\S+")
_html = re.compile(r"<.*?>")
_mention = re.compile(r"[@#]\w+")
_non_letter = re.compile(r"[^a-z\s]+")
_token = re.compile(r"[a-z]+")
stemmer = SnowballStemmer("english")

def load_raw(csv_path: str|Path) -> pd.DataFrame:
    c = COLS
    df = pd.read_csv(csv_path, header=None, names=[c["label"],c["title"],c["text"]],
                     encoding="utf-8", on_bad_lines="skip")
    df[c["label"]] = pd.to_numeric(df[c["label"]], errors="coerce").astype("Int64")
    df[c["title"]] = df[c["title"]].astype(str).str.strip()
    df[c["text"]]  = df[c["text"]].astype(str).str.strip()
    df["text"] = (df[c["title"]].fillna("") + ". " + df[c["text"]].fillna("")).str.strip()
    return df

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label"].isin([1,2])].copy()
    df["label_text"] = df["label"].map({1:"negative",2:"positive"})
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len()>0].drop_duplicates(subset=["label","text"])
    return df

def normalize_minimal(s: str) -> str:
    s = s.lower()
    s = _url.sub(" ", s); s = _html.sub(" ", s); s = _mention.sub(" ", s)
    return re.sub(r"\s+"," ", s).strip()

def process_ml_stem(s: str, stopwords: set) -> str:
    s = normalize_minimal(s)
    s = _non_letter.sub(" ", s)
    toks = _token.findall(s)
    toks = [t for t in toks if t not in stopwords and len(t)>1]
    toks = [stemmer.stem(t) for t in toks]
    return " ".join(toks)

def preprocess_series(series: pd.Series, mode="ml_stem", stopwords: set|None=None, batch_rows=200_000):
    if stopwords is None:
        stopwords = set()  # you can wire in spaCy/NLTK if desired
    out = []
    fn = (lambda x: process_ml_stem(x, stopwords)) if mode=="ml_stem" else normalize_minimal
    for start in tqdm(range(0, len(series), batch_rows), desc=f"preprocess({mode})", unit="rows"):
        chunk = series.iloc[start:start+batch_rows].tolist()
        out.extend([fn(x if isinstance(x,str) else "") for x in chunk])
    return pd.Series(out, index=series.index)
