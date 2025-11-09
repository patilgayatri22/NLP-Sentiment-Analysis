import numpy as np, matplotlib.pyplot as plt, json, os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score

def to_binary(y): return (y.astype(int)==2).astype(int)

def tune_threshold(y_true, y_prob, lo=0.3, hi=0.7, steps=41):
    ts = np.linspace(lo, hi, steps)
    f1s = [f1_score(y_true, (y_prob>=t).astype(int)) for t in ts]
    best = ts[int(np.argmax(f1s))]
    return float(best)

def save_curves(y_true, y_prob, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC (AUC={roc_auc:.4f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.savefig(os.path.join(out_dir, f"{prefix}roc.png")); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec); plt.title("Precision-Recall")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.savefig(os.path.join(out_dir, f"{prefix}pr.png")); plt.close()

def dump_report(y_true, y_pred, y_prob, out_path, threshold):
    rep = classification_report(y_true, y_pred, target_names=["negative","positive"], digits=4, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    with open(out_path, "w") as f:
        json.dump({"threshold":threshold, "report":rep, "confusion_matrix":cm}, f, indent=2)
