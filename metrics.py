import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_and_save(y_true, y_pred, label_names, output_dir, prefix=""):
    """
    Print classification metrics and save a confusion matrix plot.

    Args:
        y_true:      list of ground-truth integer labels
        y_pred:      list of predicted integer labels (None entries are dropped)
        label_names: ordered list of class name strings
        output_dir:  directory to write the confusion matrix PNG
        prefix:      filename prefix (e.g. model name)

    Returns:
        dict of scalar metric values
    """
    paired = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not paired:
        print("No valid predictions to evaluate.")
        return {}

    yt, yp = zip(*paired)
    n_total = len(y_true)
    n_valid = len(yt)
    print(f"\nParsed {n_valid}/{n_total} responses successfully.")

    acc       = accuracy_score(yt, yp)
    f1_mac    = f1_score(yt, yp, average="macro",    zero_division=0)
    f1_wt     = f1_score(yt, yp, average="weighted", zero_division=0)
    prec      = precision_score(yt, yp, average="macro", zero_division=0)
    rec       = recall_score(yt, yp, average="macro",    zero_division=0)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:            {acc:.4f}")
    print(f"F1 (macro):          {f1_mac:.4f}")
    print(f"F1 (weighted):       {f1_wt:.4f}")
    print(f"Precision (macro):   {prec:.4f}")
    print(f"Recall (macro):      {rec:.4f}")
    print("\nPer-class report:")
    print(classification_report(yt, yp, target_names=label_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(yt, yp, labels=list(range(len(label_names))))
    fig_size = max(6, len(label_names) * 1.8)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
    )
    plt.title(f"{prefix} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"{prefix}confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix → {cm_path}")

    return {
        "accuracy": acc,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "precision_macro": prec,
        "recall_macro": rec,
        "n_valid": n_valid,
        "n_total": n_total,
    }
