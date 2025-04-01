import numpy as np
from sklearn.metrics import f1_score

# Load arrays
y_true = np.load("./test_paired_true_type_infonval_cellViT256.npy")
y_pred = np.load("./test_paired_pred_type_infonval_cellViT256.npy")

# Find unique class labels
classes = np.unique(np.concatenate((y_true, y_pred)))

# Compute per-class F1 scores
for cls in classes:
    y_true_bin = (y_true == cls).astype(int)
    y_pred_bin = (y_pred == cls).astype(int)

    f1 = f1_score(y_true_bin, y_pred_bin)
    print(f"Class {cls}: F1 = {f1:.3f}")
