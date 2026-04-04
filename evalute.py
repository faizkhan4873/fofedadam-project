import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def evaluate_model(model, X, y):
    model.eval()

    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X).numpy().flatten()

    preds_binary = (preds > 0.5).astype(int)

    acc = accuracy_score(y, preds_binary)

    try:
        auc = roc_auc_score(y, preds)
    except:
        auc = 0.5  # fallback if only one class

    f1 = f1_score(y, preds_binary, average="macro")

    return acc, auc, f1