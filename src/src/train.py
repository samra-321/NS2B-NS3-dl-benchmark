"""Train and evaluate a PyTorch MLP on NS2B-NS3 bioactivity data,
then compare against the classical RF/SVM baseline from the MS thesis."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from model import BioactivityMLP


def load_data():
    X = np.load("data/X_fingerprints.npy")
    y = np.load("data/y_labels.npy")
    return X, y


def train_mlp(X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    model = BioactivityMLP(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    best_auc = 0
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).numpy()
        val_auc = roc_auc_score(y_val, val_preds)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, val_auc={val_auc:.4f}")

    return model, best_auc


def train_baselines(X_train, y_train, X_val, y_val):
    results = {}

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    results["RandomForest"] = rf_auc

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_auc = roc_auc_score(y_val, svm.predict_proba(X_val)[:, 1])
    results["SVM"] = svm_auc

    return results


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training PyTorch MLP...")
    mlp_model, mlp_auc = train_mlp(X_train, y_train, X_val, y_val)

    print("\nTraining classical baselines...")
    baseline_results = train_baselines(X_train, y_train, X_val, y_val)

    print("\n=== Results ===")
    print(f"PyTorch MLP ROC-AUC: {mlp_auc:.4f}")
    for name, auc in baseline_results.items():
        print(f"{name} ROC-AUC: {auc:.4f}")
