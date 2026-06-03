"""
Abstract base class for model wrappers.

All models must implement the same interface so explainers and evaluators
can work with any of them via duck-typing.
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)


class ModelWrapper(ABC):
    """Common interface for all classifier models."""

    name: str = "base"

    def __init__(self, results_dir=None):
        self.results_dir = results_dir or f"results_{self.name}"
        os.makedirs(self.results_dir, exist_ok=True)
        self.model = None

    @abstractmethod
    def build(self, n_features: int):
        """Construct the underlying model."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, fast_mode=False):
        """Train the model. Returns self."""

    @abstractmethod
    def predict_proba(self, X):
        """Return P(attack) for each sample. Shape: (n,)."""

    @property
    def supports_gradients(self) -> bool:
        """Whether saliency/Grad-CAM apply to this model."""
        return False

    def evaluate(self, X_test, y_test):
        """Run standard evaluation + save confusion matrix + classification report."""
        y_prob = self.predict_proba(X_test)
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n  [{self.name}] Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {self.name.upper()}")
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/confusion_matrix.png", dpi=150)
        plt.close()

        return {
            "name": self.name,
            "accuracy": float(acc),
            "f1": float(f1),
            "auc": float(auc),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }