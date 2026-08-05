"""
Random Forest model for NSL-KDD intrusion detection.

Uses sklearn's joblib parallel backend to leverage all M1 cores.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from .base import ModelWrapper


class RandomForestModel(ModelWrapper):
    """Random Forest binary classifier."""

    name = "rf"

    def build(self, n_features):
        self.n_features = n_features
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,           # all M1 cores
            random_state=42,
            class_weight="balanced",
        )
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None, fast_mode=False):
        if fast_mode:
            self.model.set_params(n_estimators=80, max_depth=12)
        self.model.fit(X_train, y_train)
        self._plot_builtin_importance()
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def _plot_builtin_importance(self):
        """Random Forest's native Gini-based feature importance."""
        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        plt.figure(figsize=(8, 5))
        plt.barh(range(len(idx)), imp[idx], color="#16a085", alpha=0.85)
        plt.yticks(range(len(idx)), [f"feature_{j}" for j in idx])
        plt.xlabel("Gini importance"); plt.title("Random Forest Built-in Feature Importance")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{self.results_dir}/builtin_importance.png", dpi=150)
        plt.close()