"""
XGBoost model for NSL-KDD intrusion detection.

Uses the histogram-based tree method for fast CPU training on Apple Silicon.
"""

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from .base import ModelWrapper


class XGBoostModel(ModelWrapper):
    """XGBoost binary classifier."""

    name = "xgboost"

    def build(self, n_features):
        self.n_features = n_features
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",        # fastest CPU algorithm
            n_jobs=-1,                 # use all M1 cores
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None, fast_mode=False):
        if fast_mode:
            self.model.set_params(n_estimators=80)
        if X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Training curve (logloss)
        results = self.model.evals_result()
        if results:
            epochs = range(1, len(results["validation_0"]["logloss"]) + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, results["validation_0"]["logloss"], label="Train")
            plt.plot(epochs, results["validation_1"]["logloss"], label="Val")
            plt.xlabel("Boosting round"); plt.ylabel("Log loss")
            plt.title("XGBoost Training History"); plt.legend(); plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/training_history.png", dpi=150)
            plt.close()

        # Built-in feature importance plot
        self._plot_builtin_importance()
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def _plot_builtin_importance(self):
        """XGBoost's native feature importance — used as ground-truth reference."""
        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        plt.figure(figsize=(8, 5))
        plt.barh(range(len(idx)), imp[idx], color="#27ae60", alpha=0.85)
        plt.yticks(range(len(idx)), [f"feature_{j}" for j in idx])
        plt.xlabel("Gain"); plt.title("XGBoost Built-in Feature Importance")
        plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig(f"{self.results_dir}/builtin_importance.png", dpi=150)
        plt.close()