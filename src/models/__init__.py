"""Model wrappers for the XAI-NIDS pipeline."""

from .base import ModelWrapper
from .cnn import CNNModel, configure_gpu
from .xgboost_model import XGBoostModel
from .rf import RandomForestModel

__all__ = ["ModelWrapper", "CNNModel", "XGBoostModel", "RandomForestModel", "configure_gpu"]