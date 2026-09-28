"""Ink detection on 8 µm Herculaneum surface volumes (ResNet3D-50 + 2D decoder)."""
from .modeling import InkDetector
from .inference import predict_stack, predict_surface, read_stack, save_prediction

__all__ = ["InkDetector", "predict_stack", "predict_surface", "read_stack", "save_prediction"]
