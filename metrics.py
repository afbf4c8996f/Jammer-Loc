# metrics.py

import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging
from pathlib import Path
import numpy as np

# Obtain a module-specific logger
logger = logging.getLogger(__name__)

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute accuracy, F1 (macro and weighted), precision, and recall.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": accuracy, 
        "f1_macro": f1_macro, 
        "f1_weighted": f1_weighted, 
        "precision": precision, 
        "recall": recall
    }

def save_metrics(metrics: dict, filename: str, out_dir: str) -> None:
    """
    Save metrics dict to JSON in out_dir/filename, converting numpy types
    to native Python scalars/lists and writing atomically via a .tmp file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / filename
    tmp_path   = out_dir / (filename + ".tmp")

    def _convert(obj):
        # NumPy scalars
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let json handle everything else or raise
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Dump to temporary file first
    with open(tmp_path, "w") as f:
        json.dump(metrics, f, default=_convert, indent=4)
        f.flush()
        os.fsync(f.fileno())

    # Atomically move into place
    os.replace(tmp_path, final_path)
    logger.info(f"Metrics saved to {final_path}")



