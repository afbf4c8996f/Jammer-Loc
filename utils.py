#utils.py
import os
import torch
from pathlib import Path

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# List of Columns not to be used for train/val/testing
LEAKY_COLS = [
    # "receiver_id",
    # "dt_ms",
    "rate_payload_mhr_seq_num",
    # "rate_payload_data_tx_timestamp_ms",
    "rate_payload_data_idx",
   # "rate_event_cnt_rto",
    ##########################
    "rate_payload_fcs",
    #"rate_rssi",
    #"rate_dgc",
    #"rate_ipatovpeak",
    #"rate_ipatovpower",
    #"rate_ipatovf1",
    #"rate_ipatovf2",
    #"rate_ipatovf3",
    #"rate_ipatovfpindex",
    'event_cnt_rto'
]


def get_device():
    """
    Return the best available compute device in priority:
      1) CUDA  (NVIDIA GPUs)
      2) MPS   (Apple-Silicon GPUs, macOS 12.3+)
      3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Check for Apple-Silicon Metal backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




def get_run_output_dir(cfg) -> str:
    """
    Build ./outputs/<task>_<mode>_<framework>/  and return it.
    Keeps DL/ML, actual/rate, classification/regression completely separate.
    """
    sub = f"{cfg.model.task}_{cfg.model.input_mode}_{cfg.model.framework}"
    out_dir = Path(cfg.output.directory) / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)



def build_metrics_filepath(
    base_output_dir: str,
    task: str,
    mode: str,
    framework: str,
    model_name: str,
) -> Path:
    """
    Construct the path to the metrics JSON for HPO.
    """
    subdir = f"{task}_{mode}_{framework}"
    model_dir = Path(base_output_dir) / subdir / model_name

    if task == "regression":
        suffix = "regression_dl" if framework == "dl" else "regression"
        filename = f"{model_name}_{suffix}_metrics.json"
    elif task == "classification":
        # Both ML and DL save their classification metrics as <model_name>_metrics.json
        suffix = "classification_dl" if framework == "dl" else "classification_ml"
        filename = f"{model_name}_metrics.json"
    else:  # classification
        filename = f"{model_name}_metrics.json"

    return model_dir / filename
