# ===== hpo.py =====
import argparse
import subprocess
import json
import logging
import yaml
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from config_loader import load_config
from optuna.pruners import MedianPruner
from utils import build_metrics_filepath
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ─── Copy of CLI-override helper from main.py ────────────────────────────
def _cli_override(cfg, override_str: str) -> None:
    """
    Override configuration via CLI. Keys must exist in the config.
    Format: "key1.subkey=val,key2=val"
    """
    # Split override_str into key=val pairs, allowing commas inside values
    parts = override_str.split(",")
    kvs = []
    for token in parts:
        if "=" in token:
            kvs.append(token)
        else:
            if not kvs:
                raise ValueError(f"Invalid override segment '{token}'. Expected 'key=val'.")
            kvs[-1] += "," + token

    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Invalid override '{kv}'. Expected 'key=val'.")
        key, val_str = kv.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()

        # Traverse into cfg to find the target attribute
        tgt = cfg
        parts = key.split(".")
        for seg in parts[:-1]:
            if not hasattr(tgt, seg):
                raise AttributeError(f"Unknown config attribute: '{seg}' in '{key}'")
            tgt = getattr(tgt, seg)
        final_attr = parts[-1]
        if not hasattr(tgt, final_attr):
            raise AttributeError(f"Unknown config attribute: '{key}'")

        # Parse the string value into a bool, int, float, or leave as string
        if val_str.lower() in {"true", "false"}:
            val = val_str.lower() == "true"
        elif val_str.replace(".", "", 1).isdigit():
            val = float(val_str) if "." in val_str else int(val_str)
        else:
            val = val_str

        # If the original config attribute is a list, wrap/expand accordingly
        orig = getattr(tgt, final_attr)
        if isinstance(orig, list):
            if isinstance(val, str) and "," in val_str:
                # split comma-separated list
                val = [v.strip() for v in val_str.split(",")]
            elif not isinstance(val, list):
                val = [val]

        setattr(tgt, final_attr, val)

# ────────────────────────────────────────────────────────────────────────



def parse_args():
    ap = argparse.ArgumentParser("Per-model hyperparameter search")
    ap.add_argument("--task",       choices=["classification","regression"], required=True)
    ap.add_argument("--mode",       choices=["actual","rate"],         required=True)
    ap.add_argument("--framework",  choices=["ml","dl"],              required=True)
    ap.add_argument("--config",            default="config.yaml",
                    help="Base YAML config")
    ap.add_argument("--config_override",   default="",
                    help="Extra key=val overrides for main.py")
    ap.add_argument("--trials",     type=int, default=50)
    ap.add_argument("--timeout",    type=int, default=None,
                    help="Timeout (seconds) for the entire study")
    ap.add_argument("--storage",          default="sqlite:///hpo.db")
    ap.add_argument("--study-name",       default=None,
                    help="Prefix for Optuna study names & output files")
    ap.add_argument("--log-dir",   dest="log_dir",   default=None,
                    help="TensorBoard dir for HParams summary (outer loop)")
    ap.add_argument("--tb-logdir", dest="tb_logdir", default=None,
                    help="Parent TensorBoard dir for main.py inner-loop logs")
    ap.add_argument("--seed",       type=int, default=42)
    return ap.parse_args()


def objective(trial, args, model_name, fixed_overrides):
    # Start with the CLI‐fixed overrides (which include the fixed model_name)
    ov = list(fixed_overrides)
    ov = [
        f"model.task={args.task}",
        f"model.input_mode={args.mode}",
        f"model.framework={args.framework}",
    ]

    # Shared search spaces
    N_EST = [50, 100, 200,500,1000, 2000]
    MAX_D = [3, 5, 7, 9, 10, 12, 20, 30]

    # ML algorithms
    if args.framework == "ml":
        alg = model_name  # fixed
        # Only tune hyperparams for that alg
        if alg == "xgboost":
            ov = [
                f"model.xgb_n_estimators={trial.suggest_categorical('n_estimators', N_EST)}",
                f"model.max_depth={trial.suggest_categorical('max_depth', MAX_D)}",
                f"model.learning_rate={trial.suggest_categorical('xgb_learning_rate',[1e-3,1e-2,5e-2,1e-1])}",
                f"model.subsample={trial.suggest_float('subsample',0.5,1.0)}",
                f"model.colsample_bytree={trial.suggest_float('colsample_bytree',0.5,1.0)}",
                f"model.reg_alpha={trial.suggest_float('reg_alpha',1e-8,10,log=True)}",
                f"model.reg_lambda={trial.suggest_float('reg_lambda',1e-8,10,log=True)}",
            ]
        elif alg == "random_forest":
            ov = [
                f"model.rf_n_estimators={trial.suggest_categorical('n_estimators', N_EST)}",
                f"model.max_depth={trial.suggest_categorical('max_depth', MAX_D)}",
                f"model.min_samples_leaf={trial.suggest_int('min_samples_leaf',1,10)}",
                f"model.max_features={trial.suggest_float('max_features',0.5,1.0)}",
            ]
        else:  # knn
            ov = [
               f"model.knn_n_neighbors={trial.suggest_categorical('n_neighbors',[5,7,9,11,13,15,17,21])}",
                f"model.knn_weights={trial.suggest_categorical('knn_weights',['uniform','distance'])}",
            ]
        chosen = alg

    # DL architectures
    else:
        m = model_name  # fixed
        ov = [
            f"training.learning_rate={trial.suggest_categorical('training_learning_rate',[1e-4,1e-3,1e-2])}",
            f"model.hidden_size={trial.suggest_categorical('hidden_size',[128,256,512])}",
            f"training.batch_size={trial.suggest_categorical('batch_size',[128,256,512,1024])}",
            f"model.dropout={trial.suggest_categorical('dropout', [0.2, 0.3, 0.4])}",
        ]
        if m == "simple_nn":
            ov.append(f"model.num_blocks={trial.suggest_int('num_blocks',1,3)}")
        elif m == "convmixer1d":
            ov = [
                f"model.num_blocks={trial.suggest_int('num_blocks',1,3)}",
                f"model.kernel_size={trial.suggest_categorical('kernel_size',[3,5])}",
            ]
        elif m == "transformer_tabular":
            ov = [
                f"model.num_layers={trial.suggest_int('num_layers',1,4)}",
                "model.nhead=4",
                f"model.dim_feedforward={trial.suggest_categorical('dim_feedforward',[128,256,512,1024,2048])}",
            ]
        else:
            raise ValueError(f"Unsupported DL model '{m}'")
        chosen = m

    # Launch main.py with these overrides
    cmd = [
        "python", "main.py",
        "--config", args.config,
        "--config_override", ",".join(ov)
    ]
    if args.tb_logdir:
        cmd = ["--tb-logdir", str(Path(args.tb_logdir) / f"trial_{trial.number}")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logging.error("=== main.py STDOUT ===\n%s", res.stdout)
        logging.error("=== main.py STDERR ===\n%s", res.stderr)
        raise RuntimeError("main.py failed within HPO")




    mf = build_metrics_filepath(
        base_output_dir="outputs",
        task=args.task,
        mode=args.mode,
        framework=args.framework,
        model_name=chosen,
    )


    # ─── DEBUG: dump cwd & dir contents ──────────────────────────────
    import os
    logging.error("🔍 HPO trial cwd: %s", os.getcwd())
    logging.error("🔍 Looking for metrics at: %s", mf)
    try:
        listing = os.listdir(os.path.dirname(mf))
    except Exception as e:
        logging.error("Failed to list %s: %s", os.path.dirname(mf), e)
        listing = []
    logging.error("📂 Directory listing: %s", listing)

    # ─── Try to load metrics ──────────────────────────────────────────
    try:
        with open(mf) as f:
            m = json.load(f)
    except FileNotFoundError:
        logging.error("Expected metrics file not found: %s", mf)
        logging.error("=== main.py STDOUT ===\n%s", res.stdout)
        logging.error("=== main.py STDERR ===\n%s", res.stderr)
        # Re‐raise so Optuna sees the failure
        raise


    # ─── Pick the HPO objective ────────────────────────────────────────
    if args.framework == "ml":
        if args.task == "classification":
            score = m["val"]["f1_macro"]
        else:
            # For regression, optimize validation loss (e.g. MSE) if available,
            # otherwise fall back to median_distance
            score = m["val"].get("val_loss", m["val"].get("median_distance"))
    else:
        if args.task == "classification":
            score = m["val_accuracy"][-1]
        else:
            score = m["val"].get("val_loss", m["val"].get("median_distance"))

    trial.report(score, 0)
    return score


def run_hpo_for_model(model_name, args):
    # keep task, mode and framework in the base name
    prefix = args.study_name or f"{args.task}_{args.mode}_{args.framework}"
    # append model and timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    study_name = f"{prefix}_{model_name}_{timestamp}"
    study_name = f"{prefix}_{model_name}"
    logging.info(f"\n▶ Starting HPO study '{study_name}'")

    # fixed overrides lock in this model
    fixed = []
    # fixed overrides lock in exactly one method per trial:
    if args.framework == "ml":
        if args.task == "classification":
            # For ML‐classification, main.py will read cfg.model.algorithms
            fixed.append(f"model.algorithms={model_name}")
        else:
            # For ML‐regression, main.py (train_val_regression_models) now reads cfg.model.regressors
            fixed.append(f"model.regressors={model_name}")
    else:
        # For both DL‐classification and DL‐regression, main.py always uses cfg.model.name
        fixed.append(f"model.name={model_name}")

    # Choose what needs to be maximized or minimized
    direction = "maximize" if args.task == "classification" else "minimize"
    # Choose sampler & pruner based on framework
    if args.framework == "dl":
        # DL: multivariate TPE  Hyperband up to 50 epochs
       sampler = TPESampler(seed=args.seed, multivariate=True)
       pruner  = HyperbandPruner(min_resource=1, max_resource=50, reduction_factor=3)
    else:
        # ML: univariate TPE  MedianPruner for fast stopping
        sampler = TPESampler(seed=args.seed, multivariate=False)
        pruner  = MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=False
    )

    study.optimize(
        lambda t: objective(t, args, model_name, fixed),
        n_trials=args.trials,
        timeout=args.timeout
    )

    best = study.best_params
    best_file = f"config_best_{prefix}_{model_name}.yaml"
    with open(best_file, "w") as fh:
        yaml.dump(best, fh)
    logging.info(f"✔ Best config for '{model_name}' → {best_file}")

    # outer‐loop HParams summary
    if args.log_dir:
        sw = SummaryWriter(args.log_dir)
        sw.add_hparams(best, {"val_score": study.best_value})
        sw.close()

    return best_file


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    if args.config_override:
        _cli_override(base_cfg, args.config_override)


    # expand 'all' into concrete lists
    if args.framework == "ml":
        if args.task == "classification":
            raw = base_cfg.model.algorithms
            models = ["xgboost","random_forest","knn"] if raw == ["all"] else raw
        else:
            raw = getattr(base_cfg.model, "regressors", base_cfg.model.algorithms)
            models = ["xgboost","random_forest"] if raw == ["all"] else raw
    else:
        all_dl = ["simple_nn","convmixer1d","transformer_tabular"]
        if args.task == "classification":
            raw = base_cfg.model.dl_models
            models = all_dl if raw == ["all"] else raw
        else:
            raw = base_cfg.model.dl_regressors
            models = all_dl if raw == ["all"] else raw

    logging.info(f"Models to tune: {models}")

    for m in models:
        run_hpo_for_model(m, args)

    logging.info("HPO complete. Best configs saved; run final training separately.")


if __name__ == "__main__":
    main()
