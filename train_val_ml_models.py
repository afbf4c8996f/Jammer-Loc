import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix

from metrics import compute_metrics, save_metrics
from plotting import plot_confusion_matrix, plot_metric_comparison
from config_loader import ModelConfig
from config_loader import Config

logger = logging.getLogger(__name__)

# ────────────────────────── model factory ────────────────────────────────
  # for type hints or ensuring import

def build_ml_model(algo_name: str,  model_cfg: ModelConfig):
    """Return an un-trained scikit-learn/XGBoost/KNN model configured via cfg.model"""
    # XGBoost handles both classification and regression
    if algo_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor
        common_args = dict(
            n_estimators    = model_cfg.xgb_n_estimators,
            max_depth       = model_cfg.max_depth,
            learning_rate   = model_cfg.learning_rate,
            subsample       = model_cfg.subsample,
            colsample_bytree= model_cfg.colsample_bytree,
            reg_alpha       = model_cfg.reg_alpha,
            reg_lambda      = model_cfg.reg_lambda,
            random_state    = 42,
        )
        if model_cfg.task == "classification":
            return XGBClassifier(**common_args,
                                  objective="multi:softprob",
                                  eval_metric="mlogloss",
                                  early_stopping_rounds=model_cfg.early_stopping_rounds,
                                  )
        else:
            return XGBRegressor(**common_args,
                                 objective="reg:squarederror",
                                 eval_metric="rmse",
                                 early_stopping_rounds=model_cfg.early_stopping_rounds,
                                 )

    # Random Forest for classification & regression
    elif algo_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        common_args = dict(
            n_estimators    = model_cfg.rf_n_estimators,
            max_depth       = model_cfg.max_depth,
            min_samples_leaf= model_cfg.min_samples_leaf,
            max_features    = model_cfg.max_features,
            random_state    = 42,
            n_jobs          = -1,
        )
        if model_cfg.task == "classification":
            return RandomForestClassifier(**common_args)
        else:
            return RandomForestRegressor(**common_args)

    # KNN only for classification
    elif algo_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors    = model_cfg.knn_n_neighbors,
            weights        = model_cfg.knn_weights,
        )
    else:
        raise ValueError(f"Unknown algo_name = {algo_name!r}")
# ───────────────────── registry helper (wild-card ‘all’) ──────────────────
def _list_available_algos() -> List[str]:
    """Return the names that build_ml_model() understands."""
    return ["xgboost", "random_forest", "knn"]


def _resolve_algo_names(cfg_names: List[str]) -> List[str]:
    """
    Expand wildcard and validate names.

    • If “all” appears, train every available algorithm.
    • Otherwise ensure each name is known.
    """
    if "all" in cfg_names:
        resolved = _list_available_algos()
    else:
        resolved = cfg_names
    unknown = [n for n in resolved if n not in _list_available_algos()]
    if unknown:
        raise ValueError(
            f"Unknown algorithm(s): {unknown}. "
            f"Valid options: {_list_available_algos()}"
        )
    return resolved

# ─────────────────────────── core pipeline ───────────────────────────────
def train_ml_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
    inverse_label_mapping: Optional[dict] = None,
):
    """
    Train one or more classical ML classifiers, save metrics / models,
    and return a nested metrics-dictionary.
    """
    output_dir = config.output.directory
    os.makedirs(output_dir, exist_ok=True)

    cfg_algos = getattr(config.model, "algorithms", ["xgboost"])
    algorithms_to_run = _resolve_algo_names(cfg_algos)

    all_model_metrics: Dict[str, Dict] = {}

    for algo_name in algorithms_to_run:
        logger.info("\n=== Training %s ===", algo_name.upper())
        # per-algo output folder
        alg_out = os.path.join(output_dir, algo_name)
        os.makedirs(alg_out, exist_ok=True)

        model = build_ml_model(algo_name, config.model)
        if algo_name == "xgboost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else: 
            model.fit(X_train, y_train)
        logger.info("Fitted %s on %d samples.", algo_name, len(X_train))

        metrics_out: Dict[str, Dict[str, float]] = {}
        for split_name, X_split, y_true in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
        ]:
            y_pred = model.predict(X_split)
            split_metrics = compute_metrics(y_true, y_pred)
            metrics_out[split_name] = split_metrics

            logger.info(
                "%s - Acc: %.4f  F1_macro: %.4f  Precision: %.4f  Recall: %.4f",
                split_name.upper(),
                split_metrics["accuracy"],
                split_metrics["f1_macro"],
                split_metrics["precision"],
                split_metrics["recall"],
            )

            if (
                split_name == "test"
                and inverse_label_mapping is not None
                and len(inverse_label_mapping) <= 50
            ):
                cm = confusion_matrix(y_true, y_pred)
                plot_confusion_matrix(
                    cm,
                    inverse_label_mapping,
                    title=f"Confusion Matrix – {algo_name.upper()}",
                    output_dir=alg_out,
                )

        # save metrics and model in alg_out
        save_metrics(metrics_out, f"{algo_name}_metrics.json", alg_out)
        model_path = os.path.join(alg_out, f"{algo_name}_model.pkl")
        joblib.dump(model, model_path)
        logger.info("Saved model → %s", model_path)

        all_model_metrics[algo_name] = metrics_out

    # summary directory for comparison plots, now placed under each algo folder
    summary_dir = os.path.join(alg_out, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision", "recall"]:
        plot_metric_comparison(all_model_metrics, metric, summary_dir)


    return all_model_metrics

