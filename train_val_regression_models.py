import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from metrics import save_metrics

logger = logging.getLogger(__name__)

# ─── Registry of available ML regressors ─────────────────────────────────
def _build_xgb(model_cfg):
    """
    Build an XGBRegressor using hyperparameters from your Pydantic config.
    """
    return XGBRegressor(
        n_estimators         = model_cfg.xgb_n_estimators,
        max_depth            = model_cfg.max_depth,
        learning_rate        = model_cfg.learning_rate,
        subsample            = model_cfg.subsample,
        colsample_bytree     = model_cfg.colsample_bytree,
        reg_alpha            = model_cfg.reg_alpha,
        reg_lambda           = model_cfg.reg_lambda,
        random_state         = 42,
    )

def _build_rf(model_cfg):
    """
    Build a RandomForestRegressor using hyperparameters from your Pydantic config.
    """
    return RandomForestRegressor(
        n_estimators        = model_cfg.rf_n_estimators,
        max_depth           = model_cfg.max_depth,
        min_samples_leaf    = model_cfg.min_samples_leaf,
        max_features        = model_cfg.max_features,
        random_state        = 42,
        n_jobs              = -1,
    )

ML_REGRESSOR_REGISTRY: Dict[str, callable] = {
    "xgboost": _build_xgb,
    "random_forest": _build_rf,
}


def list_available_ml_regressors() -> List[str]:
    return list(ML_REGRESSOR_REGISTRY.keys())


def _resolve_regressor_names(cfg_names: List[str]) -> List[str]:
    if "all" in cfg_names:
        resolved = list_available_ml_regressors()
    else:
        resolved = cfg_names
    unknown = [n for n in resolved if n not in ML_REGRESSOR_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown regressor(s): {unknown}. "
            f"Valid options: {list_available_ml_regressors()}"
        )
    return resolved


def train_regression_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    config,
    external_test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict]:
    """
    Train one or more ML regressors for localization.
    Saves metrics + models; returns a nested metrics dict.
    """
    # Grab your tuned ML params from the Pydantic config
    model_cfg = config.model

    out_dir = config.output.directory
    os.makedirs(out_dir, exist_ok=True)

    # Determine which algos to run
    algorithms_cfg = model_cfg.regressors or model_cfg.algorithms
    algorithms = _resolve_regressor_names(algorithms_cfg)

    thresh_cm = getattr(model_cfg, "threshold_cm", 50)
    agg_metrics: Dict[str, Dict] = {}

    for algo in algorithms:
        logger.info(f"\n───[ REGRESSION | {algo.upper()} ]────────────────────")
        alg_out = os.path.join(out_dir, algo)
        os.makedirs(alg_out, exist_ok=True)

        # Build both X and Y regressors with your tuned params
        build_fn = ML_REGRESSOR_REGISTRY[algo]
        reg_x = build_fn(model_cfg)
        reg_y = build_fn(model_cfg)

        # Train
        X_train = df_train[feature_cols].values
        
        reg_x.fit(X_train, df_train["x_cm"].values)
        reg_y.fit(X_train, df_train["y_cm"].values)
        logger.info("Fitted two regressors on %d samples.", len(X_train))

        def evaluate(df: pd.DataFrame) -> Dict[str, float]:
            X = df[feature_cols].values
            true_x = df["x_cm"].values
            true_y = df["y_cm"].values

            pred_x = reg_x.predict(X)
            pred_y = reg_y.predict(X)

            rmse_x = root_mean_squared_error(true_x, pred_x)
            rmse_y = root_mean_squared_error(true_y, pred_y)

            mae_x  = mean_absolute_error(true_x, pred_x)
            mae_y  = mean_absolute_error(true_y, pred_y)
            r2_x   = r2_score(true_x, pred_x)
            r2_y   = r2_score(true_y, pred_y)

            dist_err = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            return {
                "RMSE_X": rmse_x,
                "RMSE_Y": rmse_y,
                "MAE_X":  mae_x,
                "MAE_Y":  mae_y,
                "R2_X":   r2_x,
                "R2_Y":   r2_y,
                "mean_distance":    dist_err.mean(),
                "median_distance":  np.median(dist_err),
                "p90_distance":     np.percentile(dist_err, 90),
                "max_distance":     dist_err.max(),
                f"fraction_within_{thresh_cm}cm": float(np.mean(dist_err <= thresh_cm)),
            }

        # Compute metrics on each split
        metrics_all = {
            "train": evaluate(df_train),
            "val":   evaluate(df_val),
            "test":  evaluate(df_test),
        }

        logger.info("[DEBUG] About to save ML‐regression metrics for %s to %s", algo, alg_out)

        if external_test_df is not None:
            metrics_all["external"] = evaluate(external_test_df)

        # Save metrics + models
        fname = f"{algo}_regression_metrics.json"
        logger.info("[DEBUG] Saving metrics file '%s' into '%s'", fname, alg_out)
        save_metrics(metrics_all, fname, alg_out)
        logger.info("[DEBUG] Files now in %s: %s", alg_out, os.listdir(alg_out))
        dump(reg_x, os.path.join(alg_out, f"{algo}_model_X.pkl"))
        dump(reg_y, os.path.join(alg_out, f"{algo}_model_Y.pkl"))
        logger.info("Saved metrics and models for %s", algo)

        agg_metrics[algo] = metrics_all

    return agg_metrics
