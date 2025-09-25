# xgboost_baseline_self_prune_light.py

import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) Load CIR & targets
base_outdir = Path("outputs")  # adjust as needed
internal_npz = base_outdir / "internal_baked.npz"
external_npz = base_outdir / "external_baked.npz"

data_int = np.load(internal_npz)
data_ext = np.load(external_npz)

X_int, Y_int = data_int["X"], data_int["Y"]  # (N_int, L, 3), (N_int, 2)
X_ext, Y_ext = data_ext["X"], data_ext["Y"]  # (N_ext, L, 3), (N_ext, 2)

# 2) Split external into “calib” and “hold”
idxs = np.arange(X_ext.shape[0])
calib_idxs, hold_idxs = train_test_split(idxs, test_size=0.50, random_state=0)

X_ext_calib, Y_ext_calib = X_ext[calib_idxs], Y_ext[calib_idxs]
X_ext_hold,  Y_ext_hold  = X_ext[hold_idxs],  Y_ext[hold_idxs]

def flatten_cir_all(X):
    # X: (n_samples, L, 3) → (n_samples, L*3)
    n, L, C = X.shape
    return X.reshape(n, L * C)


X_int_flat       = flatten_cir_all(X_int)
X_ext_calib_flat = flatten_cir_all(X_ext_calib)
X_ext_hold_flat  = flatten_cir_all(X_ext_hold)

X_int_sub = X_int_flat
Y_int_sub = Y_int

# 5) Build pool = subsampled internal + external_calib
X_pool = np.vstack([X_int_sub, X_ext_calib_flat])      # (n_int_sub + n_ext_calib, L)
Y_pool = np.vstack([Y_int_sub, Y_ext_calib])           # same row count, columns=2

# 6) Quick initial XGBoost to compute residuals for self‐pruning
dpool_x = xgb.DMatrix(X_pool, label=Y_pool[:, 0])
dpool_y = xgb.DMatrix(X_pool, label=Y_pool[:, 1])

common_params = {
    "tree_method": "hist", 
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 12,
    "subsample": 0.8,
    "colsample_bytree": 0.645614570099021,
    "lambda": 1.8007140198129195e-07,
    "alpha": .0032112643094417484,
    "seed": 0,
    "nthread": 36
}


# Train only 200 rounds (no early stopping yet)
bst_init_x = xgb.train(common_params, dpool_x, num_boost_round=200, verbose_eval=False)
bst_init_y = xgb.train(common_params, dpool_y, num_boost_round=200, verbose_eval=False)

# 7) Compute Euclidean residuals on the pool
preds_pool_x = bst_init_x.predict(dpool_x)
preds_pool_y = bst_init_y.predict(dpool_y)
residuals = np.sqrt((preds_pool_x - Y_pool[:, 0])**2 + (preds_pool_y - Y_pool[:, 1])**2)

threshold = np.quantile(residuals, 0.98)
keep_mask = residuals < threshold

# 8) Prune out the top 2% worst‐predicted samples
X_pruned_pool = X_pool[keep_mask]
Y_pruned_pool = Y_pool[keep_mask]

# 9) Train/val split on the pruned pool (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_pruned_pool, Y_pruned_pool, test_size=0.20, random_state=0
)

dtrain_x = xgb.DMatrix(X_train, label=y_train[:, 0])
dval_x   = xgb.DMatrix(X_val,   label=y_val[:, 0])
dtrain_y = xgb.DMatrix(X_train, label=y_train[:, 1])
dval_y   = xgb.DMatrix(X_val,   label=y_val[:, 1])

# 10) Final XGBoost with early stopping (200 rounds, ES=20)
bst_final_x = xgb.train(
    common_params,
    dtrain_x,
    num_boost_round=200,
    early_stopping_rounds=20,
    evals=[(dtrain_x, "train"), (dval_x, "val")],
    verbose_eval=False
)
bst_final_y = xgb.train(
    common_params,
    dtrain_y,
    num_boost_round=200,
    early_stopping_rounds=20,
    evals=[(dtrain_y, "train"), (dval_y, "val")],
    verbose_eval=False
)

# 11) Evaluate on hold‐out
dhold = xgb.DMatrix(X_ext_hold_flat)
preds_x = bst_final_x.predict(dhold)
preds_y = bst_final_y.predict(dhold)

true_x = Y_ext_hold[:, 0]
true_y = Y_ext_hold[:, 1]

rmse_x = np.sqrt(mean_squared_error(true_x, preds_x))
rmse_y = np.sqrt(mean_squared_error(true_y, preds_y))
mae_x  = mean_absolute_error(true_x, preds_x)
mae_y  = mean_absolute_error(true_y, preds_y)

euclid_dist = np.sqrt((preds_x - true_x)**2 + (preds_y - true_y)**2)
mean_dist   = euclid_dist.mean()
median_dist = np.median(euclid_dist)
p90_dist    = np.percentile(euclid_dist, 90)
frac_within = float((euclid_dist <= 30.0).mean())

print("XGBoost hold‐out metrics (self‐prune, magnitude only):")
print(f"  rmse_x_cm:       {rmse_x:.4f}")
print(f"  rmse_y_cm:       {rmse_y:.4f}")
print(f"  mae_x_cm:        {mae_x:.4f}")
print(f"  mae_y_cm:        {mae_y:.4f}")
print(f"  mean_distance:   {mean_dist:.4f}")
print(f"  median_distance: {median_dist:.4f}")
print(f"  p90_distance:    {p90_dist:.4f}")
print(f"  fraction_within_30cm: {frac_within:.4f}")



