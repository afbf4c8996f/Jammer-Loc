#!/usr/bin/env python3
"""
Train an XGBoost regressor on the Euclidean distance target and compute five feature metrics:
  - SHAP mean absolute values
  - XGBoost built-in feature importances
  - Mutual information
  - Eta-correlation (correlation ratio, with automatic binning)
  - Mean rank across the above four metrics

Usage:
    python xgb_explainers.py --input_pq path/to/data.parquet
"""
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

# Your helper for time-stratified split
from data_utils import group_stratified_split_by_time_fixed

def correlation_ratio(feature: pd.Series, target: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the correlation ratio (eta) between a feature and a continuous target.
    If the feature has many unique values, it is binned into `n_bins` quantiles first.
    """
    if feature.nunique() > n_bins:
        binned = pd.qcut(feature, q=n_bins, duplicates='drop')
        cats = binned.cat.codes.values
    else:
        cats = pd.Categorical(feature).codes

    y = target
    y_mean = y.mean()
    ss_between = sum(
        (cats == cat).sum() * (y[cats == cat].mean() - y_mean) ** 2
        for cat in np.unique(cats)
    )
    ss_total = ((y - y_mean) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total != 0 else np.nan

def main():
    parser = argparse.ArgumentParser(
        description="Compute XGBoost explainers and ranking on Euclidean distance target"
    )
    parser.add_argument(
        "--input_pq", type=str, required=True,
        help="Path to input Parquet file containing x_cm, y_cm, date_time, etc."
    )
    args = parser.parse_args()

    # Columns to drop (metadata/drifting features). 'event_cnt_prej' is retained.
    common_drop = [
        'split',
        'payload_data_tx_timestamp_ms',
        'payload_data_idx',
        'payload_mhr_seq_num',
        'event_cnt_rto',
        'payload_fcs',
        'event_cnt_sfdto',
        'dgc',
        'ipatovfpindex',
    ]

    # 1) Load
    df = pd.read_parquet(args.input_pq)

    # 2) Sanity-check required cols
    for col in ('x_cm', 'y_cm', 'date_time'):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not in input data")

    # 3) Build target = distance from origin
    df['distance'] = np.sqrt(df['x_cm']**2 + df['y_cm']**2)

    # 4) Time-stratified 60/20/20 split by 'date_time' on the new 'distance'
    df_split = group_stratified_split_by_time_fixed(
        df, 'date_time', 'distance', 0.6, 0.2, 0.2, 42
    )
    df_train = df_split[df_split.split == 'train']
    df_val   = df_split[df_split.split == 'val']    # reserved but not used here
    df_test  = df_split[df_split.split == 'test']

    # 5) Prepare: drop unwanted columns, keep only numeric features
    def prepare(df_sub: pd.DataFrame):
        y = df_sub['distance'].values
        drop_list = common_drop + ['date_time', 'distance', 'x_cm', 'y_cm', 'label']
        feats = df_sub.drop(columns=[c for c in drop_list if c in df_sub.columns], errors='ignore')
        feats = feats.select_dtypes(include=[np.number])
        return feats.values, y, feats.columns.tolist()

    X_train, y_train, feature_names = prepare(df_train)
    X_test,  y_test,  _             = prepare(df_test)

    # 6) Scale features
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 7) Train XGBoost with tuned hyperparameters and full threading
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.645614570099021,
        max_depth=12,
        n_estimators=100,
        reg_alpha=0.0032112643094417484,
        reg_lambda=1.8007140198129195e-07,
        subsample=0.7159725093210578,
        learning_rate=0.1,
        tree_method='hist',
        n_jobs = -1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train_s, y_train)

    # 8) Compute explainers on test set
    explainer     = shap.Explainer(model)
    shap_out      = explainer(X_test_s)
    shap_vals     = shap_out.values
    shap_mean_abs = np.mean(np.abs(shap_vals), axis=0)

    xgb_importances = model.feature_importances_
    mi              = mutual_info_regression(X_test_s, y_test, random_state=42)

    df_test_s = pd.DataFrame(X_test_s, columns=feature_names)
    eta_vals  = [correlation_ratio(df_test_s[col], y_test) for col in feature_names]

    # 9) Assemble results DataFrame
    results = pd.DataFrame({
        'shap_mean_abs':   shap_mean_abs,
        'xgb_importance':  xgb_importances,
        'mutual_info':     mi,
        'eta_correlation': eta_vals
    }, index=feature_names)
    results.index.name = 'feature'

    # 10) Compute mean rank (lower = more important)
    ranks = results.rank(ascending=False, method='min')
    results['mean_rank'] = ranks.mean(axis=1)

    # Sort features by mean_rank ascending
    results = results.sort_values('mean_rank')

    # 11) Print final table
    print(results)

if __name__ == '__main__':
    main()
