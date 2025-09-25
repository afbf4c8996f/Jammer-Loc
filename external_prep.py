# external_prep.py

import pandas as pd
import numpy as np
import logging
from data_utils import (
    load_csv_files,
    remove_multi_label_times,
    apply_config_based_preprocessing,
    merge_positions_with_df,
    extract_cir_features
)

from utils import LEAKY_COLS

logger = logging.getLogger(__name__)


def prepare_external_dataset(
    cfg,               # your Config object
    data_path: str,    # path to external CSV directory 
    coords_xlsx: str,  # Positions.xlsx
    exclude_positions: list,
    inv_map: dict,     # only for classification label remap
    scaler,            # scaler_cls (ML‐clf) or std_scaler (ML‐/DL‐reg)
    feature_cols: list
):
    """
    Returns X_ext, y_ext ready for .predict() or .evaluate():
      - classification: y_ext is 1D int labels
      - regression:    y_ext is Nx2 array [x_cm, y_cm]
    """
    # 1) load & time‐clean
    df_ext, _ = load_csv_files(
        data_path,
        parse_cir=cfg.data.parse_cir
    )
    if df_ext.empty:
        logger.error("No external data at %s", data_path)
        return None, None
    
    # 2) Remove multi-labeled time stamps if any
    df_ext = remove_multi_label_times(df_ext, "date_time", "label")

    
    # 3) extract CIR features only if we parsed them extract all the CIR features (delay, band‐power, etc.) and drop raw cir_cmplx

    if cfg.data.parse_cir:
        df_ext = extract_cir_features(df_ext)


    # 4) regression only: bring in the true coords
    if cfg.model.task == "regression":
        df_ext = merge_positions_with_df(
            df_ext, coords_xlsx,
            excluded_positions=exclude_positions,
            merge_how="inner"
        )

    # 5) lowercase & config-based preprocess
    df_ext.columns = df_ext.columns.str.lower()
    df_ext = apply_config_based_preprocessing([df_ext], cfg)[0]

    # 6) drop any leftover LEAKY_COLS
    df_ext.drop(columns=[c for c in LEAKY_COLS if c in df_ext.columns],
                inplace=True, errors="ignore")

    # 7) classification: remap to your internal label indices
    if cfg.model.task == "classification":
        df_ext["label"] = df_ext["label"].map({orig: idx for idx, orig in inv_map.items()})
        df_ext.dropna(subset=["label"], inplace=True)
        y_ext = df_ext["label"].astype(int).values
    else:
        # regression: collect ground-truth x_cm, y_cm
        y_ext = df_ext[["x_cm","y_cm"]].values

    # 8) ensure every FEATURE_COL exists
    for c in feature_cols:
        if c not in df_ext.columns:
            df_ext[c] = 0.0

    # 9) scale *only* the continuous slice, then re-attach binary/OHE flags
   

    # how many continuous dims did the scaler see?
    n_cont = scaler.n_features_in_
    # first n_cont names are the continuous features; the rest are flags
    cont_cols = feature_cols[:n_cont]
    bin_cols  = feature_cols[n_cont:]

    logger.info(
        f"Scaling external: {n_cont} continuous dims, "
        f"{len(bin_cols)} binary flags → {len(feature_cols)} total"
    )

    # transform continuous ones
    X_cont = scaler.transform(df_ext[cont_cols].values)
    # grab the binary/OHE columns untouched
    X_bin = df_ext[bin_cols].values if bin_cols else np.empty((len(df_ext), 0))

    # recombine into the full feature matrix
    X_ext = np.hstack([X_cont, X_bin])
    

    return X_ext, y_ext
