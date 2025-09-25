#!/usr/bin/env python3
"""
prepare_and_validate_dataset.py

1) Configures logging immediately so all INFO logs appear
2) Validates CIR feature code on a toy example
3) Loads & validates the real dataset: CSV + CIR → coords merge
4) Applies full preprocessing
5) Caches the result (Parquet) and a fitted StandardScaler
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from config_loader import load_config
from data_utils import (
    load_csv_files,
    remove_multi_label_times,
    extract_cir_features,
    compute_delay_spread,
    compute_band_power_ratio,
)

# ─── 0) Configure logging before anything else ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True    # ensure we override any existing handlers
)


def validate_cir_features():
    """Sanity-check compute_delay_spread & compute_band_power_ratio on a toy CIR."""
    logging.info("=== CIR feature validation (toy example) ===")
    df_toy = pd.DataFrame({
        "cir_cmplx": [np.array([1+1j, 0+0j, 1-1j], dtype=complex)],
        "label":     [0],
        "receiver_id":[1],
        "date_time": [pd.Timestamp("2025-05-13 00:00:00")],
    })
    df1 = compute_delay_spread(df_toy, cir_col="cir_cmplx", time_bin_width=1.0016e-9)
    df2 = compute_band_power_ratio(df1, cir_col="cir_cmplx", cutoff_frequency_hz=1e6,time_bin_width=1.0016e-9)

    print(f"Mean delay    : {df1['cir_mean_delay'].iloc[0]:.3f}")
    print(f"RMS spread    : {df1['cir_rms_delay'].iloc[0]:.3f}")
    print(f"Low power     : {df2['cir_low_power'].iloc[0]:.3f}")
    print(f"High power    : {df2['cir_high_power'].iloc[0]:.3f}")
    print(f"Band power r. : {df2['cir_band_power_ratio'].iloc[0]:.3f}")
    logging.info("=== CIR feature validation complete ===\n")


def validate_merge(df_main: pd.DataFrame, coords_path: str, excluded_positions: list):
    """
    Merge df_main with the coords sheet and assert no rows lost or NaNs.
    Returns (merged_df, x_col_name, y_col_name).
    """
    logging.info("Starting coordinate-merge validation…")

    # Load & normalize coords headers
    df_coords = pd.read_excel(coords_path)
    df_coords.columns = (
        df_coords.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s*\(cm\)", "_cm", regex=True)
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.replace(r"[\s\-]+", "_", regex=True)
            .str.strip("_")
    )
    logging.info("Coords columns after normalization: %s", df_coords.columns.tolist())

    # Pick the position column
    pos_cols = [c for c in df_coords.columns if "position" in c]
    if not pos_cols:
        raise KeyError(f"No 'position' column in {coords_path}: {df_coords.columns.tolist()}")
    pos_col = "position" if "position" in pos_cols else pos_cols[0]
    logging.info("Using '%s' as position column", pos_col)

    # Pick x_cm & y_cm
    x_cols = [c for c in df_coords.columns if re.fullmatch(r"x.*_cm", c)]
    y_cols = [c for c in df_coords.columns if re.fullmatch(r"y.*_cm", c)]
    if not x_cols or not y_cols:
        raise KeyError(f"Could not find x_cm/y_cm in {coords_path}: {df_coords.columns.tolist()}")
    x_col, y_col = x_cols[0], y_cols[0]
    logging.info("Using '%s' and '%s' as coordinate columns", x_col, y_col)

    # Ensure unique position labels
    assert df_coords[pos_col].is_unique, f"Duplicates in '{pos_col}'"

    # Drop excludes
    if excluded_positions:
        before = len(df_coords)

        # Split your excludes into “pure strings” vs. the ones that look like integers
        excl_strs = [x for x in excluded_positions if not str(x).isdigit()]
        excl_nums = []
        for x in excluded_positions:
            if str(x).isdigit():
                excl_nums.append(int(x))

        # Build masks against the **original** df_coords[pos_col]:
        mask_str = df_coords[pos_col].isin(excl_strs)
        mask_num = df_coords[pos_col].isin(excl_nums)

        # Drop any row matching either mask, _without_ casting pos_col itself
        df_coords = df_coords[~(mask_str | mask_num)].copy()

        logging.info(
            "Removed %d rows from coords matching %s",
            before - len(df_coords),
            excluded_positions
        )
        labels    = set(df_main['label'].unique())
        positions = set(df_coords[pos_col].unique())

        missing = labels - positions
        if missing:
            raise ValueError(
                f"No coordinates for these labels: {sorted(missing)}"
            )

        extra = positions - labels
        if extra:
            logging.warning(
                "Coords sheet contains %d extra positions not in data: %s",
                len(extra), sorted(extra)
            )


    # Inner merge
    merged = df_main.merge(
        df_coords[[pos_col, x_col, y_col]],
        left_on="label", right_on=pos_col,
        how="inner"
    )

    lost = len(df_main) - len(merged)
    assert lost == 0, f"{lost} rows lost during merge!"
    assert merged[[x_col, y_col]].notna().all().all(), "NaNs in merged coords!"
    logging.info("Merge validation passed; %d rows total", len(merged))

    # Drop the extra position key
    return merged.drop(columns=[pos_col]), x_col, y_col


def main():
    cfg = load_config("config.yaml")
    logging.info("Loaded config.yaml (parse_cir=%s)", cfg.data.parse_cir)

    # ─── 1) Load & dedupe CSVs ──────────────────────────────────────
    df, _ = load_csv_files(cfg.data.path, parse_cir=cfg.data.parse_cir)
    logging.info("Loaded CSVs: %d rows", len(df))

    df = remove_multi_label_times(df, date_col="date_time", label_col="label")
    logging.info("After dedupe: %d rows", len(df))


    # ─── 2) CIR feature extraction ─────────────────────────────────
    if cfg.data.parse_cir:
        logging.info("Extracting CIR features…")
        df = extract_cir_features(df)
        logging.info("After CIR features: %d cols", df.shape[1])
    else:
        logging.info("Skipping CIR feature extraction")
    
    if 'cir_cmplx' in df.columns:
        df = df.drop(columns=['cir_cmplx'])
        logging.info("Dropped the cir_cmplx column")
    

    # ─── 3) Merge validation ───────────────────────────────────────
    df, x_col, y_col = validate_merge(df, cfg.data.coords_xlsx, cfg.data.exclude_positions)

    

    # ─── 5) Cache to Parquet ──────────────────────────────────────
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    parquet_file = out_dir / "processed_dataset.parquet"
    df.to_parquet(parquet_file, index=False)
    logging.info("Saved processed dataset → %s", parquet_file)


if __name__ == "__main__":
    # validate_cir_features()  # toy CIR check
    main()
