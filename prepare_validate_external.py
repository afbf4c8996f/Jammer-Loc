#!/usr/bin/env python3
"""
prepare_and_validate_external.py

1) Loads the external UWB CSV data + optional CIR parsing
2) Cleans timestamps and applies full config-based preprocessing
3) Validates that every external label has coords in Positions.xlsx
4) Merges (x_cm, y_cm) onto the external set, checks for unexpected drops or NaNs
5) Saves the final external DataFrame to Parquet for fast reloads
"""

import logging
import re
from pathlib import Path

import pandas as pd
from config_loader import load_config
from data_utils import (
    load_csv_files,
    remove_multi_label_times,
    extract_cir_features,
    
)

# ─── 0) Configure logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    force=True
)

def normalize_coords_df(df_coords: pd.DataFrame) -> pd.DataFrame:
    """Normalize Excel headers exactly like merge_positions_with_df."""
    df_coords.columns = (
        df_coords.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s*\(cm\)", "_cm", regex=True)
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.replace(r"[\s\-]+", "_", regex=True)
            .str.strip("_")
    )
    return df_coords

def validate_and_merge_external(df_ext: pd.DataFrame, coords_path: str, excluded_positions: list) -> pd.DataFrame:
    """
    Validate and merge external DataFrame with coords sheet.
    Logs any missing labels, ensures only expected rows drop,
    and asserts no NaNs in the merged coordinates.
    Returns merged_ext (with x_cm, y_cm added).
    """
    logging.info("Starting external coordinate‐merge validation…")

    # 1) Load & normalize coords sheet
    df_coords = pd.read_excel(coords_path)
    df_coords = normalize_coords_df(df_coords)
    logging.info("Coords sheet columns: %s", df_coords.columns.tolist())

    # 2) Identify position column
    pos_cols = [c for c in df_coords.columns if "position" in c]
    if not pos_cols:
        raise KeyError(f"No column containing 'position' in {coords_path}")
    pos_col = "position" if "position" in pos_cols else pos_cols[0]
    logging.info("Using '%s' as position column", pos_col)

    # 3) Identify x_cm and y_cm columns
    x_cols = [c for c in df_coords.columns if re.fullmatch(r"x.*_cm", c)]
    y_cols = [c for c in df_coords.columns if re.fullmatch(r"y.*_cm", c)]
    if not x_cols or not y_cols:
        raise KeyError(f"Could not find x_cm/y_cm in {coords_path}")
    x_col, y_col = x_cols[0], y_cols[0]
    logging.info("Using '%s' for X and '%s' for Y", x_col, y_col)

    # 4) Compute label sets
    ext_labels   = set(df_ext["label"].unique())
    coord_labels = set(df_coords[pos_col].unique())
    missing_in_coords = ext_labels - coord_labels
    if missing_in_coords:
        logging.warning("External labels missing in coords sheet: %s", sorted(missing_in_coords))
    else:
        logging.info("All external labels found in coords sheet")

    # 5) Drop excluded positions from coords
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
        labels    = set(df_ext['label'].unique())
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

    # 6) Merge
    before_count = len(df_ext)
    merged = df_ext.merge(
        df_coords[[pos_col, x_col, y_col]],
        left_on="label", right_on=pos_col,
        how="inner"
    )
    
    after_count = len(merged)
    lost = before_count - after_count
    expected_lost = sum(1 for lbl in df_ext["label"] if lbl in missing_in_coords)
    if lost != expected_lost:
        raise AssertionError(f"Unexpected row drop: lost={lost}, expected={expected_lost}")
    logging.info("Rows dropped due to missing coords: %d (expected %d)", lost, expected_lost)

    # 7) Check for NaNs in coords
    if merged[[x_col, y_col]].isna().any().any():
        nan_rows = merged[merged[[x_col, y_col]].isna().any(axis=1)].index.tolist()
        raise AssertionError(f"NaNs found in merged coords at rows: {nan_rows}")
    logging.info("No NaNs in merged coordinates")

    # 8) Drop position column and return
    return merged.drop(columns=[pos_col])


def main():
    cfg = load_config("config.yaml")

    ext_path = cfg.data.external_test_path
    if not ext_path:
        logging.error("No external_test_path set in config.yaml")
        return

    # 1) Load external CSVs with fast C engine
    df_ext, _ = load_csv_files(
        root_path=ext_path,
        engine="c",
        parse_cir=cfg.data.parse_cir
    )
    logging.info("Loaded external data: %d rows", len(df_ext))

    # 2) Remove multi-labeled timestamps
    df_ext = remove_multi_label_times(df_ext, date_col="date_time", label_col="label")
    logging.info("After timestamp cleanup: %d rows", len(df_ext))

    # 3) CIR feature extraction if enabled
    if cfg.data.parse_cir:
        logging.info("Extracting CIR features on external data…")
        df_ext = extract_cir_features(df_ext)
        logging.info("After CIR features: %d cols", df_ext.shape[1])
    else:
        logging.info("Skipping CIR feature extraction for external data")
    
    if 'cir_cmplx' in df_ext.columns:
        df_ext = df_ext.drop(columns=['cir_cmplx'])
        logging.info("Dropped the cir_cmplx column")
    

    # 4) Validate & merge coordinates
    merged_ext = validate_and_merge_external(
        df_ext,
        cfg.data.coords_xlsx,
        cfg.data.exclude_positions
    )

    # 5) Cache to Parquet
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    ext_file = out_dir / "external_processed_dataset.parquet"
    merged_ext.to_parquet(ext_file, index=False)
    logging.info("Saved external processed dataset → %s", ext_file)


if __name__ == "__main__":
    main()
