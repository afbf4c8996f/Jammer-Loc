#!/usr/bin/env python3
"""
prepare_full_cir_dataset.py

Produces a fully preprocessed, per-event CIR tensor dataset for the CIR-Transformer:
  • Parses raw CSVs + CIR complex taps
  • Deduplicates timestamps
  • Merges true (x_cm, y_cm) coordinates
  • Assigns composite event_id for uniqueness
  • Verifies and drops incomplete events
  • Computes FFT→|H|, sin(arg H), cos(arg H)
  • Groups per-event, stacks into (4,3,L) tensors
  • Saves X (N_events,4,3,L), y (N_events,2), event_id, receiver_ids

Reads parameters from config.yaml and writes to ./data/<output_name>
"""
import argparse
import logging
from pathlib import Path

import numpy as np
from data_utils import load_csv_files, remove_multi_label_times, merge_positions_with_df
from config_loader import load_config

# 0) Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare full_CIR dataset for CIR-Transformer using config.yaml"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to config.yaml defining data.path, data.coords_xlsx, data.exclude_positions"
    )
    parser.add_argument(
        "--output-name", default="full_cir_dataset.npz",
        help="Filename for the output npz bundle (saved under ./data/)"
    )
    args = parser.parse_args()

    # 1) Load config values
    cfg = load_config(args.config)
    input_root  = cfg.data.path
    coords_xlsx = cfg.data.coords_xlsx
    excludes    = cfg.data.exclude_positions or []

    # 2) Prepare output directory
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / args.output_name

    # 3) Load & parse raw CIR taps
    df, _ = load_csv_files(input_root, parse_cir=True)
    logger.info("Loaded %d rows from %s", len(df), input_root)

    # 4) Remove multi-labeled timestamps
    df = remove_multi_label_times(df, date_col="date_time", label_col="label")
    logger.info("After dedupe: %d rows", len(df))

    # 5) Merge true coordinates (inner join)
    df = merge_positions_with_df(
        df,
        coords_xlsx,
        excluded_positions=excludes,
        merge_how="inner"
    )
    logger.info("After coord merge: %d rows", len(df))

    # # Assign event_id exactly as the payload sequence number
    df["event_id"] = df["payload_mhr_seq_num"]

    # 7) Verify and drop any events that didn’t get all receivers
    #    (expected_count = number of unique receivers in this dataset)
    receiver_ids = sorted(df["receiver_id"].unique(), key=lambda x: int(x))
    expected_count = len(receiver_ids)

    group_sizes = df.groupby("event_id")["receiver_id"].nunique()
    bad = group_sizes[group_sizes != expected_count]
    if not bad.empty:
        logger.warning(
            "Dropping %d events with incomplete receiver counts: %s",
            len(bad), list(bad.index)
        )
        df = df[~df["event_id"].isin(bad.index)].copy()

    # Recompute event_ids after drop
    group_sizes = df.groupby("event_id")["receiver_id"].nunique()
    event_ids = group_sizes.index.to_numpy()
    n_events = len(event_ids)
    logger.info("Perfect grouping: %d events", n_events)

    # 8) Build per-event tensors
    #    - X: (N_events, num_receivers, 3 channels, L taps)
    #    - y: (N_events, 2 coords)
    L = len(df["cir_cmplx"].iloc[0])  # number of taps per CIR

    X = np.zeros((n_events, expected_count, 3, L), dtype=float)
    y = np.zeros((n_events, 2), dtype=float)

    # Create a sorted multi-index for fast lookup
    df_indexed = (
        df.set_index(["event_id", "receiver_id"])
          .sort_index()
    )

    for i, eid in enumerate(event_ids):
        # extract all rows for this event
        for j, rid in enumerate(receiver_ids):
            cir = df_indexed.loc[(eid, rid), "cir_cmplx"]
            cir_arr = np.asarray(cir, dtype=complex)
            H = np.fft.fft(cir_arr, n=L)
            mag = np.abs(H)
            phi = np.angle(H)
            X[i, j, 0, :] = mag
            X[i, j, 1, :] = np.sin(phi)
            X[i, j, 2, :] = np.cos(phi)
        # set regression targets
        sub = df[df["event_id"] == eid]
        y[i, 0] = sub["x_cm"].iloc[0]
        y[i, 1] = sub["y_cm"].iloc[0]

    # 9) Save final data bundle
    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        event_id=event_ids,
        receiver_ids=np.array([int(r) for r in receiver_ids], dtype=int)
    )
    logger.info("Saved full_CIR dataset → %s", output_file)


if __name__ == "__main__":
    main()

