import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import load_config
from data_utils import load_csv_files, merge_positions_with_df


def prepare_cir_array(df: pd.DataFrame) -> np.ndarray:
    """
    Convert DataFrame with 'cir_cmplx' to array of shape (N, L, 3):
      [magnitude, sin(phase), cos(phase)] per tap, using vectorized ops.
    """
    # Stack all CIR arrays into a 2D complex matrix (N, L)
    cir_matrix = np.vstack(df['cir_cmplx'].values)  # shape (N, L), dtype=complex
    # Compute magnitude and phase across entire matrix
    mag = np.abs(cir_matrix)                         # shape (N, L)
    ph  = np.angle(cir_matrix)                      # shape (N, L)
    # Stack channels: (N, L, 3)
    X = np.stack((mag, np.sin(ph), np.cos(ph)), axis=2)
    # sanity check 
    assert X.ndim == 3 and X.shape[2] == 3, \
        f"CIR array wrong shape: {X.shape} (expected (N, L, 3))"
    return X.astype(np.float32)


def bake_and_save(df: pd.DataFrame, coords_path: str, exclude: list, out_file: Path):
    """
    Merge positions, prepare CIR features and (x,y) coords, save to .npz.
    Returns the arrays X and Y for quick verification.
    """
    # merge coordinates
    df_merged = merge_positions_with_df(
        df, coords_path,
        excluded_positions=exclude,
        merge_how='inner'
    )
    # extract CIR features and coordinates
    X = prepare_cir_array(df_merged)
    Y = df_merged[['x_cm', 'y_cm']].values.astype(np.float32)
    # save compressed
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, X=X, Y=Y)
    return X, Y


def main():
    parser = argparse.ArgumentParser(
        description="Bake internal & external CIR datasets to .npz"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # load config
    cfg = load_config(args.config)

    # load internal
    logger.info("Loading internal data from %s", cfg.data.path)
    df_int, _ = load_csv_files(cfg.data.path, parse_cir=True)
    logger.info("Internal rows: %d", len(df_int))

    # load external
    logger.info("Loading external data from %s", cfg.data.external_test_path)
    df_ext, _ = load_csv_files(cfg.data.external_test_path, parse_cir=True)
    logger.info("External rows: %d", len(df_ext))

    # output paths
    internal_out = Path(cfg.output.directory) / 'internal_baked.npz'
    external_out = Path(cfg.output.directory) / 'external_baked.npz'

    # bake and save
    X_int, Y_int = bake_and_save(
        df_int, cfg.data.coords_xlsx, cfg.data.exclude_positions, internal_out
    )

    logger.info(f"  → Features baked: channels={['magnitude','sin_phase','cos_phase']} "
            f"(samples={X_int.shape[0]}, shape per sample={X_int.shape[1:]})")

    X_ext, Y_ext = bake_and_save(
        df_ext, cfg.data.coords_xlsx, cfg.data.exclude_positions, external_out
    )

    logger.info(f"  → Features baked: channels={['magnitude','sin_phase','cos_phase']} "
            f"(samples={X_ext.shape[0]}, shape per sample={X_ext.shape[1:]})")

    # proof: reload and verify integrity
    logger.info("Verifying internal dataset at %s", internal_out)
    data = np.load(internal_out)
    np.testing.assert_allclose(data['X'], X_int, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data['Y'], Y_int, rtol=1e-6, atol=1e-6)
    assert data['X'].shape[0] == len(df_int)
    logger.info("Internal verified: X%s, Y%s", data['X'].shape, data['Y'].shape)

    logger.info("Verifying external dataset at %s", external_out)
    data = np.load(external_out)
    np.testing.assert_allclose(data['X'], X_ext, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(data['Y'], Y_ext, rtol=1e-6, atol=1e-6)
    assert data['X'].shape[0] == len(df_ext)
    logger.info("External verified: X%s, Y%s", data['X'].shape, data['Y'].shape)

    logger.info("Baking complete and verified.")


if __name__ == "__main__":
    main()
