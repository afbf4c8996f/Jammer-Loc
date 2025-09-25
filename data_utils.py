# data_utils.py
import sys
import numpy as np
import ast
import torch
from pathlib import Path
from sklearn.preprocessing import  StandardScaler
import logging 
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Optional
from copy import deepcopy
from joblib import Parallel, delayed


# Obtain a module-specific logger
logger = logging.getLogger(__name__)



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize the dataset with features and labels.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
        """
        assert len(X) == len(y), "Features and labels must have the same length!"
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def scale_data(train_X, val_X, test_X, *, cfg=None):
    """
    Fits StandardScaler on train and applies it to val / test.
    If cfg.model.scale_regression is False and we're in a regression run,
    scaling is skipped (behaves like identity).
    """
    # Log why we're scaling (or skipping) for this run
    if cfg is not None:
        if cfg.model.task == "regression" and not cfg.model.scale_regression:
            logger.info("Skipping feature scaling for regression (scale_regression=False)")
        else:
            logger.info(
                f"Applying StandardScaler to {cfg.model.task} features "
                f"(input_mode={cfg.model.input_mode})"
            )

    # If regression and scaling disabled, return identity transform
    if cfg and cfg.model.task == "regression" and not cfg.model.scale_regression:
        class _DummyScaler:
            n_features_in_ = train_X.shape[1]
            def transform(self, X): 
                return X
        return train_X, val_X, test_X, _DummyScaler()

    # Otherwise fit & apply StandardScaler
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled   = scaler.transform(val_X)
    test_X_scaled  = scaler.transform(test_X)
    return train_X_scaled, val_X_scaled, test_X_scaled, scaler


def _parse_cir(s: str) -> np.ndarray:
    # string of Python list → ndarray[complex]
    return np.array(ast.literal_eval(s), dtype=complex)

def load_csv_files(
    root_path: str,
    pattern: str = '[0-9][0-9].csv',
    delimiter: str = ';',
    engine: str = 'python',
    skipinitialspace: bool = True,
    parse_cir: Optional[bool] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSVs matching `pattern` under `root_path`,
    normalize column names, drop constant cols, dedupe,
    and optionally parse 'cir_cmplx' into complex‐ndarrays in parallel.
    """
    root = Path(root_path)
    if not root.exists():
        logger.error(f"Path '{root}' does not exist.")
        return pd.DataFrame(), []

    files = list(root.rglob(pattern))
    if not files:
        logger.error(f"No files matching '{pattern}' in '{root}'.")
        return pd.DataFrame(), []

    dfs: List[pd.DataFrame] = []
    for fp in tqdm(files, desc='Loading CSV files'):
        if not fp.is_file():
            continue

        try:
            df = pd.read_csv(
                fp,
                sep=delimiter,
                engine=engine,
                skipinitialspace=skipinitialspace
            )
        except Exception as e:
            logger.error(f"Failed reading '{fp}': {e}")
            continue

        # Normalize column names
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'[\s\-()]', '_', regex=True)
              .str.replace(r'_+', '_', regex=True)
              .str.strip('_')
        )

        # Extract label & receiver_id
        try:
            df['label']       = int(fp.stem)
            df['receiver_id'] = fp.parent.name
        except Exception:
            logger.error(f"Cannot extract label/receiver from '{fp}'. Skipping.")
            continue

        dfs.append(df)

    if not dfs:
        logger.error("No valid dataframes loaded.")
        return pd.DataFrame(), []

    # Concatenate
    data = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total records loaded: {len(data)}")

    # Drop constant columns
    const_cols = [c for c in data.columns if data[c].nunique() <= 1]
    if const_cols:
        data.drop(columns=const_cols, inplace=True)
        logger.info(f"Dropped {len(const_cols)} constant columns: {const_cols}")

    # Deduplicate ignoring receiver_id
    if 'receiver_id' in data.columns:
        before = len(data)
        subset = [c for c in data.columns if c != 'receiver_id']
        data = data.drop_duplicates(subset=subset, keep='first')
        logger.info(f"Dropped {before - len(data)} duplicates ignoring receiver_id.")

    # Require parse_cir flag to be explicit
    if parse_cir is None:
        raise ValueError(
            "load_csv_files(): `parse_cir` must be explicitly set "
            "(e.g. parse_cir=cfg.data.parse_cir)"
        )

    # Optional parallel CIR parsing
    if parse_cir and 'cir_cmplx' in data.columns:
        parsed = Parallel(n_jobs=-1)(
            delayed(_parse_cir)(cell) for cell in data['cir_cmplx']
        )
        data['cir_cmplx'] = parsed

    # leave cir_cmplx in place for feature extraction if parse_cir=True, otherwise make sure it is included in the list
    for col in ['payload','cir_cmplx']:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column '{col}'")
    return data, const_cols




def remove_multi_label_times(df: pd.DataFrame, date_col='date_time', label_col='label') -> pd.DataFrame:
    """
    Remove any date_times that have multiple distinct labels.
    """
    group_label_counts = df.groupby(date_col)[label_col].nunique()
    multi_label_times = group_label_counts[group_label_counts > 1].index

    df_clean = df[~df[date_col].isin(multi_label_times)].copy()
    logger.info(f"Removed {len(multi_label_times)} multi-labeled timestamps. "
                f"Original shape={df.shape}, new shape={df_clean.shape}")
    return df_clean

def group_stratified_split_by_time_fixed(
    df: pd.DataFrame,
    date_col: str,
    label_col: str,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    random_seed=42
) -> pd.DataFrame:
    """
    Returns df with 'split' in {train, val, test}, ensuring no leftover date_times.
    Expects each date_time has exactly 1 label.
    """

    # --- Step 1: Create unique_times_df
    unique_times_df = df.groupby(date_col)[label_col].first().reset_index()

    # --- Step 2: Split by label
    label_to_times = {}
    for label_value in unique_times_df[label_col].unique():
        label_mask = (unique_times_df[label_col] == label_value)
        these_times = unique_times_df.loc[label_mask, date_col].values
        label_to_times[label_value] = these_times

    rng = np.random.default_rng(random_seed)

    train_times = set()
    val_times   = set()
    test_times  = set()

    # --- Step 3: For each label, shuffle + slice
    for label_value, times_array in label_to_times.items():
        rng.shuffle(times_array)
        n = len(times_array)

        # a) Round or int
        n_train = int(round(train_ratio * n))
        n_val   = int(round(val_ratio * n))
        n_test  = n - n_train - n_val  # leftover

        train_part = times_array[:n_train]
        val_part   = times_array[n_train : n_train + n_val]
        test_part  = times_array[n_train + n_val : n_train + n_val + n_test]
        # The last slice ensures no leftover remain unassigned

        train_times.update(train_part)
        val_times.update(val_part)
        test_times.update(test_part)

    # --- Step 4: Assign rows to partitions
    df['split'] = None
    df.loc[df[date_col].isin(train_times), 'split'] = 'train'
    df.loc[df[date_col].isin(val_times),   'split'] = 'val'
    df.loc[df[date_col].isin(test_times),  'split'] = 'test'

    return df



# ────────────────────────────────────────────────────────────────────


def apply_config_based_preprocessing(
    df_list: List[pd.DataFrame],
    cfg
) -> List[pd.DataFrame]:
    """
    Returns new DataFrames with preprocessing applied according to cfg.model:
      - Deduplicate (ignoring receiver_id) and log drops
      - Optional rto encoding (classification only)
      - Compute rate features exactly once when input_mode == 'rate'
      - Drop common and rate-specific columns
      - Handle receiver_id (one-hot or drop)

    All column names are assumed lowercase (loader has normalized them).
    """
    mode = cfg.model.input_mode    # 'actual' or 'rate'
    use_receiver     = getattr(cfg.model, 'use_receiver_id', False)
    one_hot_receiver = getattr(cfg.model, 'one_hot_receiver_id', False)
    encode_rto       = getattr(cfg.model, 'encode_event_rto', False)


    # Columns to drop in all modes ,
    common_drop = [
        'date_time', 'split','payload_data_tx_timestamp_ms',
        'payload_data_idx',
        'payload_mhr_seq_num',
        'event_cnt_rto',  
        'payload_fcs',     # remove drifitng feature
        'event_cnt_sfdto', # remove SFD timeouts (drifting feature)
        'event_cnt_prej', 
        'ipatovfpindex',   # it's an index 
        'dgc'              # the gain control irrelevant 
    ]



    processed = []
    for df in df_list:
        df = df.copy()

        # 0) Deduplicate ignoring receiver_id
        n0 = len(df)
        if 'receiver_id' in df.columns:
            subset = [c for c in df.columns if c != 'receiver_id']
        else:
            subset = df.columns.tolist()
        df = df.drop_duplicates(subset=subset)
        dropped = n0 - len(df)
        if dropped:
            pct = dropped / n0 * 100
            logger.info(f"Dropped {dropped} duplicate rows ({pct:.2f}%) from split.")

        # 1) event_cnt_RTO encoding (classification only)
        if encode_rto and 'event_cnt_rto' in df.columns:
            df['event_cnt_rto'] = df['event_cnt_rto'].replace({0: 0, 94: 1})

        # 2) Compute rate features exactly once (needs date_time & receiver_id intact)
        if mode == 'rate':
            df = compute_rate_features(df)

        # 3) Drop common columns (including date_time) now that rate features exist
        drops = [c for c in common_drop if c in df.columns]
        if drops:
            df.drop(columns=drops, inplace=True)
        
        # 4) Drop rate-specific extras when in rate mode
        # if mode == 'rate':
        #   drops = [c for c in rate_extra_drop if c in df.columns]
        #    if drops:
        #        df.drop(columns=drops, inplace=True)

        # 5) reciever_id handling (one-hot or drop) after all rate & common columns are processed
        if one_hot_receiver and 'receiver_id' in df.columns:
            dummies = pd.get_dummies(df['receiver_id'], prefix='recid', drop_first=False)
            df.drop(columns=['receiver_id'], inplace=True)
            df = pd.concat([df, dummies], axis=1)
        elif not use_receiver and 'receiver_id' in df.columns:
            df.drop(columns=['receiver_id'], inplace=True)

        processed.append(df)

    return processed

# ────────────────────────────────────────────────────────────────────


def shuffle_dataframes(df_list: List[pd.DataFrame], random_state: int = 42) -> List[pd.DataFrame]:
    """
    Shuffle each DataFrame in the list and return the new list.
    """
    return [
        df_sub.sample(frac=1, random_state=random_state)
              .reset_index(drop=True)
        for df_sub in df_list
    ]


def check_num_classes(config, detected_num_classes: int):
    """
    Checks if config.model.num_classes is None or mismatched, handle or exit.
    """
    if config.model.num_classes is None:
        config.model.num_classes = detected_num_classes
        logging.info(f"Auto-detected num_classes: {config.model.num_classes}")
    else:
        if config.model.num_classes != detected_num_classes:
            logging.error(
                f"Mismatch: config.model.num_classes={config.model.num_classes}, but data has {detected_num_classes} classes."
            )
            sys.exit(1)


def check_input_size(config, actual_features_count: int):
    """
    Checks if config.model.input_size is None or mismatched.
    If None, auto-sets it to actual_features_count.
    Otherwise, if mismatch, logs error and exits.
    """

    
    if config.model.input_size is None:
        config.model.input_size = actual_features_count
        logging.info(f"Auto-detected input_size={actual_features_count}")
    else:
        if config.model.input_size != actual_features_count: # intentionally not catching, pytorch errors more informative
            logging.warning(
                f"Config input_size={config.model.input_size} does not match "
                f"actual feature count={actual_features_count}. "
                f"Overriding to use {actual_features_count}."
            )
            config.model.input_size = actual_features_count


            


def compute_rate_features(
    df: pd.DataFrame,
    time_col: str = 'date_time',
    label_col: str = 'label',
    receiver_col: str = 'receiver_id',
    event_cnt_RTO: str = 'event_cnt_rto',
    time_unit: str = 'ms',
    compute_diff: bool = False,
    compute_rate: bool = True,
    drop_original_features: bool = True,
    drop_first_row_per_group: bool = True,
    drop_dt_zero: bool = True,
    drop_time_col: bool = False
) -> pd.DataFrame:
    """
    Computes dt_<time_unit>, optional diff_<feature>, and optional rate_<feature>
    for numeric columns (excluding label_col, receiver_col, and time_col).
    Allows toggling diff and rate computations independently.

    Steps:
      1. Identify numeric columns excluding label_col, receiver_col, time_col.
      2. Sort by [label_col, receiver_col, time_col].
      3. Group by both label_col and receiver_col and compute:
         dt = (time[t] - time[t-1]) in ms or s.
         diff_feat = (feat[t] - feat[t-1]) if compute_diff=True.
         rate_feat = diff_feat / dt           if compute_rate=True.
      4. Optionally drop rows where dt==0 (avoids infinite rates).
      5. Optionally drop original features (drop_original_features).
      6. Optionally drop the first row per group (which has NaNs in dt or diff/rate).
      7. Optionally drop the time column (drop_time_col).

    Args:
        df (pd.DataFrame): DataFrame with at least time_col, label_col, receiver_col.
        time_col (str): Name of the timestamp column.
        label_col (str): Name of the label column.
        receiver_col (str): Name of the receiver ID column to group by.
        event_cnt_RTO (str): Column name of the event counter to exclude.
        time_unit (str): 'ms' for milliseconds, 's' for seconds.
        compute_diff (bool): Whether to create diff_<feature> columns.
        compute_rate (bool): Whether to create rate_<feature> columns.
        drop_original_features (bool): If True, remove the original numeric feature columns.
        drop_first_row_per_group (bool): If True, drop each group's first row of NaNs.
        drop_dt_zero (bool): If True, drop rows where dt_<time_unit> == 0.
        drop_time_col (bool): If True, drop the time_col column after processing.

    Returns:
        pd.DataFrame: Transformed DataFrame with dt_<time_unit> and diff_/rate_ columns.
    """

    df = df.copy()
    # Normalize Receiver_ID to lowercase so grouping works
    if 'Receiver_ID' in df.columns and 'receiver_id' not in df.columns:
        df.rename(columns={'Receiver_ID': 'receiver_id'}, inplace=True)

    # Identify numeric features except label_col, receiver_col, time_col
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not c.startswith("diff_")
  and not c.startswith("cir_")]
    features_to_transform = [
        c for c in numeric_cols
        if c not in [label_col, receiver_col, event_cnt_RTO, # double check this!
                     "x_cm", "y_cm"]
        and c != time_col
    ]

    # Ensure time_col is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])

    # Sort by label, receiver, then time
    df.sort_values(by=[label_col, receiver_col, time_col], inplace=True)
    # Group by both label and receiver so streams remain separate
    grouped = df.groupby([label_col, receiver_col])

    # Create dt_<time_unit>
    dt_col_name = f'dt_{time_unit}'
    if time_unit == 'ms':
        df[dt_col_name] = grouped[time_col].diff().dt.total_seconds() * 1000
    else:
        df[dt_col_name] = grouped[time_col].diff().dt.total_seconds()

    # Compute diffs and/or rates
    for feat in features_to_transform:
        diff_col = f'diff_{feat}'
        rate_col = f'rate_{feat}'

        if compute_diff or compute_rate:
            df[diff_col] = grouped[feat].diff()
        else:
            continue

        if compute_rate:
            df[rate_col] = df[diff_col] / df[dt_col_name]
        else:
            df.drop(columns=[rate_col], inplace=True, errors='ignore')

        if not compute_diff:
            df.drop(columns=[diff_col], inplace=True, errors='ignore')

    # Drop rows with dt == 0
    if drop_dt_zero:
        zero_mask = df[dt_col_name] == 0
        n_zero = zero_mask.sum()
        if n_zero > 0:
            logger.info("Dropping %d rows with %s == 0.", n_zero, dt_col_name)
            df = df[~zero_mask].copy()

    # Drop original features
    if drop_original_features and features_to_transform:
        df.drop(columns=features_to_transform, inplace=True, errors='ignore')

    # Drop first row per group (NaNs)
    if drop_first_row_per_group:
        mask_first_row = df[dt_col_name].isna()
        n_first = mask_first_row.sum()
        if n_first > 0:
            logger.info(
                "Dropping %d first-row-per-group entries (NaN %s).",
                n_first, dt_col_name
            )
            df = df[~mask_first_row].copy()

    # Optionally drop time column
    if drop_time_col and time_col in df.columns:
        df.drop(columns=[time_col], inplace=True)

    # Ensure time_col still present for train/val split
    assert time_col in df.columns, (
        f"{time_col} Disappeared inside compute_rate_features; "
        "this column is required for the train/val/test split."
    )

    return df


def merge_positions_with_df(
    df_main: pd.DataFrame,
    coords_xlsx_path: str,
    label_col: str = 'label',
    x_col: str = 'x_cm',
    y_col: str = 'y_cm',
    excluded_positions:  Optional[List[str]] = None, 
    merge_how: str = "left"
) -> pd.DataFrame:
    """
    Left-join X,Y coordinates onto df_main[label_col].
    Column names are canonicalised to lower case.
    """
    logger = logging.getLogger(__name__)

    # 1) Load the coordinates sheet
    df_coords = pd.read_excel(coords_xlsx_path)

    # 1) Normalize headers: lowercase, strip parentheses, spaces/dashes → underscores
    # 2) Normalize headers:
    #   • lowercase & strip whitespace
    #   • convert any "(cm)" suffix into "_cm"
    #   • remove any other parentheses
    #   • replace spaces/dashes with underscores
    df_coords.columns = (
        df_coords.columns
            .str.strip()
            .str.lower()
            .str.replace(r"\s*\(cm\)", "_cm", regex=True)     # X (cm) → x_cm
            .str.replace(r"\s*\(.*?\)", "", regex=True)        # remove other (...)
            .str.replace(r"[\s\-]+", "_", regex=True)          # spaces/dashes → _
            .str.strip("_")
    )

    # 2) Identify the position column dynamically:
    pos_candidates = [c for c in df_coords.columns if "position" in c]
    if not pos_candidates:
        raise KeyError(f"No column matching 'position' in {coords_xlsx_path}: found {df_coords.columns.tolist()}")
    if len(pos_candidates) > 1:
        # e.g. both 'position' and 'position_cm'—prefer exact match
        if "position" in pos_candidates:
            pos_col = "position"
        else:
            pos_col = pos_candidates[0]
    else:
        pos_col = pos_candidates[0]

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
        
    # 4) Merge
    merged = df_main.merge(
        df_coords[[pos_col, x_col, y_col]],
        how=merge_how,
        left_on=label_col, right_on=pos_col
    )
    # 5) Drop the extra column
    merged.drop(columns=[pos_col], inplace=True, errors="ignore")
    return merged


def compute_delay_spread(
    df: pd.DataFrame,
    cir_col: str,
    time_bin_width: float
) -> pd.DataFrame:
    """
    Compute the mean delay (μ) and RMS delay spread (σ) from a complex CIR vector.

    Args:
        df:             Input DataFrame containing one column of complex‐tap arrays.
        cir_col:        Name of the column holding each row’s array of complex taps.
        time_bin_width: Delay between successive taps, in seconds (e.g. ~1.0016e-9 for DW3000).

    Returns:
        A new DataFrame with two added real‐valued columns:
         - 'cir_mean_delay' : the power‐weighted average tap delay (seconds)
         - 'cir_rms_delay'  : the RMS delay spread around that mean (seconds)
    """
    # Sanity‐check inputs
    assert time_bin_width > 0, f"Expected positive time_bin_width, got {time_bin_width}"
    # 1) Stack the per‐row arrays into (N, M)
    tap_matrix = np.vstack(df[cir_col].values)
    assert tap_matrix.ndim == 2, f"Expected tap_matrix 2D, got shape {tap_matrix.shape}"

    # 2) Compute per‐tap power
    power = np.abs(tap_matrix) ** 2                        # (N, M)

    # 3) Build the delay axis in seconds
    M    = tap_matrix.shape[1]
    taps = np.arange(M) * time_bin_width                   # (M,)

    # 4) Total power per row, avoid zeros
    total_power = power.sum(axis=1)                        # (N,)
    safe_total  = np.where(total_power == 0, 1e-12, total_power)

    # 5) Mean delay (μ = Σ p_i t_i / Σ p_i)
    mean_delay = (power * taps).sum(axis=1) / safe_total   # (N,)

    # 6) RMS delay spread (σ = sqrt( Σ p_i (t_i–μ)² / Σ p_i ))
    diff      = taps[None, :] - mean_delay[:, None]        # (N, M)
    rms_delay = np.sqrt((power * diff**2).sum(axis=1) / safe_total)

    # 7) Return new DataFrame
    df_out = df.copy()
    df_out['cir_mean_delay'] = mean_delay
    df_out['cir_rms_delay']  = rms_delay
    return df_out




def compute_band_power_ratio(
    df: pd.DataFrame,
    cir_col: str,
    cutoff_frequency_hz: float,
    time_bin_width: float,
    n_fft: Optional[int] = None
) -> pd.DataFrame:
    """
    For each row, take the complex‐valued tap vector in `cir_col`,
    FFT it (to length n_fft or #taps), compute the PSD, then sum
    'low' = everything below cutoff_frequency_hz, and 'high' = everything above.

    Args:
        df:                   Input DataFrame with a column `cir_col` of length-M arrays.
        cir_col:              Name of the column holding each row’s array of complex taps.
        cutoff_frequency_hz:  Physical frequency (Hz) at which to split low vs. high power.
        time_bin_width:       Sampling interval between taps (seconds).
        n_fft:                FFT length; defaults to M (the number of taps).

    Returns:
        A new DataFrame with three added real‐valued columns:
          • cir_low_power
          • cir_high_power
          • cir_band_power_ratio
    """
    # 1) Stack the per‐row arrays into shape (n_samples, M)
    tap_matrix = np.vstack(df[cir_col].values)       # → (N_samples, M)
    M = tap_matrix.shape[1]
    nfft = n_fft or M

    # 2) FFT along the tap‐dimension
    H   = np.fft.fft(tap_matrix, n=nfft, axis=1)     # → (N_samples, nfft) complex
    psd = np.abs(H) ** 2                             # power spectral density

    # 3) Determine which bin corresponds to cutoff_frequency_hz
    #    Sampling rate = 1 / Δt
    fs = 1.0 / time_bin_width                        # Hz
    #    Frequency resolution per bin:
    freq_res = fs / nfft                             # Hz/bin
    #    Compute bin index (floor ensures we include all below cutoff)
    cutoff_bin = int(np.floor(cutoff_frequency_hz / freq_res))
    #    Clip to valid range [0, nfft]
    cutoff_bin = max(0, min(cutoff_bin, nfft))

    # 4) Sum full‐spectrum bins, split at computed cutoff_bin
    low  = psd[:, :cutoff_bin].sum(axis=1)
    high = psd[:, cutoff_bin:].sum(axis=1)

    # 5) Avoid zero‐division
    high_safe = np.where(high == 0, 1e-12, high)
    ratio     = low / high_safe

    # 6) Return new DataFrame with added columns
    df_out = df.copy()
    df_out['cir_low_power']        = low
    df_out['cir_high_power']       = high
    df_out['cir_band_power_ratio'] = ratio
    return df_out




def extract_cir_features(df: pd.DataFrame) -> pd.DataFrame:
    #  delay‐spread & RMS
    df = compute_delay_spread(df, cir_col="cir_cmplx", time_bin_width=1.0016e-9)
    #  band‐power ratio
    df = compute_band_power_ratio(df, cir_col="cir_cmplx", cutoff_frequency_hz=1e6,time_bin_width=1.0016e-9)
    
    return df.drop(columns=["cir_cmplx"])