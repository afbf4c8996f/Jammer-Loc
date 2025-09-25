import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.data import DataLoader
from models import get_model
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from config_loader import load_config
from utils import ensure_output_dir, get_device, get_run_output_dir
from data_utils import (
    load_csv_files,
    remove_multi_label_times,
    group_stratified_split_by_time_fixed,
    shuffle_dataframes,
    scale_data,
    CustomDataset,
    check_num_classes,
    apply_config_based_preprocessing,
    check_input_size,
    merge_positions_with_df,
    extract_cir_features
)
from external_prep import prepare_external_dataset
from utils import LEAKY_COLS

from plotting import plot_confusion_matrix, plot_training_curves
from metrics import save_metrics, compute_metrics
from train_utils import train_and_select_best_epoch, evaluate_on_test
import train_val_ml_models as cls_ml
import train_val_regression_models as reg_ml
import joblib

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CLI override helper with validation

def _cli_override(cfg, override_str: str) -> None:
    """
    Override configuration via CLI. Keys must exist in the config.
    Format: "key1.subkey=val,key2=val"
    """
    for kv in override_str.split(","):
        if "=" not in kv:
            raise ValueError(f"Invalid override '{kv}'. Expected 'key=val'.")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        tgt = cfg
        parts = k.split('.')
        for seg in parts[:-1]:
            if not hasattr(tgt, seg):
                raise AttributeError(f"Unknown config attribute: '{seg}' in '{k}'")
            tgt = getattr(tgt, seg)
        final_attr = parts[-1]
        if not hasattr(tgt, final_attr):
            raise AttributeError(f"Unknown config attribute: '{k}'")
        # parse value
        if v.lower() in {"true", "false"}:
            val = v.lower() == "true"
        elif v.replace(".", "", 1).isdigit():
            val = float(v) if "." in v else int(v)
        else:
            val = v
        # If original is a list, wrap single val
        orig = getattr(tgt, final_attr)
        if isinstance(orig, list) and not isinstance(val, list):
            val = [val]
        setattr(tgt, final_attr, val)

# -----------------------------------------------------------------------------
def main() -> None:
    # Parse CLI
    ap = argparse.ArgumentParser("Unified pipeline")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--config_override")
    ap.add_argument("--coords_xlsx")
    ap.add_argument("--external_test_path")
    ap.add_argument("--exclude_positions")
    ap.add_argument("--tb-logdir", default=None,
                   help="Optional TensorBoard logdir for inner training curves")
    args = ap.parse_args()
    writer = SummaryWriter(args.tb_logdir) if args.tb_logdir else None

    # Load and override config
    cfg = load_config(args.config)

    if args.config_override:
        _cli_override(cfg, args.config_override)
    
                
    
    
    # Parquet paths from YAML 
    internal_pq = cfg.data.internal_parquet 
    external_pq = cfg.data.external_parquet

    # Decide scheduler mode based on the task
    scheduler_mode = "max" if cfg.model.task == "classification" else "min"

    # Logging setup
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("▶▶▶ Effective CFG: task=%s, framework=%s, algorithms=%s, regressors=%s", cfg.model.task,cfg.model.framework,cfg.model.algorithms,getattr(cfg.model, "regressors", None))


    # Output directory for this run
    out_dir = get_run_output_dir(cfg)
    ensure_output_dir(out_dir)
    cfg.output.directory = out_dir
    logger.info("Outputs → %s", out_dir)

    # Regression-specific paths
    coords_xlsx        = args.coords_xlsx or cfg.data.coords_xlsx
    external_test_path = args.external_test_path or cfg.data.external_test_path
    exclude_positions  = (
        args.exclude_positions.split(",") if args.exclude_positions
        else cfg.data.exclude_positions or []
    )
    exclude_positions = [e.strip() for e in exclude_positions if e.strip()]

    # ─── 1) Load internal dataset ───────────────────────────────────────────
    if internal_pq:
        df = pd.read_parquet(internal_pq)
        logger.info(f"Loaded internal Parquet → {internal_pq} ({len(df)} rows)")
    
        # Drop only the coordinates for classification immediately
        if cfg.model.task == "classification":
            df.drop(columns=["x_cm", "y_cm"], errors="ignore", inplace=True)
    
     # For regression we keep 'label' so split still works
    else:
        # Full CSV‐based workflow
        df, _ = load_csv_files(
            cfg.data.path,
            parse_cir=cfg.data.parse_cir
        )

        if df.empty:
             logger.error("No data loaded – abort.")
             sys.exit(1)

        # Remove multi-labeled time stamps if any
        df = remove_multi_label_times(df, "date_time", "label")

        # ── Hand-crafted feature engineering ────────────────────────────
    
        # Extract all the CIR features (delay, band‐power, etc.) and drop raw cir_cmplx
        if cfg.data.parse_cir:
            df = extract_cir_features(df)

    # ────────────────────────────────────────────────────────────────
    
    # Time-based stratified split
    df_split = group_stratified_split_by_time_fixed(
        df, "date_time", "label", 0.6, 0.2, 0.2, 42
    )
    df_train = df_split[df_split.split == "train"].copy()
    df_val   = df_split[df_split.split == "val"].copy()
    df_test  = df_split[df_split.split == "test"].copy()
    df_train, df_val, df_test = shuffle_dataframes([df_train, df_val, df_test])

    # Common preprocessing
    df_train, df_val, df_test = apply_config_based_preprocessing(
        [df_train, df_val, df_test], cfg
    )
    for df_ in (df_train, df_val, df_test):
        if 'split' in df_.columns:
            df_.drop(columns=['split'], inplace=True)
        df_.columns = df_.columns.str.lower()
        bad = [c for c in df_.columns if c!='label' and not pd.api.types.is_numeric_dtype(df_[c])]
        if bad:
            logger.warning("Dropped non-numeric columns: %s", bad)
            df_.drop(columns=bad, inplace=True)

    # Features list
    FEATURE_COLS = [c for c in df_train.columns if c!='label' and pd.api.types.is_numeric_dtype(df_train[c])]
    FEATURE_COLS = [c for c in FEATURE_COLS if c not in LEAKY_COLS]
    logger.info("[DEBUG] Features: %s", FEATURE_COLS)

    # Dynamic detection of binary vs. continuous
    bin_cols = [
        c for c in FEATURE_COLS
        if set(df_train[c].unique()).issubset({0,1})
    ]
    cont_cols = [c for c in FEATURE_COLS if c not in bin_cols]

    # (Optional) double-check feature lists for external as well
    logger.info(">>> INTERNAL cont_cols (%d): %s", len(cont_cols), cont_cols)
    logger.info(">>> INTERNAL bin_cols  (%d): %s", len(bin_cols),  bin_cols)


    # Assign labels for downstream use
    y_train, y_val, y_test = (
        df_train.label.values,
        df_val.label.values,
        df_test.label.values
    )

    # ------------------------------------------------------------------------
    # REGRESSION 
    if cfg.model.task == "regression":
        # # Merge coords & drop leaks (skip if loaded from Parquet)
        if not internal_pq:
            def _merge(df_sub):
                return merge_positions_with_df(
                    df_sub, coords_xlsx,
                    excluded_positions=exclude_positions,
                    merge_how="inner"
                )
            df_train = _merge(df_train)
            df_val   = _merge(df_val)
            df_test  = _merge(df_test)
            for d in (df_train, df_val, df_test):
                d.drop(columns=[c for c in LEAKY_COLS if c in d.columns], inplace=True, errors='ignore')

        # Remove coordinate targets from feature groups
        bin_cols_r = [c for c in bin_cols]
        cont_cols_r = [c for c in cont_cols if c not in {"x_cm","y_cm"}]
        feature_cols_r = cont_cols_r + bin_cols_r
        logger.info("[DEBUG] Regression continuous cols: %s", cont_cols_r)
        logger.info("[DEBUG] Regression binary cols: %s", bin_cols_r)
        logger.info("[DEBUG] Regression feature count: %d", len(feature_cols_r))
        logger.info("[DEBUG] Regression features: %s", feature_cols_r)

       
        # Scale only continuous for both ML& DL 
        X_tr_cont = df_train[cont_cols_r].values
        X_va_cont = df_val  [cont_cols_r].values
        X_te_cont = df_test [cont_cols_r].values
        X_tr_c_s, X_va_c_s, X_te_c_s, std_scaler = scale_data(
            X_tr_cont, X_va_cont, X_te_cont, cfg=cfg
        )

        # Log scaler parameters for reproducibility
        logger.info("Scaler n_features_in_: %d", std_scaler.n_features_in_)
        logger.info("Scaler mean_         : %s", np.array2string(std_scaler.mean_, precision=3))
        logger.info("Scaler var_          : %s", np.array2string(std_scaler.var_,  precision=3))



        df_train[cont_cols_r] = X_tr_c_s
        df_val  [cont_cols_r] = X_va_c_s
        df_test [cont_cols_r] = X_te_c_s

        # External-test via Parquet or CSV
        ext_df_proc = None
        if external_pq:
            df_ext = pd.read_parquet(external_pq)
            logger.info(f"Loaded external Parquet → {external_pq} ({len(df_ext)} rows)")
            if cfg.model.task == "classification":
                df_ext.drop(columns=["x_cm", "y_cm"], errors="ignore", inplace=True)
            else:
                df_ext.drop(columns=["label"], errors="ignore", inplace=True)
            df_ext = apply_config_based_preprocessing([df_ext], cfg)[0]
            df_ext.drop(columns=[c for c in LEAKY_COLS if c in df_ext.columns],
                        inplace=True, errors="ignore")
                # (Optional) double-check feature lists for external as well
                # Sanity-check external feature lists match

            logger.info(">>> EXTERNAL cont_cols_r (%d): %s", len(cont_cols_r), cont_cols_r)
            logger.info(">>> EXTERNAL bin_cols_r  (%d): %s", len(bin_cols_r),  bin_cols_r)

            logger.info(f"Applying StandardScaler to external continuous features "
                        f"({len(cont_cols_r)} dims, input_mode={cfg.model.input_mode})"
                         )
                     
            bad = [c for c in df_ext.columns if c!='label' and not pd.api.types.is_numeric_dtype(df_ext[c])]
            if bad:
                df_ext.drop(columns=bad, inplace=True)
            X_ext = np.hstack([std_scaler.transform(df_ext[cont_cols_r].values),
                               df_ext[bin_cols_r].values])
            
            if cfg.model.task == "classification":
                y_ext = df_ext["label"].astype(int).values
                ext_df_proc = pd.DataFrame(X_ext, columns=feature_cols_r)
                ext_df_proc["label"] = y_ext
            else:
                y_ext = df_ext[["x_cm","y_cm"]].values
                ext_df_proc = pd.DataFrame(X_ext, columns=feature_cols_r)
                coords_df = pd.DataFrame(y_ext, columns=["x_cm","y_cm"])
                ext_df_proc = pd.concat([ext_df_proc, coords_df], axis=1)
        elif external_test_path:
            X_ext, y_ext = prepare_external_dataset(
                cfg, external_test_path,
                coords_xlsx, exclude_positions,
                None, std_scaler, feature_cols_r
            )
            if X_ext is not None:
                ext_df_proc = pd.DataFrame(X_ext, columns=feature_cols_r)
                coords_df   = pd.DataFrame(y_ext,   columns=["x_cm","y_cm"])
                ext_df_proc = pd.concat([ext_df_proc, coords_df], axis=1)
       
        # ----- DL-regression -----
        if cfg.model.framework == "dl":
            device = get_device()
            dl_out = Path(out_dir) / cfg.model.name
            dl_out.mkdir(parents=True, exist_ok=True)

             # Build feature arrays: continuous (already scaled) + binary
            X_tr_r = np.hstack([df_train[cont_cols_r].values, df_train[bin_cols_r].values])
            X_val_r = np.hstack([df_val[cont_cols_r].values,   df_val[bin_cols_r].values])
            X_te_r = np.hstack([df_test[cont_cols_r].values,  df_test[bin_cols_r].values])

            # DataLoaders
            tr_ds = CustomDataset(
                torch.tensor(X_tr_r).float(),
                torch.tensor(df_train[["x_cm","y_cm"]].values).float()
            )
            va_ds = CustomDataset(
                torch.tensor(X_val_r).float(),
                torch.tensor(df_val[["x_cm","y_cm"]].values).float()
            )
            te_ds = CustomDataset(
                torch.tensor(X_te_r).float(),
                torch.tensor(df_test[["x_cm","y_cm"]].values).float()
            )
            tr_loader = DataLoader(tr_ds, batch_size=cfg.training.batch_size, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=cfg.training.batch_size, shuffle=False)
            te_loader = DataLoader(te_ds, batch_size=cfg.training.batch_size, shuffle=False)

            model = get_model(
                cfg.model.name,
                input_size=len(cont_cols_r) + len(bin_cols_r),
                hidden_size=cfg.model.hidden_size,
                num_outputs=2
            ).to(device)
            crit = nn.MSELoss()
            opt  = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
            sch  = ReduceLROnPlateau(
                opt,
                mode=scheduler_mode,
                factor=0.1,
                patience=cfg.training.patience
            )


            # train & checkpoint
            _, ep_metrics = train_and_select_best_epoch(
                model, crit, opt,
                tr_loader, va_loader, device,
                cfg.training.num_epochs, sch,
                cfg.training.patience,
                str(dl_out), cfg.model.name,
                cfg,
                task="regression",
                writer=writer
            )

            # load best
            chk = torch.load(dl_out / f"{cfg.model.name}_best_checkpoint.pth", map_location="cpu",weights_only=True)
            model.load_state_dict(chk["model_state_dict"])

            # helper
            def eval_reg(loader, split_name):
                """
                Evaluate the given DataLoader for regression and log split-specific metrics.
                Returns a dict of regression metrics.
                """
                logger.info(f"➡️  Evaluating on {split_name} set…")

                preds, trues = [], []
                model.eval()
                with torch.no_grad():
                    for Xb, yb in loader:
                        out =  model(Xb.to(device))
                        preds.append(out.detach().cpu().numpy())
                        trues.append(yb.detach().cpu().numpy())

                preds = np.vstack(preds)
                trues = np.vstack(trues)

                # Compute the same distance-based and per-axis metrics
                d = np.sqrt((preds[:,0] - trues[:,0])**2 + (preds[:,1] - trues[:,1])**2)
                metrics = {
                    "RMSE_X":     root_mean_squared_error(trues[:,0], preds[:,0]),
                    "RMSE_Y":     root_mean_squared_error(trues[:,1], preds[:,1]),
                    "MAE_X":      mean_absolute_error(trues[:,0], preds[:,0]),
                    "MAE_Y":      mean_absolute_error(trues[:,1], preds[:,1]),
                    "R2_X":       r2_score(trues[:,0], preds[:,0]),
                    "R2_Y":       r2_score(trues[:,1], preds[:,1]),
                    "mean_distance":   d.mean(),
                    "median_distance": np.median(d),
                    "p90_distance":    np.percentile(d, 90),
                    "max_distance":    d.max(),
                    f"fraction_within_{cfg.model.threshold_cm}cm":
                                    float((d <= cfg.model.threshold_cm).mean()),
                }

                logger.info(
                    f"{split_name.title()} results – "
                    f"RMSE_X={metrics['RMSE_X']:.3f}, "
                    f"RMSE_Y={metrics['RMSE_Y']:.3f}, "
                    f"mean_distance={metrics['mean_distance']:.3f}"
                )
                return metrics

            results = {
                "train": eval_reg(tr_loader,"train"),
                "val":   eval_reg(va_loader,"val"),
                "test":  eval_reg(te_loader,"test")
            }
            if ext_df_proc is not None:
                logger.info("🛫 Switching to external test set…")
                ext_ds = CustomDataset(torch.tensor(ext_df_proc[feature_cols_r].values).float(),
                                       torch.tensor(ext_df_proc[["x_cm","y_cm"]].values).float())
                ext_loader = DataLoader(ext_ds, batch_size=cfg.training.batch_size, shuffle=False)
                results["external"] = eval_reg(ext_loader,"external")

            save_metrics(results, f"{cfg.model.name}_regression_dl_metrics.json", str(dl_out))
            logger.info("DL-regression artifacts saved to %s", dl_out)
            # Close TensorBoard writer if one was created
            if writer:
                writer.close()
            return

        # ----- ML-regression -----
        reg_ml.train_regression_models(
            df_train, df_val, df_test,
             feature_cols_r, cfg, ext_df_proc
        )
        # Close TensorBoard writer if one was created
        if writer:
            writer.close()
        return
    
    # ===== CLASSIFICATION =====
    # Build a mapping from *all* labels in train/val/test
    all_labels = np.concatenate([
        df_train.label.values,
        df_val  .label.values,
        df_test .label.values
    ])
    # Build forward (original → idx) and inverse (idx → original) maps
    unique_labels = np.unique(all_labels)
    label_map   = { label: idx for idx, label in enumerate(unique_labels) }
    inv_map     = { idx: label for idx, label in enumerate(unique_labels) }

    # Map each split, guarding against unseen labels
    y_train = np.array([label_map[l] for l in df_train.label.values])
    y_val   = df_val.label.map(label_map)
    y_test  = df_test.label.map(label_map)

    if y_val.isna().any() or y_test.isna().any():
        bad = set(df_val.label[y_val.isna()]) | set(df_test.label[y_test.isna()])
        raise RuntimeError(f"Found unseen labels in val/test: {bad!r}")

    # Now safely cast to ints
    y_val  = y_val.astype(int).values
    y_test = y_test.astype(int).values
    # Tell our code how many classes we have
    check_num_classes(cfg, len(unique_labels))
  

    # Scale only continuous
    X_tr_cont = df_train[cont_cols].values
    X_va_cont = df_val  [cont_cols].values
    X_te_cont = df_test [cont_cols].values
    X_tr_c_s, X_va_c_s, X_te_c_s, scaler_cls = scale_data(
        X_tr_cont, X_va_cont, X_te_cont, cfg=cfg
    )

    # ── Sanity‐check scaler on internal data ─────────────────────────────
    logger.info("Classifier scaler n_features_in_: %d", scaler_cls.n_features_in_)
    logger.info(
        "Classifier scaler mean_: %s",
        np.array2string(scaler_cls.mean_, precision=3, separator=",")
    )
    logger.info(
        "Classifier scaler var_: %s",
        np.array2string(scaler_cls.var_, precision=3, separator=",")
    )


    # Concatenate continuous + binary
    X_train_cls = np.hstack([X_tr_c_s, df_train[bin_cols].values])
    X_val_cls   = np.hstack([X_va_c_s, df_val  [bin_cols].values])
    X_test_cls  = np.hstack([X_te_c_s, df_test [bin_cols].values])

    # ===== ML classification =====

    if cfg.model.framework == "ml":
        X_ext_cls, y_ext_cls = None, None
        if external_test_path:
            X_ext_cls, y_ext_cls = prepare_external_dataset(
                cfg, external_test_path,
                coords_xlsx, exclude_positions,
                inv_map, scaler_cls,  cont_cols + bin_cols
            )

            # ── Sanity‐check scaler on external data ───────────────────────────
            logger.info(
                "Using same scaler on EXTERNAL: n_features_in_=%d", 
                scaler_cls.n_features_in_
            )
            logger.info(
                "External scaler mean_: %s",
                np.array2string(scaler_cls.mean_, precision=3, separator=",")
            )
            logger.info(
                "External scaler var_: %s",
                np.array2string(scaler_cls.var_, precision=3, separator=",")
            )

            logger.info(">>> EXTERNAL cont_cols (%d): %s",
                            len(cont_cols), cont_cols)
            logger.info(">>> EXTERNAL bin_cols  (%d): %s",
                            len(bin_cols),  bin_cols)

        _all_metrics = cls_ml.train_ml_model(
            X_train_cls, y_train,
            X_val_cls,   y_val,
            X_test_cls,  y_test,
            cfg, inv_map
        )
        if X_ext_cls is not None:
            # same external loop as before, unchanged…
            for algo in cls_ml._resolve_algo_names(cfg.model.algorithms):
                alg_out = Path(out_dir)/algo
                model_file = alg_out/f"{algo}_model.pkl"
                if not model_file.exists(): continue
                clf = joblib.load(str(model_file))
                y_pred = clf.predict(X_ext_cls)
                metrics_ext = compute_metrics(y_ext_cls, y_pred)
                if len(inv_map)<=50:
                    cm = confusion_matrix(y_ext_cls, y_pred)
                    plot_confusion_matrix(cm, inv_map,
                                          title=f"Confusion Matrix – {algo} External",
                                          output_dir=str(alg_out))
                save_metrics(metrics_ext, f"{algo}_external_metrics.json", str(alg_out))
                logger.info("Saved external metrics for %s", algo)
        return

    # ===== DL classification =====
    device = get_device()
    check_input_size(cfg, X_train_cls.shape[1])

    dl_out = Path(out_dir)/cfg.model.name
    dl_out.mkdir(exist_ok=True)

    tr_ds = CustomDataset(torch.tensor(X_train_cls).float(), torch.tensor(y_train).long())
    va_ds = CustomDataset(torch.tensor(X_val_cls).float(), torch.tensor(y_val).long())
    te_ds = CustomDataset(torch.tensor(X_test_cls).float(), torch.tensor(y_test).long())
    tr_loader = DataLoader(tr_ds, batch_size=cfg.training.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.training.batch_size, shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=cfg.training.batch_size, shuffle=False)

    model = get_model(cfg.model.name,
                      input_size=X_train_cls.shape[1],
                      hidden_size=cfg.model.hidden_size,
                      num_classes=cfg.model.num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    sch  = ReduceLROnPlateau(
            opt,
            mode=scheduler_mode,
            factor=0.1,
            patience=cfg.training.patience
        )


    _, ep_metrics = train_and_select_best_epoch(
        model, crit, opt,
        tr_loader, va_loader, device,
        cfg.training.num_epochs, sch,
        cfg.training.patience, str(dl_out), cfg.model.name,
        cfg,
        writer=writer
    )

    chk = torch.load(dl_out/f"{cfg.model.name}_best_checkpoint.pth", map_location="cpu",weights_only=True)
    model.load_state_dict(chk["model_state_dict"])

    # internal test
    logger.info("➡️  Evaluating on test set…")
    test_loss, test_mx, cm, auc, *_ = evaluate_on_test(
        model, crit, te_loader, device, cfg.model.num_classes, split_name="test"
    )
    plot_confusion_matrix(cm, inv_map,
                          title=f"Confusion Matrix – {cfg.model.name}",
                          output_dir=str(dl_out))
    plot_training_curves(ep_metrics, cfg.model.name, str(dl_out))

    final_metrics = {
        **ep_metrics,
        "test_loss":        test_loss,
        "test_accuracy":    test_mx.get("accuracy"),
        "test_f1_macro":    test_mx.get("f1_macro"),
        "test_f1_weighted": test_mx.get("f1_weighted"),
        "test_precision":   test_mx.get("precision"),
        "test_recall":      test_mx.get("recall"),
        "overall_auc":      auc,
    }
    save_metrics(final_metrics,
                 f"{cfg.model.name}_metrics.json",
                 str(dl_out))
    
    logger.info("DL classification finished.")

    if external_test_path:
        X_ext_dl, y_ext_dl = prepare_external_dataset(
            cfg, external_test_path,
            coords_xlsx, exclude_positions,
            inv_map, scaler_cls, cont_cols + bin_cols
        )
        if X_ext_dl is not None:
            # ── Sanity‐check same scaler on external DL data ─────────────────────────
            logger.info(
                "DL‐Classifier external scaler n_features_in_=%d", 
                scaler_cls.n_features_in_
            )
            logger.info(
                "DL‐Classifier external scaler mean_       = %s",
                np.array2string(scaler_cls.mean_, precision=3, separator=",")
            )
            logger.info(
                "DL‐Classifier external scaler var_        = %s",
                np.array2string(scaler_cls.var_,  precision=3, separator=",")
            )

            ext_ds = CustomDataset(torch.tensor(X_ext_dl).float(),
                                   torch.tensor(y_ext_dl).long())
            ext_loader = DataLoader(ext_ds,
                                    batch_size=cfg.training.batch_size,
                                    shuffle=False)
            logger.info("🛫 Switching to external test set…")
            logger.info("➡️  Evaluating on external set…")
            ext_loss, ext_mx, ext_cm, ext_auc, *_ = evaluate_on_test(
                model, crit, ext_loader, device, cfg.model.num_classes, split_name="external"
            )
            plot_confusion_matrix(
                ext_cm, inv_map,
                title=f"Confusion Matrix – {cfg.model.name} External",
                output_dir=str(dl_out),
            )
            save_metrics(
                {**ext_mx, "test_loss": ext_loss, "overall_auc": ext_auc},
                f"{cfg.model.name}_external_metrics.json",
                str(dl_out),
            )
            logger.info("Saved external DL classification metrics.")

    if writer:
        writer.close()

if __name__=="__main__":
    main()

