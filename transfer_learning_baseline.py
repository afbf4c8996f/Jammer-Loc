# transfer_learning_baseline.py

#———————————————————————————————————————Imports———————————————————————————————————————
import argparse
import logging
import copy
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary and is being deprecated where possible.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module="sklearn.utils.deprecation"
)
warnings.filterwarnings(
    "ignore",
    message="n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
    module="umap.umap_"
)

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.amp as amp
from itertools import cycle
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt  # only used for saving PDFs if you decide to re‐enable plots
from sklearn.decomposition import PCA
from config_loader import load_config
from utils import ensure_output_dir, get_run_output_dir, get_device
from metrics import save_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_utils import load_csv_files, merge_positions_with_df

# ──────────────────────────────────────────────────────────────────────────────
# compute_mse_per_sample (same as before)
# ──────────────────────────────────────────────────────────────────────────────
class CIRDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]).float()

def compute_mse_per_sample(model, X, batch_size, device, num_workers=4):
    """
    Runs X through model (in eval mode) in batches,
    returns a numpy array of per-sample MSEs.
    """
    loader = DataLoader(
        CIRDataset(X),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    mses = []
    model.eval()
    with torch.no_grad():
        for Xb in loader:
            Xb = Xb.to(device)
            rec = model(Xb, noise_std=0.0)  # no noise for pruning
            per_sample_mse = ((rec - Xb) ** 2).mean(dim=(1, 2))
            mses.append(per_sample_mse.cpu().numpy())
    return np.concatenate(mses)

# ──────────────────────────────────────────────────────────────────────────────
# Model definitions (identical to original)
# ──────────────────────────────────────────────────────────────────────────────
class SeqWithNoise(nn.Module):
    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, noise_std=0.0):
        for b in self.blocks:
            x = b(x, noise_std=noise_std)
        return x

class DropPath(nn.Module):
    """Stochastic depth per sample (a.k.a. DropPath)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask

class ConvNeXtBlock1D(nn.Module):
    """ConvNeXt–style block for 1D sequences."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv   = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm     = nn.LayerNorm(dim, eps=1e-6)  # operates on (B, L, C)
        self.pwconv1  = nn.Linear(dim, 4 * dim)
        self.act      = nn.GELU()
        self.pwconv2  = nn.Linear(4 * dim, dim)
        self.gamma    = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_prob=drop_path)

    def forward(self, x, noise_std: float = 0.0):
        shortcut = x
        x = self.dwconv(x)
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        x = x.permute(0, 2, 1)              # (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x) * self.gamma
        x = x.permute(0, 2, 1)              # (B, C, L)
        x = self.drop_path(x)
        return shortcut + x                 # residual add

class ConvNeXtAutoencoder1D(nn.Module):
    """
    A 3‐stage ConvNeXt‐style autoencoder for sequences of shape (B, L, C).
    Downsamples by 2× at each of 3 stages, then upsamples back.
    """
    def __init__(
        self,
        in_ch: int,
        hidden_dim: int = 64,
        depths: list = [2, 2, 2],
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        total_blocks = sum(depths) * 2 + depths[-1]
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_iter = iter(dp_rates)

        # Encoder stem
        self.stem = nn.Conv1d(in_ch, dims[0], kernel_size=3, padding=1)

        # Encoder stages
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamps = nn.ModuleList()
        for i, num_blocks in enumerate(depths):
            blocks = [ConvNeXtBlock1D(dims[i], drop_path=next(dp_iter)) for _ in range(num_blocks)]
            self.encoder_blocks.append(SeqWithNoise(*blocks))
            if i < len(depths) - 1:
                self.encoder_downsamps.append(
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)
                )

        # Bottleneck
        self.bottleneck = SeqWithNoise(*[
            ConvNeXtBlock1D(dims[-1], drop_path=next(dp_iter))
            for _ in range(depths[-1])
        ])

        # Decoder stages
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(reversed(depths)):
            in_dim = dims[-1 - i]
            if i < len(depths) - 1:
                out_dim = dims[-2 - i]
                self.decoder_upsamples.append(
                    nn.ConvTranspose1d(
                        in_dim, out_dim,
                        kernel_size=3, stride=2,
                        padding=1, output_padding=1
                    )
                )
                block_dim = out_dim
            else:
                block_dim = in_dim

            blocks = [ConvNeXtBlock1D(block_dim, drop_path=next(dp_iter))
                      for _ in range(num_blocks)]
            self.decoder_blocks.append(SeqWithNoise(*blocks))

        # Final projection
        self.final_conv = nn.Conv1d(dims[0], in_ch, kernel_size=3, padding=1)

    def forward(self, x, noise_std: float = 0.0, return_feats: bool = False):
        """
        x: (B, L, C)
        noise_std: if >0, adds Gaussian noise to inputs instead of masking
        returns: reconstructed tensor of same shape (and optionally bottleneck feats)
        """
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        x = x.permute(0, 2, 1)  # to (B, C, L)
        x = self.stem(x)

        # Encoder
        for blk, down in zip(self.encoder_blocks, self.encoder_downsamps):
            x = blk(x, noise_std=noise_std)
            x = down(x)
        if len(self.encoder_blocks) > len(self.encoder_downsamps):
            x = self.encoder_blocks[-1](x, noise_std=noise_std)

        # Bottleneck
        feats = self.bottleneck(x, noise_std=noise_std)

        # Decoder
        x = feats
        for up, blk in zip(self.decoder_upsamples, self.decoder_blocks):
            x = up(x)
            x = blk(x, noise_std=noise_std)
        x = self.decoder_blocks[-1](x)

        # Final conv & back to (B, L, C)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)

        if return_feats:
            return x, feats.permute(0, 2, 1)
        else:
            return x

class RegHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)
        )
    def forward(self, x):
        # x: (B, L, C_feat) → pool over L
        x = x.permute(0, 2, 1)
        x = nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, C_feat)
        return self.net(x)

# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ————————————— Setup & Hyperparams —————————————
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Check required keys
    for attr in ('hidden_size', 'fine_tune_epochs'):
        if not hasattr(cfg.model, attr):
            raise KeyError(f"config.model missing required key: {attr}")

    device = get_device()
    use_amp = (device.type == "cuda")
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(
        level=logging.INFO, force=True,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("▶ Starting transfer‐learning baseline (no adversarial)")

    #————————————————————————————————————————————————————————————————
    # Research overrides (identical to original)
    WEIGHT_DECAY     = 1e-5
    WARMUP_EPOCHS    = 5
    ADAPT_EPOCHS     = 20  # not used, but keep defined
    LAMBDA_ADAPT_MAX = 0.3  # not used
    PATIENCE_ADAPT_AUC=3   # not used

    LAMBDA_FT_MAX   = 0.5   # not used (no adversarial)
    ALPHA_RECON_FT  = 0.3
    ALPHA_RECON_MIN = 0.05
    BETA_REG_FT     = 1.0

    DIAG_SUBSAMPLE = 3000
    K_HEATMAP_SAMPLES = 100
    K_SERIES       = 6
    PATIENCE       = 15     # early‐stop fine‐tune if no RMSE_cm improvement
    Noise          = 0.6

    # ————————————— Data loading + Norm ————————————————————————————————————————————————————
    base_outdir = Path(cfg.output.directory)
    out_dir     = Path(get_run_output_dir(cfg))
    ensure_output_dir(out_dir)

    internal_npz = base_outdir / "internal_baked.npz"
    external_npz = base_outdir / "external_baked.npz"
    data_int = np.load(internal_npz)
    data_ext = np.load(external_npz)

    X_int, Y_int = data_int["X"], data_int["Y"]
    X_ext, Y_ext = data_ext["X"], data_ext["Y"]

    # Compute per‐tap clamp thresholds on the raw magnitude channel (same as before)
    mag = X_int[..., 0]
    q1  = np.percentile(mag, 25, axis=0)
    q3  = np.percentile(mag, 75, axis=0)
    iqr = q3 - q1
    clamp_min = q1 - 1.5 * iqr
    clamp_max = q3 + 1.5 * iqr

    X_int[..., 0] = np.clip(X_int[..., 0], clamp_min, clamp_max)
    X_ext[..., 0] = np.clip(X_ext[..., 0], clamp_min, clamp_max)

    raw_means = X_int.mean((0, 1))
    raw_stds  = X_int.std ((0, 1)) + 1e-6
    norm_means = np.array(raw_means, dtype=np.float32)
    norm_stds  = np.array(raw_stds, dtype=np.float32)

    X_int = (X_int - norm_means[None, None, :]) / norm_stds[None, None, :]
    X_ext = (X_ext - norm_means[None, None, :]) / norm_stds[None, None, :]

    assert X_int.ndim == 3 and Y_int.ndim == 2
    assert X_ext.ndim == 3 and Y_ext.ndim == 2
    assert X_int.shape[2] == X_ext.shape[2]
    assert X_ext.shape[0] == Y_ext.shape[0]

    # ————————————— Model & Opt for Pretrain —————————————————————————————————————————
    mae = ConvNeXtAutoencoder1D(
        in_ch=X_int.shape[2],
        hidden_dim=cfg.model.hidden_size,
        depths=[2, 2, 2],
        drop_path_rate=0.1
    ).to(device)

    bottleneck_dim = cfg.model.hidden_size * 4
    reg_head = RegHead(bottleneck_dim).to(device)

    opt_mae = optim.Adam(mae.parameters(), lr=cfg.training.learning_rate, weight_decay=WEIGHT_DECAY)
    opt_reg = optim.Adam(reg_head.parameters(), lr=3e-3, weight_decay=WEIGHT_DECAY)

    # Learning‐rate schedulers for AE pretraining
    pre_warmup      = min(WARMUP_EPOCHS, cfg.training.num_epochs)
    post_pre_epochs = max(cfg.training.num_epochs - WARMUP_EPOCHS, 1)
    sched_pre = SequentialLR(
        opt_mae,
        schedulers=[
            LinearLR(opt_mae, start_factor=0.1, end_factor=1.0, total_iters=pre_warmup),
            CosineAnnealingLR(opt_mae, T_max=post_pre_epochs)
        ],
        milestones=[pre_warmup]
    )

    crit_mae = nn.MSELoss()
    scaler = amp.GradScaler() if device.type == "cuda" else None

    disable_tqdm = not sys.stdout.isatty()

    pretrain_ckpt = out_dir / "best_pretrain_baseline.pt"
    if pretrain_ckpt.exists():
        logger.info(f" Loading pretrained AE weights from {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device, weights_only=True)
        mae.load_state_dict(ckpt['model_state'])
        opt_mae.load_state_dict(ckpt['optimizer_state'])
    else:
        # ————————————— Stage 1: Pretrain AE —————————————
        best_pre, best_pre_w = float("inf"), None
        for ep in range(1, cfg.training.num_epochs + 1):
            mae.train()
            totL, totM = 0.0, 0
            loader = DataLoader(
                CIRDataset(X_int),
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=(device.type == "cuda"),
                prefetch_factor=2
            )
            for Xb in tqdm(
                loader,
                desc=f"Pretrain Ep {ep}/{cfg.training.num_epochs}",
                disable=disable_tqdm, leave=False
            ):
                Xb = Xb.to(device)
                opt_mae.zero_grad()
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    recon = mae(Xb, noise_std=Noise)
                    loss  = crit_mae(recon, Xb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt_mae)
                    nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                    scaler.step(opt_mae); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                    opt_mae.step()

                n = Xb.numel()
                totL += loss.item() * n
                totM += n

            pre_loss = totL / (totM + 1e-8)
            logger.info(f"Pre {ep}/{cfg.training.num_epochs} Loss={pre_loss:.6f} LR={opt_mae.param_groups[0]['lr']:.2e}")
            if pre_loss < best_pre:
                best_pre, best_pre_w = pre_loss, copy.deepcopy(mae.state_dict())
            sched_pre.step()

        # Restore best pretrain weights
        mae.load_state_dict(best_pre_w)
        torch.save({
            'epoch': ep,
            'model_state': best_pre_w,
            'optimizer_state': opt_mae.state_dict(),
        }, pretrain_ckpt)
        logger.info(f"Saved pretrained AE to {pretrain_ckpt}")

    # ─── AE-pruning: drop the top 2% highest–MSE samples in both datasets ───
    mae.eval()
    mse_int = compute_mse_per_sample(mae, X_int, batch_size=cfg.training.batch_size, device=device)
    mse_ext = compute_mse_per_sample(mae, X_ext, batch_size=cfg.training.batch_size, device=device)

    thresh_int = np.quantile(mse_int, 0.98)
    thresh_ext = np.quantile(mse_ext, 0.98)
    keep_int  = mse_int < thresh_int
    keep_ext  = mse_ext < thresh_ext

    X_int, Y_int = X_int[keep_int], Y_int[keep_int]
    X_ext, Y_ext = X_ext[keep_ext], Y_ext[keep_ext]

    # ————————————— Stage 2: (SKIPPED) No adversarial adaptation ———————————————————————

    # ────────────────────────────────────────────────────────────────────────────
    # Stage 3: Fine-tuning on external data (transfer learning baseline)
    # ────────────────────────────────────────────────────────────────────────────

    # 1) Freeze everything initially, then unfreeze last encoder stage + decoder + final_conv + reg_head
    for p in mae.parameters():
        p.requires_grad = False
    for p in mae.encoder_blocks[-1].parameters():
        p.requires_grad = True
    for module in [mae.decoder_upsamples, mae.decoder_blocks, mae.final_conv]:
        for p in module.parameters():
            p.requires_grad = True
    # reg_head is new, so all its params are trainable by default

    # Build metadata DataFrame for external samples
    df_meta_raw, _ = load_csv_files(cfg.data.external_test_path, parse_cir=False)
    df_meta = merge_positions_with_df(
        df_meta_raw,
        cfg.data.coords_xlsx,
        excluded_positions=cfg.data.exclude_positions,
        merge_how='inner'
    )
    df_meta = df_meta.iloc[keep_ext].reset_index(drop=True)
    assert len(df_meta) == X_ext.shape[0], (
        f"df_meta rows ({len(df_meta)}) != X_ext samples ({X_ext.shape[0]})"
    )

    labels = df_meta['label'].values
    all_idx = np.arange(X_ext.shape[0])
    calib_idxs, hold_idxs = train_test_split(
        all_idx, test_size=0.50, stratify=labels, random_state=SEED
    )

    Xc, Yc = X_ext[calib_idxs], Y_ext[calib_idxs]
    scaler_y = StandardScaler().fit(Yc)
    Yc_norm = scaler_y.transform(Yc)

    calib_loader = DataLoader(
        list(zip(Xc, Yc_norm)),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        collate_fn=lambda batch: (
            torch.from_numpy(np.stack([b[0] for b in batch])).float(),
            torch.from_numpy(np.stack([b[1] for b in batch])).float()
        )
    )

    best_reg_w   = None
    best_mae_w   = None
    best_rmse_cm = float("inf")
    no_improve_ft = 0

    alpha_schedule = np.linspace(ALPHA_RECON_FT, ALPHA_RECON_MIN, cfg.model.fine_tune_epochs)

    crit_mae = nn.MSELoss()
    crit_reg = nn.MSELoss()

    sched_ft      = CosineAnnealingLR(opt_reg, T_max=cfg.model.fine_tune_epochs)
    sched_mae_ft  = CosineAnnealingLR(opt_mae, T_max=cfg.model.fine_tune_epochs)

    for ep in range(1, cfg.model.fine_tune_epochs + 1):
        alpha     = alpha_schedule[ep - 1]

        mae.train()
        reg_head.train()

        sum_lrec_ft, sum_lreg_ft = 0.0, 0.0
        totL, cnt                = 0.0, 0

        sum_rmse_cm, sum_mae_cm  = 0.0, 0.0
        batch_count              = 0

        for Xb, yb in tqdm(
            calib_loader,
            desc=f"FT Ep {ep}/{cfg.model.fine_tune_epochs}",
            disable=disable_tqdm,
            leave=False,
            unit="batch"
        ):
            Xb, yb = Xb.to(device), yb.to(device)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                recon, feats = mae(Xb, noise_std=Noise, return_feats=True)
                lrec = crit_mae(recon, Xb); sum_lrec_ft += lrec.item()

                pred = reg_head(feats)
                lreg = crit_reg(pred, yb); sum_lreg_ft += lreg.item()

                loss = alpha * lrec + BETA_REG_FT * lreg

            opt_mae.zero_grad()
            opt_reg.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_mae); scaler.unscale_(opt_reg)
                nn.utils.clip_grad_norm_(
                    list(mae.encoder_blocks[-1].parameters()) +
                    list(mae.decoder_upsamples.parameters()) +
                    list(mae.decoder_blocks.parameters()) +
                    list(mae.final_conv.parameters()) +
                    list(reg_head.parameters()),
                    max_norm=1.0
                )
                scaler.step(opt_mae); scaler.step(opt_reg); scaler.update()
            else:
                loss.backward()
                opt_mae.step()
                opt_reg.step()

            totL += loss.item() * Xb.size(0)
            cnt  += Xb.size(0)

            # Compute true-cm errors for early stopping
            pred_np = pred.detach().cpu().numpy()
            true_np = yb.detach().cpu().numpy()
            pred_cm = scaler_y.inverse_transform(pred_np)
            true_cm = scaler_y.inverse_transform(true_np)

            dists = np.linalg.norm(pred_cm - true_cm, axis=1)
            sum_rmse_cm += dists.mean()
            sum_mae_cm  += np.abs(pred_cm - true_cm).mean()
            batch_count += 1

        ft_loss   = totL / (cnt + 1e-8)
        avg_lrec  = sum_lrec_ft / batch_count
        avg_lreg  = sum_lreg_ft / batch_count
        avg_rmse_cm = sum_rmse_cm / batch_count
        avg_mae_cm  = sum_mae_cm  / batch_count

        logger.info(
            f"FT {ep}/{cfg.model.fine_tune_epochs}  "
            f"L_rec={avg_lrec:.4f}  L_reg={avg_lreg:.4f}  "
            f"RMSE_cm={avg_rmse_cm:.2f}  MAE_cm={avg_mae_cm:.2f}  "
            f"Total={ft_loss:.4f}  α={alpha:.2f}"
        )

        if avg_rmse_cm < best_rmse_cm:
            best_rmse_cm = avg_rmse_cm
            no_improve_ft = 0
            best_reg_w    = copy.deepcopy(reg_head.state_dict())
            best_mae_w    = copy.deepcopy(mae.state_dict())
        else:
            no_improve_ft += 1
            if no_improve_ft >= PATIENCE:
                logger.info(f"Early stopping FT at epoch {ep} (RMSE_cm plateau)")
                break


        sched_ft.step()
        sched_mae_ft.step()

    # Restore best fine-tuned weights
    mae.load_state_dict(best_mae_w)
    reg_head.load_state_dict(best_reg_w)

    # ————————————— Stage 4: Hold-out Eval —————————————
    mae.eval()
    reg_head.eval()

    hold_loader = DataLoader(
        list(zip(X_ext[hold_idxs], Y_ext[hold_idxs])),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        collate_fn=lambda batch: (
            torch.from_numpy(np.stack([b[0] for b in batch])).float(),
            torch.from_numpy(np.stack([b[1] for b in batch])).float()
        )
    )

    preds_list  = []
    truths_list = []

    with torch.no_grad():
        for Xb, yb in tqdm(hold_loader, desc="Hold-out Eval", leave=False, unit="batch"):
            Xb, yb = Xb.to(device), yb.to(device)
            _, feats = mae(Xb, noise_std=0.0, return_feats=True)
            pred_norm = reg_head(feats).cpu()
            preds_list.append(pred_norm)
            truths_list.append(yb.cpu())

    preds  = torch.cat(preds_list, dim=0).numpy()
    truths = torch.cat(truths_list, dim=0).numpy()

    preds_cm = scaler_y.inverse_transform(preds)

    # Compute final metrics
    mse_x  = mean_squared_error(truths[:, 0], preds_cm[:, 0])
    rmse_x = float(np.sqrt(mse_x))
    mse_y  = mean_squared_error(truths[:, 1], preds_cm[:, 1])
    rmse_y = float(np.sqrt(mse_y))
    mae_x  = mean_absolute_error(truths[:, 0], preds_cm[:, 0])
    mae_y  = mean_absolute_error(truths[:, 1], preds_cm[:, 1])
    r2_x   = r2_score(truths[:, 0], preds_cm[:, 0])
    r2_y   = r2_score(truths[:, 1], preds_cm[:, 1])

    dists = np.linalg.norm(preds_cm - truths, axis=1)
    mean_dist   = dists.mean()
    median_dist = np.median(dists)
    p90_dist    = np.percentile(dists, 90)
    frac_within = float((dists <= cfg.model.threshold_cm).mean())

    final_metrics = {
        "rmse_x_cm":       rmse_x,
        "rmse_y_cm":       rmse_y,
        "mae_x_cm":        mae_x,
        "mae_y_cm":        mae_y,
        "r2_x":            r2_x,
        "r2_y":            r2_y,
        "mean_distance":   mean_dist,
        "median_distance": median_dist,
        "p90_distance":    p90_dist,
        f"fraction_within_{cfg.model.threshold_cm}cm": frac_within,
    }

    logger.info("Hold-out regression metrics:\n%s", final_metrics)
    save_metrics(
        final_metrics,
        "ConvNeXtTransferL_holdout_metrics.json",
        str(out_dir)
    )
    logger.info(f"Saved hold-out metrics to {out_dir}/ConvNeXtTransferL_holdout_metrics.json")


    logger.info(" Transfer-learning baseline complete.")

    # UMAP of internal vs. external bottleneck features (2000 samples)
    idxs_int = np.random.choice(X_int.shape[0], 2000, replace=False)
    idxs_ext = np.random.choice(X_ext.shape[0], 2000, replace=False)

    # (Re-use or redefine this helper if needed)
    def encode_in_batches(np_array):
        loader = DataLoader(
            CIRDataset(np_array),
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )
        feats_list = []
        mae.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                _, feats = mae(batch, noise_std=0.0, return_feats=True)
                feats_list.append(feats.cpu().numpy())
        return np.vstack(feats_list)

    feats_int = encode_in_batches(X_int[idxs_int])
    feats_ext = encode_in_batches(X_ext[idxs_ext])
    pooled_int = feats_int.mean(axis=1)
    pooled_ext = feats_ext.mean(axis=1)

    umapper = umap.UMAP(random_state=SEED)
    um_dom = umapper.fit_transform(np.vstack([pooled_int, pooled_ext]))

    palette = plt.get_cmap('tab10').colors
    plt.figure()
    plt.scatter(
        um_dom[:len(pooled_int), 0],
        um_dom[:len(pooled_int), 1],
        label="Internal", s=5, color=palette[0]
    )
    plt.scatter(
        um_dom[len(pooled_int):, 0],
        um_dom[len(pooled_int):, 1],
        label="External", s=5, color=palette[1]
    )
    plt.legend()
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_dir / "umap_transfer_baseline.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
