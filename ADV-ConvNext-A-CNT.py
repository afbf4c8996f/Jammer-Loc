#———————————————————————————————————————Imports———————————————————————————————————————
import argparse
import logging
import copy
from pathlib import Path
import sys 
import numpy as np
import pandas as pd
import hashlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
from tsne_plot import plot_tsne_side_by_side
from scipy.stats import wasserstein_distance
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    # note the raw string, escaped backticks, dots, and parens:
    message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary and is being deprecated where possible.*"
)
# scikit-learn FutureWarning from utils/deprecation.py
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
    module="sklearn.utils.deprecation"
)

# UMAP UserWarning about n_jobs being overridden
warnings.filterwarnings(
    "ignore",
    message="n_jobs value .* overridden to 1 by setting random_state.*",
    category=UserWarning,
    module="umap.umap_"
)

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch.amp as amp 
from torch.autograd import Function
from itertools import cycle
from sklearn import metrics as skm
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA
from config_loader import load_config
from utils import ensure_output_dir, get_run_output_dir, get_device
from metrics import  save_metrics
from sklearn.metrics import roc_auc_score,mean_squared_error, mean_absolute_error,silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from data_utils import load_csv_files, merge_positions_with_df
# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

def sigmoid_schedule(start, end, steps, midpoint=0.3, steepness=10):
    t = np.linspace(0, 1, steps)
    return start + (end - start) / (1 + np.exp(-steepness * (t - midpoint)))

def extract_feats(model, X_np, device, batch_size):
    feats = []
    loader = DataLoader(CIRDataset(X_np), batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            _, f = model(xb, noise_std=0.0, return_feats=True)
            feats.append(f.mean(dim=2).cpu().numpy())
    return np.vstack(feats)

def hash_weights(state_dict):
    concat = b''.join([v.cpu().numpy().tobytes() for k, v in state_dict.items()])
    return hashlib.md5(concat).hexdigest()

def extract_feature_magnitudes(model, X_np, device, batch_size=128, max_taps=100):
    """
    Extracts L2 magnitudes per tap from encoder bottleneck features.
    Returns shape: (N, max_taps)
    """
    model.eval()
    feats_all = []
    loader = DataLoader(CIRDataset(X_np), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            _, feats = model(xb, noise_std=0.0, return_feats=True)  # shape (B, D, T)
            mags = feats.norm(dim=1)  # shape: (B, T)
            if mags.shape[1] > max_taps:
                mags = mags[:, :max_taps]
            elif mags.shape[1] < max_taps:
                raise ValueError(f"Unexpected encoder output taps: {mags.shape[1]} < expected {max_taps}")
            feats_all.append(mags.cpu().numpy())
    return np.vstack(feats_all)  # shape (N, T)

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
        pin_memory=(device.type=="cuda")
    )
    mses = []
    with torch.no_grad():
        for Xb in loader:
            Xb = Xb.to(device)
            rec = model(Xb, noise_std=0.0)  
            per_sample_mse = ((rec - Xb) ** 2).mean(dim=(1,2))
            mses.append(per_sample_mse.cpu().numpy())
    return np.concatenate(mses)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal for DANN
# ──────────────────────────────────────────────────────────────────────────────
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
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
        # generate binary mask
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

    def forward(self, x,noise_std: float = 0.0):
        # x: (B, C, L) → convert to (B, L, C) for LayerNorm
        shortcut = x
        x = self.dwconv(x)                  # (B, C, L)
        if self.training and noise_std>0:
            x = x + torch.randn_like(x) * noise_std

        x = x.permute(0, 2, 1)              # (B, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x) * self.gamma    # channel‐wise scaling
        x = x.permute(0, 2, 1)              # back to (B, C, L)
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
        # channel widths at each stage
        dims = [hidden_dim, hidden_dim * 2, hidden_dim * 4]
        total_blocks = sum(depths)*2 + depths[-1]
        dp_rates     = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        dp_iter      = iter(dp_rates)


        # -- Encoder stem (no downsampling) --
        self.stem = nn.Conv1d(in_ch, dims[0], kernel_size=3, padding=1)

        # -- Encoder stages --
        self.encoder_blocks   = nn.ModuleList()
        self.encoder_downsamps = nn.ModuleList()
        for i, num_blocks in enumerate(depths):
            blocks = [ConvNeXtBlock1D(dims[i], drop_path=next(dp_iter)) 
                      for _ in range(num_blocks)]
            self.encoder_blocks.append(SeqWithNoise(*blocks))
            # downsample after each stage except last
            if i < len(depths) - 1:
                self.encoder_downsamps.append(
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
                )

        # -- Bottleneck blocks --
        self.bottleneck = SeqWithNoise(*[ConvNeXtBlock1D(dims[-1], drop_path=next(dp_iter))
                                 for _ in range(depths[-1])])


        # -- Decoder stages (mirror of encoder) --
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks    = nn.ModuleList()
        for i, num_blocks in enumerate(reversed(depths)):
            # upsample from dims[-1-i] → dims[-2-i] (or dims[0] → in_ch at final)
            in_dim  = dims[-1-i]
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
                # last (deepest) stage: no upsample, just blocks at in_dim
                block_dim = in_dim
                
            blocks = [ConvNeXtBlock1D(block_dim, drop_path=next(dp_iter))
                      for _ in range(num_blocks)]
            
            self.decoder_blocks.append(SeqWithNoise(*blocks))
            
        # -- Final projection back to original channels --
        self.final_conv = nn.Conv1d(dims[0], in_ch, kernel_size=3, padding=1)

    def forward(self, x, noise_std: float = 0.0,return_feats: bool = False):
        """
        x: (B, L, C)
        noise_std: if >0, adds Gaussian noise to inputs instead of masking
        returns: reconstructed tensor of same shape
        """
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        # move to (B, C, L) for conv layers
        x = x.permute(0, 2, 1)
        x = self.stem(x)

        # Encoder
        for blk, down in zip(self.encoder_blocks, self.encoder_downsamps):
            x = blk(x, noise_std=noise_std)
            x = down(x)
        # last encoder block if depths[-1] > 0
        if len(self.encoder_blocks) > len(self.encoder_downsamps):
            x = self.encoder_blocks[-1](x)

        # Bottleneck
        feats = self.bottleneck(x,noise_std=noise_std)

        # Decoder
        x = feats
        for up, blk in zip(self.decoder_upsamples, self.decoder_blocks):
            x = up(x)
            x = blk(x, noise_std=noise_std)

        # apply the last block-group (no upsample)
        x = self.decoder_blocks[-1](x)
        # Final conv & back to (B, L, C)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)

        if return_feats:
            
            return x, feats.permute(0,2,1)
        else:
            return x
       
class RegHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 2)
        )
    def forward(self, x):
        x = x.permute(0,2,1)
        x = nn.functional.adaptive_avg_pool1d(x,1).squeeze(-1)
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2,1)
        )
    def forward(self, x):
        x = x.mean(dim=1)
        return self.net(x).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Datasets & Helpers
# ──────────────────────────────────────────────────────────────────────────────
class CIRDataset(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i): return torch.from_numpy(self.X[i]).float()

class CIRRegressionDataset(Dataset):
    def __init__(self, X,y): self.X,self.y = X,y
    def __len__(self): return self.X.shape[0]
    def __getitem__(self,i):
        return torch.from_numpy(self.X[i]).float(), torch.from_numpy(self.y[i]).float()

# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ——————————————— Setup & Hyperparams ———————————————
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for attr in ('hidden_size', 'fine_tune_epochs'):
        if not hasattr(cfg.model, attr):
            raise KeyError(f"config.model missing required key: {attr}")


    device = get_device()
    use_amp = (device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    

    logging.basicConfig(level=logging.INFO,force=True,
        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("▶ Starting pretrain-and-finetune pipeline")

    #———————————————————————————————————————————————————————————————————————————————————————————
    # Research overrides
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS= 5
    ADAPT_EPOCHS  =   40
    LAMBDA_ADAPT_MAX= 0.2  # reduced adversarial strength for adaptation 0.1 (0.3)
    PATIENCE_ADAPT_AUC = 10 # was 3  
    
    LAMBDA_FT_MAX   = 0.5   # cap fine-tune adversarial strength 0.2 (0.5)   
    ALPHA_RECON_FT=   0.5  # this was 0.5
    ALPHA_RECON_MIN =  0.1 # floor for reconstruction weight in fine-tune (before it was 0.1)
    BETA_REG_FT   =   1.0   # was 1.0
   
    DIAG_SUBSAMPLE=   3000
    K_HEATMAP_SAMPLES= 100
    K_SERIES      =   6
    PATIENCE      =   15    # early-stop fine-tune if no reg RMSE improvement
    PATIENCE_FT_CM = 10
    Noise         =   0.6 

    
    # ————————————— Data loading + Norm ————————————————————————————————————————————————————
    base_outdir = Path(cfg.output.directory)
    out_dir     = Path(get_run_output_dir(cfg))
    ensure_output_dir(out_dir)
    
    internal_npz = base_outdir/"internal_baked.npz"
    external_npz = base_outdir/"external_baked.npz"
    data_int = np.load(internal_npz)
    data_ext = np.load(external_npz)

    X_int, Y_int = data_int["X"], data_int["Y"]
    X_ext, Y_ext = data_ext["X"], data_ext["Y"]
    
    # ───── TRUNCATE TO FIRST 100 TAPS ─────
    X_int = X_int[:, :100, :]
    X_ext = X_ext[:, :100, :]

    
    # ─── A: Compute per‐tap clamp thresholds on the raw magnitude channel ───
    #     X_* shape: (N, L, C), and channel 0 = magnitude
    mag = X_int[..., 0]                              # shape (N_int, L)
    q1  = np.percentile(mag, 25, axis=0)             # shape (L,)
    q3  = np.percentile(mag, 75, axis=0)             # shape (L,)
    iqr = q3 - q1                                     # shape (L,)
    clamp_min = q1 - 1.5 * iqr                        # shape (L,)
    clamp_max = q3 + 1.5 * iqr                        # shape (L,)

    # ── Sanity check: make sure our clamp arrays align with the tap dimension ──
    assert clamp_min.ndim == 1 and clamp_min.shape[0] == X_int.shape[1], (
        f"clamp_min shape {clamp_min.shape} does not match #taps {X_int.shape[1]}"
    )
    assert clamp_max.ndim == 1 and clamp_max.shape[0] == X_int.shape[1], (
        f"clamp_max shape {clamp_max.shape} does not match #taps {X_int.shape[1]}"
    )


    # ─── B: Clamp raw‐unit magnitudes before any normalization ───
    X_int[..., 0] = np.clip(X_int[..., 0], clamp_min, clamp_max)
    X_ext[..., 0] = np.clip(X_ext[..., 0], clamp_min, clamp_max)

    # ─── C: Recompute normalization stats on the *clamped* raw data ──
    raw_means = X_int.mean((0,1))
    raw_stds  = X_int.std ((0,1)) + 1e-6

    # force to numpy arrays and rename to avoid any later shadowing
    norm_means = np.array(raw_means, dtype=np.float32)
    norm_stds  = np.array(raw_stds,  dtype=np.float32)

    # ─── D: Now normalize into zero‐mean/unit‐std for model input ───
    X_int = (X_int - norm_means[None,None,:]) / norm_stds[None,None,:]
    X_ext = (X_ext - norm_means[None,None,:]) / norm_stds[None,None,:]




    # 5)───────────────────────────── Sanity-check shapes─────────────────────────────
    assert X_int.ndim == 3 and Y_int.ndim == 2
    assert X_ext.ndim == 3 and Y_ext.ndim == 2
    assert X_int.shape[2] == X_ext.shape[2]
    assert X_ext.shape[0] == Y_ext.shape[0]

    # ————————————— Model & Opt —————————————──────────────────────────────────────────
    mae = ConvNeXtAutoencoder1D(
    in_ch=X_int.shape[2],
    hidden_dim=cfg.model.hidden_size,
    depths=[2,2,2],
    drop_path_rate=0.1
    ).to(device)

    bottleneck_dim = cfg.model.hidden_size * 4
    reg_head = RegHead(bottleneck_dim).to(device)
    dom_head = DomainClassifier(bottleneck_dim).to(device)

    opt_mae = optim.Adam(mae.parameters(), lr=cfg.training.learning_rate, weight_decay=WEIGHT_DECAY)
    opt_dom = optim.Adam(dom_head.parameters(), lr=cfg.training.learning_rate*0.1, weight_decay=WEIGHT_DECAY)
    opt_reg = optim.Adam(reg_head.parameters(), lr=3e-3, weight_decay=WEIGHT_DECAY)

    # Ensure warmup and post-warmup epochs are ≥1
    pre_warmup      = min(WARMUP_EPOCHS, cfg.training.num_epochs)
    post_pre_epochs = max(cfg.training.num_epochs - WARMUP_EPOCHS, 1)
    sched_pre = SequentialLR(opt_mae,
                             schedulers=[
                                 LinearLR(
                                    opt_mae,
                                    start_factor=0.1,
                                    end_factor=1.0,
                                    total_iters=pre_warmup
                            ),
                            CosineAnnealingLR(opt_mae, T_max=post_pre_epochs)
                            ],
                            milestones=[pre_warmup]
                        )
    # Ensure adapt-warmup and post-warmup epochs are ≥1
    adapt_warmup      = max(ADAPT_EPOCHS//5, 1)
    post_adapt_epochs = max(ADAPT_EPOCHS - ADAPT_EPOCHS//5, 1)

    sched_adapt = SequentialLR(opt_mae,
                               schedulers=[
                                   LinearLR(
                                   opt_mae,
                                   start_factor=0.1,
                                   end_factor=1.0,
                                   total_iters=adapt_warmup
                                ),
                                CosineAnnealingLR(opt_mae, T_max=post_adapt_epochs)
                            ],
                            milestones=[adapt_warmup]
                            )
                                                              
    # Fine-tune scheduler: simple cosine over fine-tune epochs
    sched_ft = CosineAnnealingLR(opt_reg, T_max=cfg.model.fine_tune_epochs)
    sched_mae_ft = CosineAnnealingLR(opt_mae, T_max=cfg.model.fine_tune_epochs)

    crit_mae = nn.MSELoss()
    crit_dom = nn.BCEWithLogitsLoss()
    crit_reg = nn.MSELoss()
    scaler   = amp.GradScaler() if device.type=="cuda" else None

    # Detect whether we’re running interactively
    disable_tqdm = not sys.stdout.isatty()

    pretrain_ckpt = out_dir / "best_pretrain.pt"
    if pretrain_ckpt.exists():

        logger.info(f"⏭️  Loading pretrained AE weights from {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device,weights_only=True)
        mae.load_state_dict(ckpt['model_state'])
        opt_mae.load_state_dict(ckpt['optimizer_state'])

        
    else:
        # ————————————— Stage 1: Pretrain —————————————
        best_pre, best_pre_w = float("inf"), None
        for ep in range(1,cfg.training.num_epochs+1):
            mae.train() 
            totL, totM=0,0
            loader=DataLoader(
                CIRDataset(X_int),
                batch_size=cfg.training.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=(device.type=="cuda"),
                prefetch_factor=2
                )
            for Xb in tqdm(loader, desc=f"Pretrain Ep {ep}/{cfg.training.num_epochs}",
                disable=disable_tqdm,leave=False):
                Xb=Xb.to(device)
                opt_mae.zero_grad()
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    recon = mae(Xb, noise_std=Noise)
                    loss  = crit_mae(recon, Xb)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt_mae)
                    nn.utils.clip_grad_norm_(mae.parameters(),1.0)
                    scaler.step(opt_mae); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(mae.parameters(),1.0)
                    opt_mae.step()
                # weight by total number of elements in Xb (B×L×C)
                n = Xb.numel()
                totL += loss.item() * n
                totM+=n
            pre_loss=totL/(totM+1e-8)
            logger.info(f"Pre {ep}/{cfg.training.num_epochs} Loss={pre_loss:.6f} LR={opt_mae.param_groups[0]['lr']:.2e}")
            if pre_loss<best_pre: best_pre, best_pre_w=pre_loss,copy.deepcopy(mae.state_dict())
            sched_pre.step()
        # restore best pretrain weights
        mae.load_state_dict(best_pre_w)
        # Save
        torch.save({
        'epoch': ep,
        'model_state':     best_pre_w,
        'optimizer_state': opt_mae.state_dict(),
        }, pretrain_ckpt)

        

    # ─── AE-pruning: drop the top 2% highest–MSE samples in both datasets (int & ext) ───
    mae.eval()
    # compute MSEs
    mse_int = compute_mse_per_sample(mae, X_int,
        batch_size=cfg.training.batch_size,
        device=device)
    mse_ext = compute_mse_per_sample(mae, X_ext,
        batch_size=cfg.training.batch_size,
        device=device)
    # keep the 98% lowest-MSE samples
    thresh_int = np.quantile(mse_int, 0.98)
    thresh_ext = np.quantile(mse_ext, 0.98)
    keep_int  = mse_int < thresh_int
    keep_ext  = mse_ext < thresh_ext

    # prune both data & labels
    X_int, Y_int = X_int[keep_int], Y_int[keep_int]
    X_ext, Y_ext = X_ext[keep_ext], Y_ext[keep_ext]
    # Sampling for t-SNE — keep this global
    sample_int = np.random.choice(len(X_int), 1000, replace=False)
    sample_ext = np.random.choice(len(X_ext), 1000, replace=False)
    # Run t-SNE BEFORE adaptation
    F_s  = extract_feats(mae, X_int[sample_int], device, cfg.training.batch_size)
    F_t  = extract_feats(mae, X_ext[sample_ext], device, cfg.training.batch_size)



    # ————————————— Stage 2: Adapt —————————————

    # ────────────────────────────────────────────────────────────────────
    # Stage 2: Adapt (ConvNeXt Autoencoder version)
    # ────────────────────────────────────────────────────────────────────


    # lam_schedule = sigmoid_schedule(0.05, LAMBDA_ADAPT_MAX, ADAPT_EPOCHS)
    lam_schedule = sigmoid_schedule(0.05, 0.2, 40, midpoint=0.5, steepness=5)


    ext_loader = DataLoader(
        CIRDataset(X_ext),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
        prefetch_factor=2,
        drop_last=True
    )

    int_loader = DataLoader(
        CIRDataset(X_int),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
        prefetch_factor=2,
        drop_last=True
    )


    # — Training —
    best_auc = 0.0
    no_improve_auc = 0
    best_ad_w = None

    for ep in range(1, ADAPT_EPOCHS + 1):
        mae.train()
        sum_lrec, sum_ldom, batch_cnt = 0.0, 0.0, 0
        totL, totM = 0.0, 0
        lam = float(lam_schedule[ep - 1])

        all_labels, all_probs = [], []
        int_iter = cycle(int_loader)

        for Xb_ext in tqdm(ext_loader, desc=f"Adapt Ep {ep}/{ADAPT_EPOCHS}", disable=disable_tqdm, leave=False, unit="batch"):
            Xb_ext = Xb_ext.to(device)
            Xb_int = next(int_iter).to(device)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                recon_ext, feats_ext = mae(Xb_ext, noise_std=Noise, return_feats=True)
                lrec = crit_mae(recon_ext, Xb_ext)
                sum_lrec += lrec.item()

                _, feats_int = mae(Xb_int, noise_std=Noise, return_feats=True)
                feats_all = torch.cat([feats_int, feats_ext], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(Xb_int.size(0), device=device),
                    torch.ones (Xb_ext.size(0), device=device)
                ], dim=0)

                logits = dom_head(grad_reverse(feats_all, lam))
                ldom = crit_dom(logits, domain_labels)
                sum_ldom += ldom.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels_np = domain_labels.detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels_np)

            loss = lrec + lam * ldom
            batch_cnt += 1

            opt_mae.zero_grad()
            opt_dom.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_mae)
                scaler.unscale_(opt_dom)
                nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                nn.utils.clip_grad_norm_(dom_head.parameters(), 1.0)
                scaler.step(opt_mae)
                scaler.step(opt_dom)
                scaler.update()
            else:
                loss.backward()
                opt_mae.step()
                opt_dom.step()

            totL += lrec.item() * Xb_ext.numel()
            totM += Xb_ext.numel()

        ad_loss = totL / (totM + 1e-8)
        avg_lrec = sum_lrec / batch_cnt if batch_cnt else float("nan")
        avg_ldom = sum_ldom / batch_cnt if batch_cnt else float("nan")
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        epoch_auc = roc_auc_score(all_labels, all_probs)

        sched_adapt.step()
        curr_hash = hash_weights(mae.state_dict())

        logger.info(
            f"Adapt {ep}/{ADAPT_EPOCHS}  "
            f"L_rec={avg_lrec:.4f}  L_dom={avg_ldom:.4f}  "
            f"Total={ad_loss:.4f}  λ={lam:.2f}  AUC={epoch_auc:.4f}  "
            f"Hash={curr_hash}"
        )

        if epoch_auc > best_auc:
            best_auc = epoch_auc
            best_ad_w = copy.deepcopy(mae.state_dict())
            best_ad_hash = hash_weights(best_ad_w)
            logger.info(f" Saving new best AUC model at epoch {ep} — Hash={best_ad_hash}")
            no_improve_auc = 0
        else:
            no_improve_auc += 1
            if no_improve_auc >= PATIENCE_ADAPT_AUC:
                logger.info(f" Early stopping Adapt at epoch {ep} (AUC plateau)")
                break

    # Use fallback if needed
    if best_ad_w is None:
        logger.warning("No best AUC found — using latest trained model as adapted model.")
        best_ad_w = copy.deepcopy(mae.state_dict())
        final_adapt_hash = hash_weights(best_ad_w)
    else:
        final_adapt_hash = hash_weights(best_ad_w)
        pretrain_hash = hash_weights(ckpt['model_state'])
        if final_adapt_hash == pretrain_hash:
            logger.warning(" Best AUC model is identical to pretrain — using final trained model instead.")
            best_ad_w = copy.deepcopy(mae.state_dict())
            final_adapt_hash = hash_weights(best_ad_w)

    # Load final chosen model
    mae.load_state_dict(best_ad_w)

    # Save it
    adapt_ckpt_path = out_dir / "best_adapt.pt"
    torch.save({
        'epoch': ep,
        'model_state': best_ad_w,
        'hash': final_adapt_hash,
    }, adapt_ckpt_path)
    logger.info(f"Saved best adapted model to {adapt_ckpt_path}")

    # Log comparison
    logger.info(" Hash of PRETRAIN weights: %s", hash_weights(ckpt['model_state']))
    logger.info(" Hash of ADAPT weights:    %s", final_adapt_hash)

    # Post-adaptation t-SNE
    # Post-adaptation t-SNE features (after GRL)
    F_s2 = extract_feats(mae, X_int[sample_int], device, cfg.training.batch_size)
    F_t2 = extract_feats(mae, X_ext[sample_ext], device, cfg.training.batch_size)
    plot_tsne_side_by_side(
    source_feats_1=F_s,  # before
    target_feats_1=F_t,
    source_feats_2=F_s2,  # after
    target_feats_2=F_t2,
    label_1="Before GRL",
    label_2="After GRL",
    out_path=out_dir / "tsne_before_after_grl.pdf",
    perplexity=30,
    max_iter=3000
    )
    #  Feature shift
    delta_src = np.linalg.norm(F_s2 - F_s)
    delta_tgt = np.linalg.norm(F_t2 - F_t)

    logger.info(f" Feature shift after adaptation — Source: Δ={delta_src:.4f}")
    logger.info(f" Feature shift after adaptation — Target: Δ={delta_tgt:.4f}")

    # restore best adaptation weights
    mae.load_state_dict(best_ad_w)


    # ────────────────────────────────────────────────────────────────────
    # Stage 3: Fine-tune (ConvNeXtAutoencoder1D)
    # ────────────────────────────────────────────────────────────────────

    # --- 1) Freeze all MAE parameters, then unfreeze last encoder stage + decoder + final proj
    for p in mae.parameters():
        p.requires_grad = False

    # Unfreeze last encoder stage blocks (depths[-1] blocks)
    for p in mae.encoder_blocks[-1].parameters():
        p.requires_grad = True

    # Unfreeze the decoder path so reconstruction can still adapt
    for module in [mae.decoder_upsamples, mae.decoder_blocks, mae.final_conv]:
        for p in module.parameters():
            p.requires_grad = True

    # Regression head remains trainable
    # (opt_reg will update reg_head; opt_mae will update unfrozen mae params)

    # Schedules for fine‐tuning:
    #  - α (reconstruction weight) decays from ALPHA_RECON_FT → ALPHA_RECON_MIN
    #  - λ_ft (adversarial weight) grows from 0 → LAMBDA_FT_MAX
    alpha_schedule     = np.linspace(ALPHA_RECON_FT, ALPHA_RECON_MIN, cfg.model.fine_tune_epochs)
    lambda_ft_schedule = np.linspace(0.0,       LAMBDA_FT_MAX,     cfg.model.fine_tune_epochs)

    # ─────────────────────────── Stratified split ───────────────────────────
    df_meta_raw, _ = load_csv_files(cfg.data.external_test_path, parse_cir=False)
    df_meta = merge_positions_with_df(
        df_meta_raw,
        cfg.data.coords_xlsx,
        excluded_positions=cfg.data.exclude_positions,
        merge_how='inner'
    )
    # ─── prune df_meta to match X_ext after AE-pruning ───
    df_meta = df_meta.iloc[keep_ext].reset_index(drop=True)

    assert len(df_meta) == X_ext.shape[0], (
        f"df_meta rows ({len(df_meta)}) != X_ext samples ({X_ext.shape[0]})"
    )
    labels  = df_meta['label'].values
    all_idx = np.arange(X_ext.shape[0])
    calib_idxs, hold_idxs = train_test_split(
        all_idx, test_size=0.50, stratify=labels, random_state=SEED
    )

    Xc, Yc = X_ext[calib_idxs], Y_ext[calib_idxs]
    scaler_y = StandardScaler().fit(Yc)
    Yc_norm = scaler_y.transform(Yc)
    calib_loader = DataLoader(
        CIRRegressionDataset(Xc, Yc_norm),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
        prefetch_factor=2
    )

    best_reg_w   = None
    best_mae_w   = None
    # Track real-world errors in cm 
    best_rmse_cm   = float("inf")
    no_improve_ft = 0





    # ────────────────────────────────────────────────────────────
    # Fine-tuning loop 
    # ────────────────────────────────────────────────────────────

    # Initialize best-RMSE tracking
    best_reg_w   = None
    best_mae_w   = None
    best_rmse_cm = float("inf")
    no_improve_ft= 0
    PATIENCE_FT_CM = PATIENCE

    for ep in range(1, cfg.model.fine_tune_epochs + 1):
        alpha     = alpha_schedule[ep - 1]
        lambda_ft = lambda_ft_schedule[ep - 1]

        mae.train(); reg_head.train()

        # Existing accumulators
        sum_lrec_ft, sum_lreg_ft, sum_ldom_ft = 0.0, 0.0, 0.0
        totL, cnt                            = 0.0, 0

        # Accumulators for true-cm metrics
        sum_rmse_cm, sum_mae_cm = 0.0, 0.0
        batch_count             = 0

        for Xb, yb in tqdm(calib_loader, desc=f"FT Ep {ep}/{cfg.model.fine_tune_epochs}",
                        disable=disable_tqdm, leave=False, unit="batch"):
            Xb, yb = Xb.to(device), yb.to(device)

            # — Forward + combined loss (unchanged) —
            with amp.autocast(device_type=device.type, enabled=use_amp):
                recon, feats = mae(Xb, noise_std=Noise, return_feats=True)
                lrec = crit_mae(recon, Xb); sum_lrec_ft += lrec.item()
                pred = reg_head(feats)
                lreg = crit_reg(pred, yb);     sum_lreg_ft += lreg.item()
                logits = dom_head(grad_reverse(feats, lambda_ft))
                ldom = crit_dom(logits, torch.zeros_like(logits)); sum_ldom_ft += ldom.item()
                loss = alpha*lrec + BETA_REG_FT*lreg + lambda_ft*ldom

            # — Backward + step (unchanged) —
            opt_mae.zero_grad(); opt_reg.zero_grad(); opt_dom.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt_mae); scaler.unscale_(opt_reg); scaler.unscale_(opt_dom)
                torch.nn.utils.clip_grad_norm_(
                    list(mae.encoder_blocks[-1].parameters()) +
                    list(mae.decoder_upsamples.parameters()) +
                    list(mae.decoder_blocks.parameters()) +
                    list(mae.final_conv.parameters()) +
                    list(reg_head.parameters()),
                    max_norm=1.0
                )
                scaler.step(opt_mae); scaler.step(opt_reg); scaler.step(opt_dom); scaler.update()
            else:
                loss.backward(); opt_mae.step(); opt_reg.step(); opt_dom.step()

            # — Accumulate total-loss for logging (unchanged) —
            totL += loss.item() * Xb.size(0)
            cnt  += Xb.size(0)

            # ── NEW: compute _per-sample_ Euclidean errors in cm ──
            pred_np   = pred.detach().cpu().numpy()
            true_np   = yb.detach().cpu().numpy()
            pred_cm   = scaler_y.inverse_transform(pred_np)
            true_cm   = scaler_y.inverse_transform(true_np)
            dists     = np.linalg.norm(pred_cm - true_cm, axis=1)  # (B,)
            sum_rmse_cm += dists.mean()                            # batch RMSE_cm
            sum_mae_cm  += np.abs(pred_cm - true_cm).mean()        # batch MAE_cm
            batch_count += 1
            # ───────────────────────────────────────────────────────

        # — Epoch metrics —
        ft_loss    = totL / (cnt + 1e-8)
        avg_lrec   = sum_lrec_ft / batch_count
        avg_lreg   = sum_lreg_ft / batch_count
        avg_ldom   = sum_ldom_ft / batch_count

        # ── Average true-cm metrics ──
        avg_rmse_cm = sum_rmse_cm / batch_count
        avg_mae_cm  = sum_mae_cm  / batch_count
        # ──────────────────────────────────

        logger.info(
            f"FT {ep}/{cfg.model.fine_tune_epochs}  "
            f"L_rec={avg_lrec:.4f}  L_reg={avg_lreg:.4f}  L_dom={avg_ldom:.4f}  "
            f"RMSE_cm={avg_rmse_cm:.2f}  MAE_cm={avg_mae_cm:.2f}  "
            f"Total={ft_loss:.4f}  α={alpha:.2f}  λ_ft={lambda_ft:.2f}"
        )

        # ── Early‐stop on rmse_cm ──
        if avg_rmse_cm < best_rmse_cm:
            best_rmse_cm  = avg_rmse_cm
            no_improve_ft = 0
            best_reg_w    = copy.deepcopy(reg_head.state_dict())
            best_mae_w    = copy.deepcopy(mae.state_dict())
        else:
            no_improve_ft += 1
            if no_improve_ft >= PATIENCE_FT_CM:
                logger.info(f"Early stopping FT at epoch {ep} (RMSE_cm plateau)")
                break
        # ──────────────────────────────

        sched_ft.step()
        sched_mae_ft.step()

    # Restore best weights
    mae.load_state_dict(best_mae_w)
    reg_head.load_state_dict(best_reg_w)


    # ————————————— Stage 4: Hold-out Eval —————————————
    # ────────────────────────────────────────────────────────────
    # Stage 4: Hold-out Evaluation (ConvNeXtAutoencoder1D version)
    # ────────────────────────────────────────────────────────────
    mae.eval()
    reg_head.eval()

    hold_loader = DataLoader(
        CIRRegressionDataset(X_ext[hold_idxs], Y_ext[hold_idxs]),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type=="cuda"),
        prefetch_factor=2
    )

    preds_list  = []
    truths_list = []

    with torch.no_grad():
        for Xb, yb in tqdm(hold_loader, desc="Hold-out Eval", leave=False, unit="batch"):
            Xb, yb = Xb.to(device), yb.to(device)
            # get reconstruction (unused here) and bottleneck features
            _, feats = mae(Xb, noise_std=0.0, return_feats=True)
            # regression prediction (normalized)
            pred_norm = reg_head(feats).cpu()
            preds_list.append(pred_norm)
            truths_list.append(yb.cpu())

    # concatenate and convert once
    preds  = torch.cat(preds_list,  dim=0).numpy()
    truths = torch.cat(truths_list, dim=0).numpy()

    # inverse-transform the normalized predictions
    preds_cm = scaler_y.inverse_transform(preds)

    # compute Euclidean distances
    dists = np.linalg.norm(preds_cm - truths, axis=1)

    # Per-axis errors
    mse_x  = skm.mean_squared_error(truths[:,0], preds_cm[:,0])
    rmse_x = float(np.sqrt(mse_x))
    mse_y  = skm.mean_squared_error(truths[:,1], preds_cm[:,1])
    rmse_y = float(np.sqrt(mse_y))
    mae_x  = skm.mean_absolute_error(truths[:,0], preds_cm[:,0])
    mae_y  = skm.mean_absolute_error(truths[:,1], preds_cm[:,1])
    r2_x   = skm.r2_score(truths[:,0], preds_cm[:,0])
    r2_y   = skm.r2_score(truths[:,1], preds_cm[:,1])

    # Euclidean distance metrics
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
        f"{cfg.model.name}_holdout_metrics.json",
        str(out_dir)
    )
    logger.info(f"Saved hold-out metrics to {out_dir}/{cfg.model.name}_holdout_metrics.json")

if __name__=="__main__":
    main()