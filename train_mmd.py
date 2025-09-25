import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import argparse, json, logging
from tqdm import tqdm
from itertools import cycle

from tsne_plot import plot_tsne_domain_alignment
from config_loader import load_config
from utils import ensure_output_dir, get_run_output_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# === Model Definitions ===

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        return x + shortcut

class Encoder1D(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, depths=[2, 2, 2]):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock1D(hidden_dim) for _ in range(sum(depths))])
        self.out_proj = nn.Conv1d(hidden_dim, hidden_dim * 4, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x

class RegHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, 2)
        )

    def forward(self, x):
        return self.net(x.mean(dim=2))

# === Dataset Definitions ===

class CIRDataset(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]).float()

class CIRRegressionDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return torch.from_numpy(self.X[i]).float(), torch.from_numpy(self.y[i]).float()

# === MMD Loss ===

def mmd_loss(x, y):
    B = x.size(0)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0)
    total1 = total.unsqueeze(1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = torch.sum(L2_distance.data) / (total.shape[0] ** 2 - total.shape[0])
    kernel_val = [torch.exp(-L2_distance / (bandwidth * (2.0 ** i))) for i in range(5)]
    kernels = sum(kernel_val) / len(kernel_val)
    XX = kernels[:B, :B]
    YY = kernels[B:, B:]
    XY = kernels[:B, B:]
    YX = kernels[B:, :B]
    return XX.mean() + YY.mean() - XY.mean() - YX.mean()

# === t-SNE Feature Extractor ===

def extract_feats(encoder, X_np, device, batch_size):
    feats = []
    loader = DataLoader(CIRDataset(X_np), batch_size=batch_size)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            f = encoder(xb)
            feats.append(f.mean(dim=2).cpu().numpy())
    return np.vstack(feats)

# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    base_outdir = Path(cfg.output.directory)
    out_dir = Path(get_run_output_dir(cfg))
    ensure_output_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load .npz data
    data_int = np.load(base_outdir / "internal_baked.npz")
    data_ext = np.load(base_outdir / "external_baked.npz")
    X_int, Y_int = data_int["X"], data_int["Y"]
    X_ext, Y_ext = data_ext["X"], data_ext["Y"]
    X_int = X_int[:, :100, :]
    X_ext = X_ext[:, :100, :]

    # Clamp + normalize inputs
    mag = X_int[..., 0]
    q1, q3 = np.percentile(mag, [25, 75], axis=0)
    iqr = q3 - q1
    X_int[..., 0] = np.clip(X_int[..., 0], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    X_ext[..., 0] = np.clip(X_ext[..., 0], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    mean, std = X_int.mean((0, 1)), X_int.std((0, 1)) + 1e-6
    X_int = (X_int - mean) / std
    X_ext = (X_ext - mean) / std

    labels = np.round(Y_ext[:, 0] + Y_ext[:, 1], 1)
    idxs = np.arange(len(Y_ext))
    calib_idx, hold_idx = train_test_split(idxs, test_size=0.5, stratify=labels, random_state=42)

    scaler_y = StandardScaler().fit(Y_int)
    Y_int_scaled = scaler_y.transform(Y_int)
    Y_ext_scaled = scaler_y.transform(Y_ext)

    encoder = Encoder1D(in_ch=X_int.shape[2], hidden_dim=cfg.model.hidden_size).to(device)
    reg_head = RegHead(cfg.model.hidden_size * 4).to(device)
    opt = optim.AdamW(list(encoder.parameters()) + list(reg_head.parameters()), lr=cfg.training.learning_rate)
    mse = nn.MSELoss()

    X_train, X_val, y_train, y_val = train_test_split(X_int, Y_int_scaled, test_size=0.1, random_state=42)
    loader_src = DataLoader(CIRRegressionDataset(X_train, y_train), batch_size=cfg.training.batch_size, shuffle=True)
    loader_val = DataLoader(CIRRegressionDataset(X_val, y_val), batch_size=cfg.training.batch_size)
    loader_tgt = DataLoader(CIRDataset(X_ext[calib_idx]), batch_size=cfg.training.batch_size, shuffle=True)
    tgt_iter = cycle(loader_tgt)

    pretrain_ckpt = out_dir / "best_pretrain_mmd.pt"

    if pretrain_ckpt.exists():
        logger.info(f"⏸️  Loading pretrained weights from {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location=device,weights_only=True)
        encoder.load_state_dict(ckpt['encoder_state'])
        reg_head.load_state_dict(ckpt['reg_head_state'])
        opt.load_state_dict(ckpt['optimizer_state'])
    else:
        logger.info("💪 Starting supervised pretraining...")
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        for epoch in tqdm(range(1, 51), desc="Pretraining", position=0):
            encoder.train(); reg_head.train(); total_loss = 0.0
            for xb, yb in loader_src:
                xb, yb = xb.to(device), yb.to(device)
                pred = reg_head(encoder(xb))
                loss = mse(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(loader_src)

            encoder.eval(); reg_head.eval(); val_loss = 0.0
            with torch.no_grad():
                for xb, yb in loader_val:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = reg_head(encoder(xb))
                    val_loss += mse(pred, yb).item()
            avg_val_loss = val_loss / len(loader_val)

            logger.info(f"Pretrain Epoch {epoch:03d} | Train MSE = {avg_train_loss:.4f} | Val MSE = {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'encoder_state': encoder.state_dict(),
                    'reg_head_state': reg_head.state_dict(),
                    'optimizer_state': opt.state_dict(),
                }, pretrain_ckpt)
                logger.info("🔖 Saved new best pretrain checkpoint.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("⏹️ Early stopping triggered.")
                    break

    # t-SNE Before MMD
    F_s = extract_feats(encoder, X_int[np.random.choice(len(X_int), 1000, replace=False)], device, cfg.training.batch_size)
    F_t = extract_feats(encoder, X_ext[calib_idx][np.random.choice(len(calib_idx), 1000, replace=False)], device, cfg.training.batch_size)
    plot_tsne_domain_alignment(F_s, F_t, "Before MMD", out_dir / "tsne_minimal_before.pdf")

    # === MMD Alignment ===
    lambda_mmd = 0.1
    mmd_bar = tqdm(range(1, 50+1), desc="MMD Alignment", position=0)
    for epoch in mmd_bar:
        encoder.train(); reg_head.train()
        total_reg_loss, total_mmd_loss = 0.0, 0.0
        for xb_src, yb_src in loader_src:
            xb_tgt = next(tgt_iter)
            xb_src, yb_src, xb_tgt = xb_src.to(device), yb_src.to(device), xb_tgt.to(device)
            f_src = encoder(xb_src)
            f_tgt = encoder(xb_tgt)
            pred = reg_head(f_src)
            reg_loss = mse(pred, yb_src)
            align_loss = mmd_loss(f_src.mean(dim=2), f_tgt.mean(dim=2))
            loss = reg_loss + lambda_mmd * align_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_reg_loss += reg_loss.item()
            total_mmd_loss += align_loss.item()
        avg_reg_loss = total_reg_loss / len(loader_src)
        avg_mmd_loss = total_mmd_loss / len(loader_src)
        logger.info(f"MMD Align Epoch {epoch:03d} | Reg = {avg_reg_loss:.4f} | MMD = {avg_mmd_loss:.4f} | λ = {lambda_mmd}")

    # t-SNE After MMD
    F_s2 = extract_feats(encoder, X_int[np.random.choice(len(X_int), 1000, replace=False)], device, cfg.training.batch_size)
    F_t2 = extract_feats(encoder, X_ext[calib_idx][np.random.choice(len(calib_idx), 1000, replace=False)], device, cfg.training.batch_size)
    plot_tsne_domain_alignment(F_s2, F_t2, "After MMD", out_dir / "tsne_minimal_after.pdf")

    # === Evaluation ===
    loader_eval = DataLoader(CIRRegressionDataset(X_ext[hold_idx], Y_ext[hold_idx]), batch_size=cfg.training.batch_size)
    encoder.eval(); reg_head.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader_eval:
            xb = xb.to(device)
            out = reg_head(encoder(xb))
            preds.append(out.cpu().numpy())
            trues.append(yb.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    preds_cm = scaler_y.inverse_transform(preds)
    dist = np.linalg.norm(preds_cm - trues, axis=1)

    metrics = {
        "rmse_x_cm": float(np.sqrt(mean_squared_error(trues[:, 0], preds_cm[:, 0]))),
        "rmse_y_cm": float(np.sqrt(mean_squared_error(trues[:, 1], preds_cm[:, 1]))),
        "mae_x_cm": float(mean_absolute_error(trues[:, 0], preds_cm[:, 0])),
        "mae_y_cm": float(mean_absolute_error(trues[:, 1], preds_cm[:, 1])),
        "r2_x": float(r2_score(trues[:, 0], preds_cm[:, 0])),
        "r2_y": float(r2_score(trues[:, 1], preds_cm[:, 1])),
        "mean_distance": float(dist.mean()),
        "median_distance": float(np.median(dist)),
        "p90_distance": float(np.percentile(dist, 90)),
        f"fraction_within_{cfg.model.threshold_cm}cm": float((dist <= cfg.model.threshold_cm).mean())
    }

    with open(out_dir / "mmd_minimal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Final hold-out evaluation:")
    logger.info(metrics)

if __name__ == "__main__":
    main()
