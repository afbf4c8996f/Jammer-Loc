# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

logger = logging.getLogger(__name__)
model_registry = {}

class ResMLPBlock(nn.Module):
    """
    Residual MLP block with pre-activation, dropout, and skip connection.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.act(out)
        out = self.fc1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        return x + out


class SimpleNN(nn.Module):
    """
    Residual MLP shared for classification & regression.

    Structure:
      - Project input -> hidden_size/2
      - num_blocks of ResMLPBlock at width hidden_size/2
      - Project -> hidden_size
      - num_blocks of ResMLPBlock at width hidden_size
      - Final head -> num_outputs
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_outputs: int,
        dropout: float = 0.1,
        num_blocks: int = 2,
    ):
        super().__init__()
        half = hidden_size // 2
        self.proj1 = nn.Linear(input_size, half)
        self.blocks1 = nn.ModuleList([
            ResMLPBlock(half, dropout) for _ in range(num_blocks)
        ])
        self.proj2 = nn.Linear(half, hidden_size)
        self.blocks2 = nn.ModuleList([
            ResMLPBlock(hidden_size, dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.proj2(x)
        for blk in self.blocks2:
            x = blk(x)
        return self.head(x)

model_registry["simple_nn"] = SimpleNN


class ConvMixer1D(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_outputs: int,
        kernel_size: int = 3,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pw0 = nn.Conv1d(1, hidden_size, 1)
        blocks = []
        for _ in range(num_blocks):
            blocks += [
                nn.Conv1d(hidden_size, hidden_size,
                          kernel_size, padding=kernel_size//2,
                          groups=hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        self.mixer = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B,1,F)
        x = F.gelu(self.pw0(x))        # (B,hidden,F)
        x = self.mixer(x)              # (B,hidden,F)
        x = self.pool(x).squeeze(-1)   # (B,hidden)
        return self.fc(x)              # (B,num_outputs)

model_registry["convmixer1d"] = ConvMixer1D

class TabularTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_outputs: int,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.input_size = input_size
        self.embed = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            assert x.shape[1] == 1, f"Expected dim1=1, got {x.shape[1]}"
            x = x.squeeze(1)
        elif x.dim() != 2:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        B, F = x.shape
        assert F == self.input_size, f"Expected {self.input_size} features, got {F}"
        x = x.unsqueeze(1)         # (B,1,F)
        x = self.embed(x)          # (B,1,hidden)
        x = self.transformer(x)    # (B,1,hidden)
        x = x.squeeze(1)           # (B,hidden)
        return self.fc(x)          # (B,num_outputs)

model_registry["transformer_tabular"] = TabularTransformer
    



def get_model(
    model_name: str,
    input_size: int,
    hidden_size: int,
    num_classes: Optional[int] = None,
    num_outputs: Optional[int] = None,
    **kwargs
) -> nn.Module:
    name = model_name.lower()
    if num_outputs is not None:
        final_dim = num_outputs
    else:
        final_dim = num_classes if num_classes is not None else (
            2 if name == "unet1d" else input_size
        )
    if name not in model_registry:
        logger.error(f"Model '{model_name}' not in registry.")
        raise ValueError(f"Unknown model '{model_name}'")
    cls = model_registry[name]
    return cls(input_size, hidden_size, final_dim, **kwargs)