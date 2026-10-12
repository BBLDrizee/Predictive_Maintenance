"""LSTM and 1D-CNN RUL models in PyTorch, trained on raw windows with early stopping (FR-5)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, window_len: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class CNNNet(nn.Module):
    def __init__(self, n_features: int, window_len: int, channels: int = 32, n_conv: int = 3, kernel: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = n_features
        for _ in range(n_conv):
            layers += [nn.Conv1d(c_in, channels, kernel, padding="same"), nn.ReLU()]
            c_in = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels * window_len, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)  # Conv1d wants (batch, channels, time)


ARCHITECTURES: dict[str, type[nn.Module]] = {"lstm": LSTMNet, "cnn": CNNNet}


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DeepModel:
    """Train/predict wrapper. Targets are scaled by ``rul_cap`` for stable optimisation."""

    def __init__(self, arch: str, n_features: int, window_len: int, rul_cap: float, seed: int = 0,
                 lr: float = 1e-3, batch_size: int = 256, max_epochs: int = 80, patience: int = 10, **arch_params: Any) -> None:
        seed_everything(seed)
        self.net = ARCHITECTURES[arch](n_features, window_len, **arch_params)
        self.scale = float(rul_cap)
        self.lr, self.batch_size, self.max_epochs, self.patience = lr, batch_size, max_epochs, patience
        self.generator = torch.Generator().manual_seed(seed)
        self.history: list[float] = []
        self.best_epoch = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["generator"] = None  # torch.Generator is not picklable; only needed for training
        return state

    def _val_rmse(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(np.mean((self.predict(x) - y) ** 2)))

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            on_epoch: Callable[[int, DeepModel], None] | None = None) -> DeepModel:
        """Adam + MSE; stop when validation RMSE has not improved for ``patience`` epochs and restore the best weights.

        ``on_epoch(epoch, model)`` lets a caller (e.g. Optuna pruning) observe or stop training by raising.
        """
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y / self.scale).float()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best, best_state, waited = float("inf"), copy.deepcopy(self.net.state_dict()), 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(xt), generator=self.generator)
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            score = self._val_rmse(x_val, y_val)
            self.history.append(score)
            if score < best - 1e-4:
                best, best_state, waited, self.best_epoch = score, copy.deepcopy(self.net.state_dict()), 0, epoch
            else:
                waited += 1
            if on_epoch is not None:
                on_epoch(epoch, self)
            if waited >= self.patience:
                break
        self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.net.eval()
        xt = torch.from_numpy(x)
        out = [self.net(xt[i:i + batch_size]) for i in range(0, len(xt), batch_size)]
        return torch.cat(out).numpy() * self.scale

    @torch.no_grad()
    def predict_mc(self, x: np.ndarray, n_samples: int = 50, seed: int = 0, batch_size: int = 4096) -> np.ndarray:
        """Monte Carlo dropout: (n_samples, n) predictions with dropout left on (FR-8)."""
        torch.manual_seed(seed)
        self.net.train()
        xt = torch.from_numpy(x)
        samples = [torch.cat([self.net(xt[i:i + batch_size]) for i in range(0, len(xt), batch_size)]) for _ in range(n_samples)]
        self.net.eval()
        return torch.stack(samples).numpy() * self.scale

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
