#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temp-aware 1D-CNN Residual Equalizer for FMF DSP
单独版本 - 仅包含 CNN 均衡器，无传统基线对比

数据集格式:
X:   (N, 60)   -> 重构为 (N, 5, 6, 2)
Y:   (N, 16)   one-hot
AUX: (N, 5)    默认 AUX[:,0] 为温度

输出:
- 全测试集指标 (Accuracy, SER, BER, EVM)
- 低温段 (-150C ~ -50C) 指标
- 星座图
- Accuracy vs Temperature
- 训练曲线
- JSON 指标文件
"""
import os
import json
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# 0) 基础工具
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def complex_from_iq(x_iq: np.ndarray) -> np.ndarray:
    return x_iq[..., 0] + 1j * x_iq[..., 1]


def iq_from_complex(z: np.ndarray) -> np.ndarray:
    return np.stack([np.real(z), np.imag(z)], axis=-1)


# =========================
# 1) 16QAM 星座图定义
# =========================
def qam16_constellation_and_bits() -> Tuple[np.ndarray, np.ndarray]:
    """
    Gray mapping per dimension:
    00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    """
    bit_pairs = [(0, 0), (0, 1), (1, 1), (1, 0)]
    level_map = {
        (0, 0): -3, (0, 1): -1, (1, 1): +1, (1, 0): +3,
    }
    pts = []
    bits = []
    for qb in bit_pairs:
        for ib in bit_pairs:
            i_level = level_map[ib]
            q_level = level_map[qb]
            pts.append(complex(i_level, q_level))
            bits.append([ib[0], ib[1], qb[0], qb[1]])
    pts = np.asarray(pts, dtype=np.complex64)
    bits = np.asarray(bits, dtype=np.uint8)
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts, bits


# =========================
# 2) 数据加载与重构
# =========================
def load_npz(npz_path: str):
    data = np.load(npz_path)
    keys = list(data.keys())
    if "X" not in data or "Y" not in data or "AUX" not in data:
        raise ValueError(f"NPZ 文件必须包含 X/Y/AUX，当前 keys={keys}")
    X = data["X"]
    Y = data["Y"]
    AUX = data["AUX"]
    return X, Y, AUX, keys


def infer_structure(X: np.ndarray, Y: np.ndarray):
    n, d = X.shape
    if d == 60:
        return {
            "window_len": 5,
            "num_modes": 6,
            "reshape_order": "tm",
            "center": 2,
            "target_mode": 0,
        }
    raise ValueError(f"当前脚本仅适配 X.shape[1]=60 的数据，实际 d={d}")


def reconstruct_cube(X: np.ndarray, window_len: int, num_modes: int,
                     reshape_order: str = "tm"):
    N, D = X.shape
    if D != window_len * num_modes * 2:
        raise ValueError(f"D={D} 与 W*M*2={window_len * num_modes * 2} 不一致")
    if reshape_order == "tm":
        X_cube = X.reshape(N, window_len, num_modes, 2)
    elif reshape_order == "mt":
        X_cube = X.reshape(N, num_modes, window_len, 2).transpose(0, 2, 1, 3)
    else:
        raise ValueError(f"未知 reshape_order={reshape_order}")
    return X_cube.astype(np.float32)


def train_val_test_split_stratified_by_group(
        groups: np.ndarray,
        seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    all_idx = np.arange(len(groups))
    train_idx, val_idx, test_idx = [], [], []
    for g in np.unique(groups):
        idx = all_idx[groups == g]
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train = max(1, int(n * 0.6))
            n_val = max(1, int(n * 0.2))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:n_train + n_val + n_test])
    return (np.array(sorted(train_idx)),
            np.array(sorted(val_idx)),
            np.array(sorted(test_idx)))


# =========================
# 3) 温度分组
# =========================
def build_temperature_groups(temps: np.ndarray, temp_group_step: float) -> np.ndarray:
    if temp_group_step is None or temp_group_step <= 0:
        return temps.astype(np.float32)
    return (np.round(temps / temp_group_step) * temp_group_step).astype(np.float32)


# =========================
# 4) Class 到标准星座图映射对齐
# =========================
def greedy_unique_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_class, n_ref = cost.shape
    used_r, used_c = set(), set()
    pairs = []
    flat = []
    for i in range(n_class):
        for j in range(n_ref):
            flat.append((cost[i, j], i, j))
    flat.sort(key=lambda x: x[0])
    for c, i, j in flat:
        if i in used_r or j in used_c:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
        if len(pairs) == min(n_class, n_ref):
            break
    row_idx = np.array([p[0] for p in pairs], dtype=int)
    col_idx = np.array([p[1] for p in pairs], dtype=int)
    return row_idx, col_idx


def estimate_class_means_from_center(
        X_cube: np.ndarray, y_idx: np.ndarray,
        center: int, target_mode: int,
) -> np.ndarray:
    z = complex_from_iq(X_cube[:, center, target_mode, :])
    class_means = np.zeros(16, dtype=np.complex64)
    for c in range(16):
        mask = (y_idx == c)
        if np.any(mask):
            class_means[c] = np.mean(z[mask])
        else:
            class_means[c] = 0.0 + 0.0j
    return class_means


def estimate_class_mapping(
        X_cube_train: np.ndarray, y_idx_train: np.ndarray,
        center: int, target_mode: int,
        save_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    canonical_pts, canonical_bits = qam16_constellation_and_bits()
    class_means = estimate_class_means_from_center(
        X_cube_train, y_idx_train, center, target_mode
    )
    valid = np.abs(class_means) > 0
    mu = class_means[valid]
    scale = np.sqrt(np.mean(np.abs(canonical_pts) ** 2) /
                    (np.mean(np.abs(mu) ** 2) + 1e-12))
    class_means_aligned = class_means * scale
    cost = np.abs(class_means_aligned[:, None] - canonical_pts[None, :]) ** 2
    try:
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost)
    except Exception:
        row_idx, col_idx = greedy_unique_assignment(cost)
    class_to_canonical = np.zeros(16, dtype=np.int64)
    for r, c in zip(row_idx, col_idx):
        class_to_canonical[r] = c
    class_constellation = canonical_pts[class_to_canonical]
    class_bits = canonical_bits[class_to_canonical]
    if save_dir is not None:
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(np.real(class_means_aligned), np.imag(class_means_aligned),
                    s=90, label="Class mean")
        plt.scatter(np.real(canonical_pts), np.imag(canonical_pts),
                    s=90, label="Canonical 16QAM")
        for c in range(16):
            plt.text(np.real(class_means_aligned[c]) + 0.02,
                     np.imag(class_means_aligned[c]) + 0.02, str(c), fontsize=10)
        plt.title("Class-Mean to Canonical 16QAM Mapping Check")
        plt.xlabel("In-phase")
        plt.ylabel("Quadrature")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mapping_consistency.png"), dpi=180)
        plt.close(fig)
    return {
        "canonical_pts": canonical_pts,
        "canonical_bits": canonical_bits,
        "class_means": class_means,
        "class_means_aligned": class_means_aligned,
        "class_to_canonical": class_to_canonical,
        "class_constellation": class_constellation,
        "class_bits": class_bits,
    }


# =========================
# 5) 指标计算
# =========================
def nearest_class_decision(z_hat: np.ndarray,
                           class_constellation: np.ndarray) -> np.ndarray:
    dist = np.abs(z_hat[:, None] - class_constellation[None, :]) ** 2
    return np.argmin(dist, axis=1)


def compute_metrics(
        z_hat: np.ndarray, y_true_idx: np.ndarray,
        class_constellation: np.ndarray, class_bits: np.ndarray,
):
    y_pred_idx = nearest_class_decision(z_hat, class_constellation)
    acc = float(np.mean(y_pred_idx == y_true_idx))
    ser = 1.0 - acc
    bits_true = class_bits[y_true_idx]
    bits_pred = class_bits[y_pred_idx]
    ber = float(np.mean(bits_true != bits_pred))
    z_ref = class_constellation[y_true_idx]
    evm_rms = float(np.sqrt(np.mean(np.abs(z_hat - z_ref) ** 2) /
                            (np.mean(np.abs(z_ref) ** 2) + 1e-12)))
    return {
        "accuracy": acc, "SER": float(ser), "BER": float(ber), "EVM": evm_rms,
        "y_pred_idx": y_pred_idx,
    }


def compute_metrics_by_temp(
        z_hat: np.ndarray, y_true_idx: np.ndarray, temp_group: np.ndarray,
        class_constellation: np.ndarray, class_bits: np.ndarray,
):
    out = {}
    for g in np.unique(temp_group):
        mask = (temp_group == g)
        if np.sum(mask) == 0:
            continue
        m = compute_metrics(z_hat[mask], y_true_idx[mask],
                            class_constellation, class_bits)
        out[float(g)] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in m.items() if k != "y_pred_idx"}
    return out


def compute_metrics_on_mask(
        z_hat: np.ndarray, y_true_idx: np.ndarray, temps: np.ndarray,
        class_constellation: np.ndarray, class_bits: np.ndarray,
        temp_min: float = -150.0, temp_max: float = -50.0,
):
    mask = (temps >= temp_min) & (temps <= temp_max)
    if np.sum(mask) == 0:
        return None
    m = compute_metrics(z_hat[mask], y_true_idx[mask],
                        class_constellation, class_bits)
    out = {k: v for k, v in m.items() if k != "y_pred_idx"}
    out["num_samples"] = int(np.sum(mask))
    out["temp_range"] = [float(temp_min), float(temp_max)]
    return out


# =========================
# 6) Dataset & Model (CNN 核心)
# =========================
class ResidualDataset(Dataset):
    def __init__(
            self, raw_cube: np.ndarray, raw_center: np.ndarray, temps: np.ndarray,
            target_iq: np.ndarray, temp_mean: float, temp_std: float,
    ):
        self.raw_cube = raw_cube.astype(np.float32)
        self.raw_center = raw_center.astype(np.float32)
        self.temps = temps.astype(np.float32)
        self.target_iq = target_iq.astype(np.float32)
        self.temp_mean = float(temp_mean)
        self.temp_std = float(temp_std) if temp_std > 1e-12 else 1.0

    def __len__(self):
        return len(self.raw_cube)

    def __getitem__(self, idx):
        x = self.raw_cube[idx]  # (W, M, 2)
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(f"期望单样本形状为 (W, M, 2)，实际得到 {x.shape}")
        x = np.transpose(x, (1, 2, 0))  # (M, 2, W)
        x = x.reshape(-1, x.shape[-1])  # (2*M, W)
        t_norm = np.asarray(
            [(self.temps[idx] - self.temp_mean) / self.temp_std],
            dtype=np.float32
        )
        return (
            torch.from_numpy(x),  # (2*M, W)
            torch.from_numpy(self.raw_center[idx]),  # (2,)
            torch.from_numpy(t_norm),  # (1,)
            torch.from_numpy(self.target_iq[idx]),  # (2,)
        )


class ResidualCNN1D(nn.Module):
    def __init__(
            self, num_modes: int, window_len: int, num_classes: int = 16,
            conv_channels: int = 64, hidden_dim: int = 128, dropout: float = 0.05,
    ):
        super().__init__()
        in_ch = 2 * num_modes
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_ch, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        feat_dim = conv_channels * window_len + 2 + 1
        self.residual_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_seq, raw_center, temp_norm):
        feat = self.feature_extractor(x_seq)  # (B, C, W)
        feat = feat.reshape(feat.size(0), -1)  # (B, C*W)
        feat = self.dropout(feat)
        feat = torch.cat([feat, raw_center, temp_norm], dim=1)
        residual = self.residual_head(feat)
        logits = self.class_head(feat)
        return residual, logits


@dataclass
class TrainOutput:
    model: nn.Module
    history: Dict[str, List[float]]
    best_val_loss: float


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=0, drop_last=False)


def nearest_logits_from_constellation(
        pred_iq: torch.Tensor, class_constellation_torch: torch.Tensor
):
    diff = pred_iq.unsqueeze(1) - class_constellation_torch.unsqueeze(0)
    dist2 = torch.sum(diff * diff, dim=-1)
    logits = -dist2
    return logits


def run_epoch(
        model, loader, optimizer, device, class_constellation_torch,
        alpha_reg=1.0, beta_ce=0.2,
):
    model.train()
    total_loss, total_n = 0.0, 0
    reg_loss_fn = nn.SmoothL1Loss()
    ce_loss_fn = nn.CrossEntropyLoss()
    for x_seq, raw_center, temp_norm, target_iq in loader:
        x_seq = x_seq.to(device)
        raw_center = raw_center.to(device)
        temp_norm = temp_norm.to(device)
        target_iq = target_iq.to(device)
        residual, _ = model(x_seq, raw_center, temp_norm)
        pred = raw_center + residual
        loss_reg = reg_loss_fn(pred, target_iq)
        logits_dist = nearest_logits_from_constellation(pred, class_constellation_torch)
        target_class = torch.argmin(
            torch.sum((target_iq.unsqueeze(1) - class_constellation_torch.unsqueeze(0)) ** 2,
                      dim=-1), dim=1
        )
        loss_ce = ce_loss_fn(logits_dist, target_class)
        loss = alpha_reg * loss_reg + beta_ce * loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bs = x_seq.size(0)
        total_loss += loss.item() * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def eval_epoch(
        model, loader, device, class_constellation_torch,
        alpha_reg=1.0, beta_ce=0.2,
):
    model.eval()
    total_loss, total_n = 0.0, 0
    reg_loss_fn = nn.SmoothL1Loss()
    ce_loss_fn = nn.CrossEntropyLoss()
    preds = []
    for x_seq, raw_center, temp_norm, target_iq in loader:
        x_seq = x_seq.to(device)
        raw_center = raw_center.to(device)
        temp_norm = temp_norm.to(device)
        target_iq = target_iq.to(device)
        residual, _ = model(x_seq, raw_center, temp_norm)
        pred = raw_center + residual
        loss_reg = reg_loss_fn(pred, target_iq)
        logits_dist = nearest_logits_from_constellation(pred, class_constellation_torch)
        target_class = torch.argmin(
            torch.sum((target_iq.unsqueeze(1) - class_constellation_torch.unsqueeze(0)) ** 2,
                      dim=-1), dim=1
        )
        loss_ce = ce_loss_fn(logits_dist, target_class)
        loss = alpha_reg * loss_reg + beta_ce * loss_ce
        bs = x_seq.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0) if len(preds) > 0 else np.zeros((0, 2), dtype=np.float32)
    return total_loss / max(total_n, 1), preds


def train_residual_model(
        train_dataset: ResidualDataset, val_dataset: ResidualDataset,
        num_modes: int, window_len: int, class_constellation: np.ndarray,
        conv_channels: int = 64, hidden_dim: int = 128, dropout: float = 0.05,
        lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 256,
        epochs: int = 40, patience: int = 8, device: str = "cpu",
        alpha_reg: float = 1.0, beta_ce: float = 0.2,
):
    model = ResidualCNN1D(
        num_modes=num_modes, window_len=window_len, num_classes=16,
        conv_channels=conv_channels, hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = build_loader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=int(batch_size), shuffle=False)
    class_constellation_iq = iq_from_complex(class_constellation).astype(np.float32)
    class_constellation_torch = torch.from_numpy(class_constellation_iq).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": []}
    for _ in range(1, epochs + 1):
        tr_loss = run_epoch(
            model, train_loader, optimizer, device,
            class_constellation_torch=class_constellation_torch,
            alpha_reg=alpha_reg, beta_ce=beta_ce,
        )
        va_loss, _ = eval_epoch(
            model, val_loader, device,
            class_constellation_torch=class_constellation_torch,
            alpha_reg=alpha_reg, beta_ce=beta_ce,
        )
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return TrainOutput(model=model, history=history, best_val_loss=float(best_val))


@torch.no_grad()
def predict_residual_model(
        model: nn.Module, dataset: ResidualDataset,
        batch_size: int = 512, device: str = "cpu",
):
    loader = build_loader(dataset, batch_size=int(batch_size), shuffle=False)
    model.eval()
    preds = []
    for x_seq, raw_center, temp_norm, _ in loader:
        x_seq = x_seq.to(device)
        raw_center = raw_center.to(device)
        temp_norm = temp_norm.to(device)
        residual, _ = model(x_seq, raw_center, temp_norm)
        pred = raw_center + residual
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if len(preds) > 0 else np.zeros((0, 2), dtype=np.float32)


# =========================
# 7) 绘图函数
# =========================
def plot_constellation(z: np.ndarray, z_ref: np.ndarray, title: str,
                       path: str, max_points: int = 1200):
    if len(z) > max_points:
        idx = np.random.choice(len(z), size=max_points, replace=False)
        z_plot, z_ref_plot = z[idx], z_ref[idx]
    else:
        z_plot, z_ref_plot = z, z_ref
    fig = plt.figure(figsize=(7.2, 7.2))
    plt.scatter(np.real(z_plot), np.imag(z_plot), s=15, alpha=0.55, label="Estimated")
    plt.scatter(np.real(z_ref_plot), np.imag(z_ref_plot), s=35, alpha=0.9, label="Ideal")
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def plot_constellation_lowtemp(
        z: np.ndarray, z_ref: np.ndarray, temps: np.ndarray,
        title: str, path: str,
        temp_min: float = -150.0, temp_max: float = -50.0,
        max_points: int = 1200,
):
    mask = (temps >= temp_min) & (temps <= temp_max)
    z, z_ref = z[mask], z_ref[mask]
    if len(z) == 0:
        return
    if len(z) > max_points:
        idx = np.random.choice(len(z), size=max_points, replace=False)
        z, z_ref = z[idx], z_ref[idx]
    fig = plt.figure(figsize=(7.2, 7.2))
    plt.scatter(np.real(z), np.imag(z), s=15, alpha=0.55, label="Estimated")
    plt.scatter(np.real(z_ref), np.imag(z_ref), s=35, alpha=0.9, label="Ideal")
    plt.title(title)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def plot_accuracy_vs_temperature(acc_curves: Dict[str, Dict[float, dict]], path: str):
    fig = plt.figure(figsize=(10, 6))
    for name, curve in acc_curves.items():
        temps = sorted(curve.keys())
        accs = [curve[t]["accuracy"] for t in temps]
        plt.plot(temps, accs, marker="o", label=name)
    plt.title("Accuracy vs Temperature")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_curve(history: Dict[str, List[float]], path: str, title: str):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)


def plot_lowtemp_bar(metrics_dict: Dict[str, dict], path: str):
    names = list(metrics_dict.keys())
    accs = [metrics_dict[k]["accuracy"] for k in names]
    bers = [metrics_dict[k]["BER"] for k in names]
    x = np.arange(len(names))
    width = 0.35
    fig = plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, accs, width, label="Accuracy")
    plt.bar(x + width / 2, bers, width, label="BER")
    plt.xticks(x, names, rotation=10)
    plt.ylim(0, 1.05)
    plt.title("Low-temperature (-150C ~ -50C) Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close(fig)

def save_prediction_pack(
        save_dir: str,
        center: int,
        target_mode: int,
        temp_mean: float,
        temp_std: float,
        test_idx: np.ndarray,
        X_cube_test: np.ndarray,
        y_test: np.ndarray,
        Y_test_onehot: np.ndarray,
        t_test: np.ndarray,
        tg_test: np.ndarray,
        y_raw_test_iq: np.ndarray,
        y_cnn_test_iq: np.ndarray,
        class_constellation: np.ndarray,
        class_bits: np.ndarray,
        class_to_canonical: np.ndarray,
):
    z_raw_test = complex_from_iq(y_raw_test_iq).astype(np.complex64)
    z_cnn_test = complex_from_iq(y_cnn_test_iq).astype(np.complex64)
    z_ref_test = class_constellation[y_test].astype(np.complex64)

    structured_path = os.path.join(save_dir, "structured_test_subset_for_residual_xt.npz")
    np.savez(
        structured_path,
        test_indices=test_idx.astype(np.int64),
        X_cube_test=X_cube_test.astype(np.float32),
        y_test_idx=y_test.astype(np.int64),
        Y_test_onehot=Y_test_onehot.astype(np.float32),
        temps_test=t_test.astype(np.float32),
        temp_group_test=tg_test.astype(np.float32),
        center=np.int64(center),
        target_mode=np.int64(target_mode),
    )

    pred_path = os.path.join(save_dir, "cnn_predictions_for_residual_xt.npz")
    np.savez(
        pred_path,
        test_indices=test_idx.astype(np.int64),
        temps_test=t_test.astype(np.float32),
        temp_group_test=tg_test.astype(np.float32),
        y_test_idx=y_test.astype(np.int64),
        raw_center_iq_test=y_raw_test_iq.astype(np.float32),
        cnn_pred_iq_test=y_cnn_test_iq.astype(np.float32),
        z_raw_test=z_raw_test,
        z_cnn_test=z_cnn_test,
        z_ref_test=z_ref_test,
        class_constellation=class_constellation.astype(np.complex64),
        class_constellation_iq=iq_from_complex(class_constellation).astype(np.float32),
        class_bits=class_bits.astype(np.uint8),
        class_to_canonical=class_to_canonical.astype(np.int64),
        temp_mean=np.float32(temp_mean),
        temp_std=np.float32(temp_std),
        center=np.int64(center),
        target_mode=np.int64(target_mode),
    )
    return structured_path, pred_path

# =========================
# 8) 主流程
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", type=str,
                        default=r"E:\pycharm file\fiberproject\fiberdsp\fmf_dataset_from_sparseT_Sij.npz")
    parser.add_argument("--save_dir", type=str,
                        default=r"E:\pycharm file\fiberproject\fiberdsp\runs_cnn_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp_col", type=int, default=0)
    parser.add_argument("--temp_group_step", type=float, default=5.0)
    parser.add_argument("--conv_channels", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--alpha_reg", type=float, default=1.0)
    parser.add_argument("--beta_ce", type=float, default=0.2)
    parser.add_argument("--lowtemp_min", type=float, default=-150.0)
    parser.add_argument("--lowtemp_max", type=float, default=-50.0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.save_dir)

    # -------- 1) 加载数据 --------
    X, Y, AUX, keys = load_npz(args.npz_path)
    y_idx = np.argmax(Y, axis=1).astype(np.int64)
    temps = AUX[:, args.temp_col].astype(np.float32)
    info = infer_structure(X, Y)
    W, M = info["window_len"], info["num_modes"]
    center, target_mode = info["center"], info["target_mode"]
    reshape_order = info["reshape_order"]
    X_cube = reconstruct_cube(X, W, M, reshape_order=reshape_order)
    np.savez(
        os.path.join(args.save_dir, "step1_structured_dataset.npz"),
        X_cube=X_cube, Y=Y, AUX=AUX, temps=temps,
    )
    print("Step 1 完成：已生成结构化数据集")
    print(f"  X_cube shape: {X_cube.shape}  (N, W, M, 2)")

    # -------- 2) 数据集划分 --------
    temp_group = build_temperature_groups(temps, args.temp_group_step)
    train_idx, val_idx, test_idx = train_val_test_split_stratified_by_group(
        temp_group, seed=args.seed
    )
    X_cube_train, X_cube_val, X_cube_test = X_cube[train_idx], X_cube[val_idx], X_cube[test_idx]
    y_train, y_val, y_test = y_idx[train_idx], y_idx[val_idx], y_idx[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    t_train, t_val, t_test = temps[train_idx], temps[val_idx], temps[test_idx]
    tg_train, tg_val, tg_test = temp_group[train_idx], temp_group[val_idx], temp_group[test_idx]

    # -------- 3) 星座图映射对齐 --------
    mapping = estimate_class_mapping(
        X_cube_train, y_train, center=center, target_mode=target_mode,
        save_dir=args.save_dir,
    )
    class_constellation = mapping["class_constellation"]
    class_bits = mapping["class_bits"]
    z_target_test = class_constellation[y_test]
    y_target_test = iq_from_complex(z_target_test).astype(np.float32)

    # -------- 4) Raw baseline (仅用于对比) --------
    y_raw_test = X_cube_test[:, center, target_mode, :].astype(np.float32)
    z_raw_test = complex_from_iq(y_raw_test)
    raw_metrics_test = compute_metrics(z_raw_test, y_test, class_constellation, class_bits)

    # -------- 5) CNN 训练 --------
    z_target_train = class_constellation[y_train]
    z_target_val = class_constellation[y_val]
    y_target_train = iq_from_complex(z_target_train).astype(np.float32)
    y_target_val = iq_from_complex(z_target_val).astype(np.float32)
    y_raw_train = X_cube_train[:, center, target_mode, :].astype(np.float32)
    y_raw_val = X_cube_val[:, center, target_mode, :].astype(np.float32)
    y_raw_test = X_cube_test[:, center, target_mode, :].astype(np.float32)

    temp_mean = float(np.mean(t_train))
    temp_std = float(np.std(t_train) + 1e-12)
    ds_train = ResidualDataset(X_cube_train, y_raw_train, t_train,
                               y_target_train, temp_mean, temp_std)
    ds_val = ResidualDataset(X_cube_val, y_raw_val, t_val,
                             y_target_val, temp_mean, temp_std)
    ds_test = ResidualDataset(X_cube_test, y_raw_test, t_test,
                              y_target_test, temp_mean, temp_std)

    print(f"\n开始训练 CNN 模型 (device={args.device})...")
    out_tempaware = train_residual_model(
        ds_train, ds_val, num_modes=M, window_len=W,
        class_constellation=class_constellation,
        conv_channels=args.conv_channels, hidden_dim=args.hidden_dim,
        dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
        batch_size=int(args.batch_size), epochs=args.epochs, patience=args.patience,
        device=args.device, alpha_reg=args.alpha_reg, beta_ce=args.beta_ce,
    )
    print(f"训练完成！最佳验证损失：{out_tempaware.best_val_loss:.6f}")

    # -------- 6) CNN 预测 --------
    y_nn_test = predict_residual_model(
        out_tempaware.model, ds_test, batch_size=args.batch_size, device=args.device,
    )
    z_nn_test = complex_from_iq(y_nn_test)
    proposed_metrics_test = compute_metrics(z_nn_test, y_test, class_constellation, class_bits)
    proposed_by_temp_test = compute_metrics_by_temp(
        z_nn_test, y_test, tg_test, class_constellation, class_bits
    )

    # -------- 7) 低温段评估 --------
    raw_lowtemp_test = compute_metrics_on_mask(
        z_raw_test, y_test, t_test, class_constellation, class_bits,
        temp_min=args.lowtemp_min, temp_max=args.lowtemp_max
    )
    proposed_lowtemp_test = compute_metrics_on_mask(
        z_nn_test, y_test, t_test, class_constellation, class_bits,
        temp_min=args.lowtemp_min, temp_max=args.lowtemp_max
    )

    structured_test_subset_path, pred_pack_path = save_prediction_pack(
        save_dir=args.save_dir,
        center=center,
        target_mode=target_mode,
        temp_mean=temp_mean,
        temp_std=temp_std,
        test_idx=test_idx,
        X_cube_test=X_cube_test,
        y_test=y_test,
        Y_test_onehot=Y_test,
        t_test=t_test,
        tg_test=tg_test,
        y_raw_test_iq=y_raw_test,
        y_cnn_test_iq=y_nn_test,
        class_constellation=class_constellation,
        class_bits=class_bits,
        class_to_canonical=mapping["class_to_canonical"],
    )

    split_path = os.path.join(args.save_dir, "dataset_split_indices.npz")
    np.savez(
        split_path,
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        temps=temps.astype(np.float32),
        temp_group=temp_group.astype(np.float32),
        center=np.int64(center),
        target_mode=np.int64(target_mode),
    )

    # -------- 8) 保存结果 --------
    metrics_pack = {
        "dataset": {
            "npz_path": args.npz_path, "keys": keys,
            "X_shape": list(X.shape), "Y_shape": list(Y.shape),
            "AUX_shape": list(AUX.shape), "W": W, "M": M,
            "center": center, "target_mode": target_mode,
            "train_size": int(len(train_idx)), "val_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
        },
        "normalization": {
            "temp_mean": temp_mean,
            "temp_std": temp_std,
        },
        "saved_artifacts": {
            "structured_dataset_npz": os.path.join(args.save_dir, "step1_structured_dataset.npz"),
            "split_indices_npz": split_path,
            "structured_test_subset_for_residual_xt": structured_test_subset_path,
            "cnn_predictions_for_residual_xt": pred_pack_path,
        },
        "raw_test": {k: v for k, v in raw_metrics_test.items() if k != "y_pred_idx"},
        "cnn_test": {k: v for k, v in proposed_metrics_test.items() if k != "y_pred_idx"},
        "cnn_by_temp_test": proposed_by_temp_test,
        "raw_lowtemp_test": raw_lowtemp_test,
        "cnn_lowtemp_test": proposed_lowtemp_test,
    }
    save_json(metrics_pack, os.path.join(args.save_dir, "metrics_cnn_only.json"))

    # -------- 9) 绘图 --------
    plot_training_curve(
        out_tempaware.history,
        os.path.join(args.save_dir, "training_curve_cnn.png"),
        "Temp-aware 1D-CNN Residual Training Curve",
    )
    plot_constellation(
        z_raw_test, class_constellation[y_test],
        "Before Equalization (Raw Center-tap)",
        os.path.join(args.save_dir, "constellation_before.png"),
    )
    plot_constellation(
        z_nn_test, class_constellation[y_test],
        "After CNN Equalization",
        os.path.join(args.save_dir, "constellation_after_cnn.png"),
    )
    plot_constellation_lowtemp(
        z_nn_test, class_constellation[y_test], t_test,
        f"Low-temperature CNN ({args.lowtemp_min:.0f}C ~ {args.lowtemp_max:.0f}C)",
        os.path.join(args.save_dir, "constellation_lowtemp_cnn.png"),
        temp_min=args.lowtemp_min, temp_max=args.lowtemp_max,
    )
    acc_curves = {
        "Raw center-tap": compute_metrics_by_temp(
            z_raw_test, y_test, tg_test, class_constellation, class_bits
        ),
        "CNN Proposed": proposed_by_temp_test,
    }
    plot_accuracy_vs_temperature(
        acc_curves, os.path.join(args.save_dir, "accuracy_vs_temperature.png"),
    )
    plot_lowtemp_bar(
        {"Raw": raw_lowtemp_test, "CNN": proposed_lowtemp_test},
        os.path.join(args.save_dir, "lowtemp_comparison_bar.png"),
    )

    # -------- 10) 打印结果 --------
    print("\n" + "=" * 60)
    print("全测试集指标：")
    print("  Raw center-tap baseline:")
    print(f"    Accuracy = {metrics_pack['raw_test']['accuracy']:.6f}")
    print(f"    BER      = {metrics_pack['raw_test']['BER']:.6f}")
    print(f"    EVM      = {metrics_pack['raw_test']['EVM']:.6f}")
    print("  CNN Proposed:")
    print(f"    Accuracy = {metrics_pack['cnn_test']['accuracy']:.6f}")
    print(f"    BER      = {metrics_pack['cnn_test']['BER']:.6f}")
    print(f"    EVM      = {metrics_pack['cnn_test']['EVM']:.6f}")
    print("\n低温段指标：")
    print("  Raw center-tap baseline:")
    print(f"    Accuracy = {metrics_pack['raw_lowtemp_test']['accuracy']:.6f}")
    print(f"    BER      = {metrics_pack['raw_lowtemp_test']['BER']:.6f}")
    print(f"    EVM      = {metrics_pack['raw_lowtemp_test']['EVM']:.6f}")
    print("  CNN Proposed:")
    print(f"    Accuracy = {metrics_pack['cnn_lowtemp_test']['accuracy']:.6f}")
    print(f"    BER      = {metrics_pack['cnn_lowtemp_test']['BER']:.6f}")
    print(f"    EVM      = {metrics_pack['cnn_lowtemp_test']['EVM']:.6f}")
    print("=" * 60)
    print(f"\n结果已保存到：{args.save_dir}")
    print("  - metrics_cnn_only.json")
    print("  - step1_structured_dataset.npz")
    print("  - dataset_split_indices.npz")
    print("  - structured_test_subset_for_residual_xt.npz")
    print("  - cnn_predictions_for_residual_xt.npz")
    print("  - training_curve_cnn.png")
    print("  - constellation_*.png")
    print("  - accuracy_vs_temperature.png")
    print("  - lowtemp_comparison_bar.png")

if __name__ == "__main__":
    main()