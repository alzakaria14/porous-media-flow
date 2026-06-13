"""
Reproducible multi-seed validation for the porous-media NN vs PINN study.

This script is intentionally independent from the notebook so the revised
manuscript can point to a repeatable local experiment. It keeps the same
physical case used in the notebook: a 50 m x 50 m saturated diffusion benchmark
with multi-cell rainfall forcing, FDM reference solution, sparse observations at
hours 1-3, and forecast evaluation at hours 4-6.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim


Lx, Ly = 50.0, 50.0
T_END = 6.0
NX, NY = 51, 51
TARGET_DT_HOURS = 0.10
INITIAL_HEAD = 1.0
BOUNDARY_HEAD = 1.0
K_DAY = 0.8
K = K_DAY / 24.0
SS = 0.15
D = K / SS

BASE_SEED = 42
FORECAST_HOURS = [4.0, 5.0, 6.0]
OBS_TIMES = np.array([1.0, 2.0, 3.0], dtype=np.float64)
OBS_HOURLY_BIAS_M = np.array([0.003, 0.007, 0.010], dtype=np.float64)
SENSOR_POINTS = np.array(
    [
        [6.0, 8.0],
        [12.0, 14.0],
        [19.0, 9.0],
        [27.0, 15.0],
        [35.0, 10.0],
        [43.0, 18.0],
        [8.0, 27.0],
        [16.0, 34.0],
        [24.0, 26.0],
        [32.0, 37.0],
        [40.0, 29.0],
        [22.0, 43.0],
    ],
    dtype=np.float64,
)

RAIN_FIELD_CONFIG = {
    "seed": BASE_SEED + 8,
    "num_cells": 18,
    "x_margin_m": 4.0,
    "y_margin_m": 4.0,
    "sigma_range_m": (1.5, 3.8),
    "intensity_range_m_per_hour": (0.0015, 0.0040),
}
RAIN_TIME_BREAKPOINTS_HOURS = np.array(
    [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.4, T_END], dtype=np.float64
)
RAIN_TIME_MULTIPLIERS = np.array(
    [0.20, 0.45, 0.78, 1.00, 0.90, 0.72, 0.55, 0.35, 0.15], dtype=np.float64
)


@dataclass
class RunConfig:
    runs: int = 10
    first_seed: int = 1001
    nn_adam: int = 1200
    nn_lbfgs: int = 60
    pinn_adam: int = 1800
    pinn_lbfgs: int = 60
    nn_hidden: int = 128
    nn_layers: int = 4
    pinn_hidden: int = 96
    pinn_layers: int = 5
    n_int: int = 1000
    n_ini: int = 600
    n_bnd: int = 600
    initial_stride: int = 2
    lr_nn: float = 1.0e-3
    lr_pinn: float = 8.0e-4
    w_nn_ini: float = 15.0
    w_nn_data: float = 35.0
    w_pde: float = 1.0
    w_ini: float = 25.0
    w_bnd: float = 25.0
    w_data: float = 30.0
    observation_noise_std_m: float = 0.003


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    x = np.linspace(0.0, Lx, NX)
    y = np.linspace(0.0, Ly, NY)
    dx = Lx / (NX - 1)
    dy = Ly / (NY - 1)
    dt_stable = 0.5 / (D * (1.0 / dx**2 + 1.0 / dy**2))
    dt = min(TARGET_DT_HOURS, 0.90 * dt_stable)
    nt = int(math.ceil(T_END / dt))
    dt = T_END / nt
    t_arr = np.linspace(0.0, T_END, nt + 1)
    return x, y, t_arr, dt, nt


X_1D, Y_1D, T_ARR, DT, NT = build_grid()
DX = Lx / (NX - 1)
DY = Ly / (NY - 1)
X_GRID, Y_GRID = np.meshgrid(X_1D, Y_1D, indexing="ij")

_rain_rng = np.random.default_rng(RAIN_FIELD_CONFIG["seed"])
RAIN_CELL_CENTERS = np.column_stack(
    [
        _rain_rng.uniform(
            RAIN_FIELD_CONFIG["x_margin_m"],
            Lx - RAIN_FIELD_CONFIG["x_margin_m"],
            size=RAIN_FIELD_CONFIG["num_cells"],
        ),
        _rain_rng.uniform(
            RAIN_FIELD_CONFIG["y_margin_m"],
            Ly - RAIN_FIELD_CONFIG["y_margin_m"],
            size=RAIN_FIELD_CONFIG["num_cells"],
        ),
    ]
).astype(np.float64)
RAIN_CELL_SIGMAS = _rain_rng.uniform(
    *RAIN_FIELD_CONFIG["sigma_range_m"], size=RAIN_FIELD_CONFIG["num_cells"]
).astype(np.float64)
RAIN_CELL_INTENSITIES = _rain_rng.uniform(
    *RAIN_FIELD_CONFIG["intensity_range_m_per_hour"],
    size=RAIN_FIELD_CONFIG["num_cells"],
).astype(np.float64)


def nearest_time_index(t_eval: float) -> int:
    return int(np.argmin(np.abs(T_ARR - t_eval)))


def rainfall_time_multiplier_np(t: float | np.ndarray) -> np.ndarray:
    return np.interp(t, RAIN_TIME_BREAKPOINTS_HOURS, RAIN_TIME_MULTIPLIERS)


def rainfall_source_np(x: np.ndarray, y: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    field = np.zeros_like(np.asarray(x, dtype=np.float64), dtype=np.float64)
    for (cx, cy), sigma, intensity in zip(
        RAIN_CELL_CENTERS, RAIN_CELL_SIGMAS, RAIN_CELL_INTENSITIES
    ):
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        field += intensity * np.exp(-r2 / (2.0 * sigma**2))
    return field * rainfall_time_multiplier_np(t)


def rainfall_source_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    centers = torch.tensor(RAIN_CELL_CENTERS, dtype=torch.float32, device=device)
    sigmas = torch.tensor(RAIN_CELL_SIGMAS, dtype=torch.float32, device=device)
    intensities = torch.tensor(RAIN_CELL_INTENSITIES, dtype=torch.float32, device=device)
    field = torch.zeros_like(x)
    for idx in range(centers.shape[0]):
        cx = centers[idx, 0]
        cy = centers[idx, 1]
        sigma = sigmas[idx]
        intensity = intensities[idx]
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        field = field + intensity * torch.exp(-r2 / (2.0 * sigma**2))

    # Differentiable piecewise-linear temporal multiplier.
    multiplier = torch.zeros_like(t)
    for i in range(len(RAIN_TIME_BREAKPOINTS_HOURS) - 1):
        t0 = float(RAIN_TIME_BREAKPOINTS_HOURS[i])
        t1 = float(RAIN_TIME_BREAKPOINTS_HOURS[i + 1])
        m0 = float(RAIN_TIME_MULTIPLIERS[i])
        m1 = float(RAIN_TIME_MULTIPLIERS[i + 1])
        mask = (t >= t0) & (t <= t1)
        frac = (t - t0) / max(t1 - t0, 1.0e-12)
        multiplier = torch.where(mask, m0 + (m1 - m0) * frac, multiplier)
    return field * multiplier


def initial_condition_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(x, dtype=np.float64), INITIAL_HEAD, dtype=np.float64)


def initial_condition_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.full_like(x, INITIAL_HEAD)


def solve_fdm() -> np.ndarray:
    h = np.zeros((NT + 1, NX, NY), dtype=np.float64)
    h[0] = initial_condition_np(X_GRID, Y_GRID)
    h[0, 0, :] = BOUNDARY_HEAD
    h[0, -1, :] = BOUNDARY_HEAD
    h[0, :, 0] = BOUNDARY_HEAD
    h[0, :, -1] = BOUNDARY_HEAD
    rx = D * DT / DX**2
    ry = D * DT / DY**2
    if rx + ry > 0.5:
        raise RuntimeError(f"Unstable FDM setting: rx+ry={rx + ry}")

    for n in range(NT):
        rainfall = rainfall_source_np(X_GRID, Y_GRID, T_ARR[n])
        h[n + 1, 1:-1, 1:-1] = (
            h[n, 1:-1, 1:-1]
            + rx * (h[n, 2:, 1:-1] - 2.0 * h[n, 1:-1, 1:-1] + h[n, :-2, 1:-1])
            + ry * (h[n, 1:-1, 2:] - 2.0 * h[n, 1:-1, 1:-1] + h[n, 1:-1, :-2])
            + DT * rainfall[1:-1, 1:-1] / SS
        )
        h[n + 1, 0, :] = BOUNDARY_HEAD
        h[n + 1, -1, :] = BOUNDARY_HEAD
        h[n + 1, :, 0] = BOUNDARY_HEAD
        h[n + 1, :, -1] = BOUNDARY_HEAD
    return h


def bilinear_sample(field: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for px, py in points_xy:
        xi = px / DX
        yi = py / DY
        i0 = int(np.floor(xi))
        j0 = int(np.floor(yi))
        i1 = min(i0 + 1, NX - 1)
        j1 = min(j0 + 1, NY - 1)
        wx = xi - i0
        wy = yi - j0
        v = (
            (1.0 - wx) * (1.0 - wy) * field[i0, j0]
            + wx * (1.0 - wy) * field[i1, j0]
            + (1.0 - wx) * wy * field[i0, j1]
            + wx * wy * field[i1, j1]
        )
        values.append(float(v))
    return np.array(values, dtype=np.float64)


def build_observations(h_fdm: np.ndarray, seed: int, noise_std: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    values = []
    for t_obs, bias in zip(OBS_TIMES, OBS_HOURLY_BIAS_M):
        idx = nearest_time_index(float(t_obs))
        base = bilinear_sample(h_fdm[idx], SENSOR_POINTS)
        noise = rng.normal(0.0, noise_std, size=base.shape)
        values.append(np.clip(base + bias + noise, BOUNDARY_HEAD, None))
    return {
        "times_h": OBS_TIMES.copy(),
        "sensor_points_xy_m": SENSOR_POINTS.copy(),
        "observed_head_m": np.vstack(values),
    }


class HydraulicHeadMLP(nn.Module):
    def __init__(self, hidden_dim: int, num_hidden: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(num_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1] / Lx
        y = xyt[:, 1:2] / Ly
        t = xyt[:, 2:3] / T_END
        return self.net(torch.cat([x, y, t], dim=1))


def initial_condition_grid_to_torch(
    device: torch.device, stride: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xx, yy = np.meshgrid(X_1D[::stride], Y_1D[::stride], indexing="ij")
    x0 = torch.tensor(xx.reshape(-1, 1), dtype=torch.float32, device=device)
    y0 = torch.tensor(yy.reshape(-1, 1), dtype=torch.float32, device=device)
    t0 = torch.zeros_like(x0)
    h0 = torch.full_like(x0, INITIAL_HEAD)
    return x0, y0, t0, h0


def observation_data_to_torch(
    obs_data: dict[str, np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    xyt_rows: list[list[float]] = []
    h_rows: list[float] = []
    for ti, t_obs in enumerate(obs_data["times_h"]):
        for si, (px, py) in enumerate(obs_data["sensor_points_xy_m"]):
            xyt_rows.append([float(px), float(py), float(t_obs)])
            h_rows.append(float(obs_data["observed_head_m"][ti, si]))
    xyt = torch.tensor(xyt_rows, dtype=torch.float32, device=device)
    h = torch.tensor(h_rows, dtype=torch.float32, device=device).reshape(-1, 1)
    return xyt, h


def sample_interior(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.rand((n, 1), device=device) * Lx,
        torch.rand((n, 1), device=device) * Ly,
        torch.rand((n, 1), device=device) * T_END,
    )


def sample_initial(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.rand((n, 1), device=device) * Lx
    y = torch.rand((n, 1), device=device) * Ly
    t = torch.zeros((n, 1), device=device)
    h = initial_condition_torch(x)
    return x, y, t, h


def sample_boundary(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_side = max(n // 4, 1)
    x1 = torch.zeros((n_side, 1), device=device)
    y1 = torch.rand((n_side, 1), device=device) * Ly
    t1 = torch.rand((n_side, 1), device=device) * T_END
    x2 = torch.full((n_side, 1), Lx, device=device)
    y2 = torch.rand((n_side, 1), device=device) * Ly
    t2 = torch.rand((n_side, 1), device=device) * T_END
    x3 = torch.rand((n_side, 1), device=device) * Lx
    y3 = torch.zeros((n_side, 1), device=device)
    t3 = torch.rand((n_side, 1), device=device) * T_END
    x4 = torch.rand((n_side, 1), device=device) * Lx
    y4 = torch.full((n_side, 1), Ly, device=device)
    t4 = torch.rand((n_side, 1), device=device) * T_END
    x = torch.cat([x1, x2, x3, x4], dim=0)
    y = torch.cat([y1, y2, y3, y4], dim=0)
    t = torch.cat([t1, t2, t3, t4], dim=0)
    h = torch.full_like(x, BOUNDARY_HEAD)
    return x, y, t, h


def pde_residual(model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    h = model(torch.cat([x, y, t], dim=1))
    ones = torch.ones_like(h)
    h_t = torch.autograd.grad(h, t, ones, create_graph=True, retain_graph=True)[0]
    h_x = torch.autograd.grad(h, x, ones, create_graph=True, retain_graph=True)[0]
    h_y = torch.autograd.grad(h, y, ones, create_graph=True, retain_graph=True)[0]
    h_xx = torch.autograd.grad(h_x, x, torch.ones_like(h_x), create_graph=True, retain_graph=True)[0]
    h_yy = torch.autograd.grad(h_y, y, torch.ones_like(h_y), create_graph=True, retain_graph=True)[0]
    rainfall = rainfall_source_torch(x, y, t, device)
    return SS * h_t - K * (h_xx + h_yy) - rainfall


def train_neural_network(
    cfg: RunConfig,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    model = HydraulicHeadMLP(cfg.nn_hidden, cfg.nn_layers).to(device)
    x0, y0, t0, h0 = initial_condition_grid_to_torch(device, cfg.initial_stride)
    obs_xyt, obs_h = observation_data_to_torch(obs_data, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_nn, weight_decay=1.0e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg.nn_adam // 3, 1), gamma=0.5)
    start = time.perf_counter()
    peak_mem = 0.0

    for _ in range(cfg.nn_adam):
        optimizer.zero_grad(set_to_none=True)
        loss_ini = torch.mean((model(torch.cat([x0, y0, t0], dim=1)) - h0) ** 2)
        loss_data = torch.mean((model(obs_xyt) - obs_h) ** 2)
        loss = cfg.w_nn_ini * loss_ini + cfg.w_nn_data * loss_data
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_mem = max(peak_mem, cuda_peak_mb())

    if cfg.nn_lbfgs > 0:
        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.6,
            max_iter=cfg.nn_lbfgs,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss_ini = torch.mean((model(torch.cat([x0, y0, t0], dim=1)) - h0) ** 2)
            loss_data = torch.mean((model(obs_xyt) - obs_h) ** 2)
            loss = cfg.w_nn_ini * loss_ini + cfg.w_nn_data * loss_data
            loss.backward()
            return loss

        lbfgs.step(closure)
        peak_mem = max(peak_mem, cuda_peak_mb())

    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, time.perf_counter() - start, peak_mem


def train_pinn(
    cfg: RunConfig,
    obs_data: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[nn.Module, float, float]:
    model = HydraulicHeadMLP(cfg.pinn_hidden, cfg.pinn_layers).to(device)
    obs_xyt, obs_h = observation_data_to_torch(obs_data, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_pinn)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    start = time.perf_counter()
    peak_mem = 0.0

    for _ in range(cfg.pinn_adam):
        optimizer.zero_grad(set_to_none=True)
        xi, yi, ti = sample_interior(cfg.n_int, device)
        x0, y0, t0, h0 = sample_initial(cfg.n_ini, device)
        xb, yb, tb, hb = sample_boundary(cfg.n_bnd, device)
        loss_pde = torch.mean(pde_residual(model, xi, yi, ti, device) ** 2)
        loss_ini = torch.mean((model(torch.cat([x0, y0, t0], dim=1)) - h0) ** 2)
        loss_bnd = torch.mean((model(torch.cat([xb, yb, tb], dim=1)) - hb) ** 2)
        loss_data = torch.mean((model(obs_xyt) - obs_h) ** 2)
        loss = (
            cfg.w_pde * loss_pde
            + cfg.w_ini * loss_ini
            + cfg.w_bnd * loss_bnd
            + cfg.w_data * loss_data
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        peak_mem = max(peak_mem, cuda_peak_mb())

    if cfg.pinn_lbfgs > 0:
        xi, yi, ti = sample_interior(cfg.n_int, device)
        x0, y0, t0, h0 = sample_initial(cfg.n_ini, device)
        xb, yb, tb, hb = sample_boundary(cfg.n_bnd, device)
        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.6,
            max_iter=cfg.pinn_lbfgs,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss_pde = torch.mean(pde_residual(model, xi, yi, ti, device) ** 2)
            loss_ini = torch.mean((model(torch.cat([x0, y0, t0], dim=1)) - h0) ** 2)
            loss_bnd = torch.mean((model(torch.cat([xb, yb, tb], dim=1)) - hb) ** 2)
            loss_data = torch.mean((model(obs_xyt) - obs_h) ** 2)
            loss = (
                cfg.w_pde * loss_pde
                + cfg.w_ini * loss_ini
                + cfg.w_bnd * loss_bnd
                + cfg.w_data * loss_data
            )
            loss.backward()
            return loss

        lbfgs.step(closure)
        peak_mem = max(peak_mem, cuda_peak_mb())

    if device.type == "cuda":
        torch.cuda.synchronize()
    return model, time.perf_counter() - start, peak_mem


def cuda_peak_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024**2))
    return 0.0


def predict_model(model: nn.Module, t_eval: float, device: torch.device) -> np.ndarray:
    x_t = torch.tensor(X_GRID.reshape(-1, 1), dtype=torch.float32, device=device)
    y_t = torch.tensor(Y_GRID.reshape(-1, 1), dtype=torch.float32, device=device)
    t_t = torch.full_like(x_t, float(t_eval), device=device)
    model.eval()
    with torch.no_grad():
        pred = model(torch.cat([x_t, y_t, t_t], dim=1)).detach().cpu().numpy()
    return pred.reshape(NX, NY)


def regression_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - reference
    return {
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "mae_m": float(np.mean(np.abs(error))),
        "max_abs_error_m": float(np.max(np.abs(error))),
    }


def evaluate_model(
    model: nn.Module,
    h_fdm: np.ndarray,
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    rmses = []
    maes = []
    maxes = []
    for hour in FORECAST_HOURS:
        idx = nearest_time_index(hour)
        pred = predict_model(model, float(T_ARR[idx]), device)
        m = regression_metrics(h_fdm[idx], pred)
        metrics[f"{prefix}_rmse_h{int(hour)}"] = m["rmse_m"]
        metrics[f"{prefix}_mae_h{int(hour)}"] = m["mae_m"]
        metrics[f"{prefix}_max_h{int(hour)}"] = m["max_abs_error_m"]
        rmses.append(m["rmse_m"])
        maes.append(m["mae_m"])
        maxes.append(m["max_abs_error_m"])
    metrics[f"{prefix}_rmse_mean"] = float(np.mean(rmses))
    metrics[f"{prefix}_mae_mean"] = float(np.mean(maes))
    metrics[f"{prefix}_max_mean"] = float(np.mean(maxes))
    return metrics


def paired_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    nn_rmse = np.array([r["nn_rmse_mean"] for r in rows], dtype=np.float64)
    pinn_rmse = np.array([r["pinn_rmse_mean"] for r in rows], dtype=np.float64)
    nn_mae = np.array([r["nn_mae_mean"] for r in rows], dtype=np.float64)
    pinn_mae = np.array([r["pinn_mae_mean"] for r in rows], dtype=np.float64)

    out: dict[str, float] = {}
    for name, a, b in [
        ("rmse", nn_rmse, pinn_rmse),
        ("mae", nn_mae, pinn_mae),
    ]:
        reduction = (a - b) / a * 100.0
        out[f"nn_{name}_mean"] = float(np.mean(a))
        out[f"nn_{name}_std"] = float(np.std(a, ddof=1))
        out[f"pinn_{name}_mean"] = float(np.mean(b))
        out[f"pinn_{name}_std"] = float(np.std(b, ddof=1))
        out[f"{name}_reduction_percent_mean"] = float(np.mean(reduction))
        out[f"{name}_reduction_percent_std"] = float(np.std(reduction, ddof=1))
        t_res = stats.ttest_rel(a, b, alternative="greater")
        out[f"{name}_paired_t_stat"] = float(t_res.statistic)
        out[f"{name}_paired_t_pvalue"] = float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
            out[f"{name}_wilcoxon_stat"] = float(w_res.statistic)
            out[f"{name}_wilcoxon_pvalue"] = float(w_res.pvalue)
        except ValueError:
            out[f"{name}_wilcoxon_stat"] = float("nan")
            out[f"{name}_wilcoxon_pvalue"] = float("nan")
    return out


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, cfg: RunConfig, stats_out: dict[str, float], rows: list[dict[str, float]]) -> None:
    lines = [
        "# Multi-seed validation summary",
        "",
        f"Runs: {cfg.runs}",
        f"Seeds: {', '.join(str(int(r['seed'])) for r in rows)}",
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"FDM grid: {NX} x {NY}, Nt={NT}, dt={DT:.6f} h, rx+ry={D * DT / DX**2 + D * DT / DY**2:.6f}",
        "",
        "## Averaged over forecast hours 4-6",
        "",
        "| Metric | NN mean ± SD | PINN mean ± SD | Mean reduction | paired t-test p | Wilcoxon p |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| RMSE (m) | {stats_out['nn_rmse_mean']:.6f} ± {stats_out['nn_rmse_std']:.6f} | "
            f"{stats_out['pinn_rmse_mean']:.6f} ± {stats_out['pinn_rmse_std']:.6f} | "
            f"{stats_out['rmse_reduction_percent_mean']:.2f}% ± {stats_out['rmse_reduction_percent_std']:.2f}% | "
            f"{stats_out['rmse_paired_t_pvalue']:.4g} | {stats_out['rmse_wilcoxon_pvalue']:.4g} |"
        ),
        (
            f"| MAE (m) | {stats_out['nn_mae_mean']:.6f} ± {stats_out['nn_mae_std']:.6f} | "
            f"{stats_out['pinn_mae_mean']:.6f} ± {stats_out['pinn_mae_std']:.6f} | "
            f"{stats_out['mae_reduction_percent_mean']:.2f}% ± {stats_out['mae_reduction_percent_std']:.2f}% | "
            f"{stats_out['mae_paired_t_pvalue']:.4g} | {stats_out['mae_wilcoxon_pvalue']:.4g} |"
        ),
        "",
        "## Timing",
        "",
        f"Mean NN training time: {np.mean([r['nn_train_seconds'] for r in rows]):.2f} s",
        f"Mean PINN training time: {np.mean([r['pinn_train_seconds'] for r in rows]):.2f} s",
        f"Mean FDM solve time: {np.mean([r['fdm_seconds'] for r in rows]):.4f} s",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_validation(out_dir: Path, rows: list[dict[str, float]]) -> None:
    seeds = [int(r["seed"]) for r in rows]
    nn_rmse = np.array([r["nn_rmse_mean"] for r in rows], dtype=np.float64)
    pinn_rmse = np.array([r["pinn_rmse_mean"] for r in rows], dtype=np.float64)
    nn_mae = np.array([r["nn_mae_mean"] for r in rows], dtype=np.float64)
    pinn_mae = np.array([r["pinn_mae_mean"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    x = np.arange(2)
    means = [nn_rmse.mean(), pinn_rmse.mean()]
    errs = [nn_rmse.std(ddof=1), pinn_rmse.std(ddof=1)]
    ax.bar(x, means, yerr=errs, capsize=5, color=["#4c78a8", "#f58518"])
    ax.set_xticks(x, ["NN", "PINN"])
    ax.set_ylabel("Mean RMSE over 4-6 h (m)")
    ax.set_title("Multi-seed temporal extrapolation error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rmse_mean_sd.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    for i, seed in enumerate(seeds):
        ax.plot([0, 1], [nn_rmse[i], pinn_rmse[i]], marker="o", color="#555555", alpha=0.7)
        ax.text(1.03, pinn_rmse[i], str(seed), va="center", fontsize=7)
    ax.set_xticks([0, 1], ["NN", "PINN"])
    ax.set_ylabel("Mean RMSE over 4-6 h (m)")
    ax.set_title("Paired RMSE comparison by random seed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_rmse_by_seed.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    means = [nn_mae.mean(), pinn_mae.mean()]
    errs = [nn_mae.std(ddof=1), pinn_mae.std(ddof=1)]
    ax.bar(x, means, yerr=errs, capsize=5, color=["#4c78a8", "#f58518"])
    ax.set_xticks(x, ["NN", "PINN"])
    ax.set_ylabel("Mean MAE over 4-6 h (m)")
    ax.set_title("Multi-seed temporal extrapolation error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mae_mean_sd.png")
    plt.close(fig)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1001)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/revision_validation"))
    parser.add_argument("--nn-adam", type=int, default=1200)
    parser.add_argument("--nn-lbfgs", type=int, default=60)
    parser.add_argument("--pinn-adam", type=int, default=1800)
    parser.add_argument("--pinn-lbfgs", type=int, default=60)
    parser.add_argument("--n-int", type=int, default=1000)
    parser.add_argument("--n-ini", type=int, default=600)
    parser.add_argument("--n-bnd", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        runs=args.runs,
        first_seed=args.first_seed,
        nn_adam=args.nn_adam,
        nn_lbfgs=args.nn_lbfgs,
        pinn_adam=args.pinn_adam,
        pinn_lbfgs=args.pinn_lbfgs,
        n_int=args.n_int,
        n_ini=args.n_ini,
        n_bnd=args.n_bnd,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    torch.set_float32_matmul_precision("high")
    process = psutil.Process(os.getpid())

    fdm_start = time.perf_counter()
    h_fdm = solve_fdm()
    fdm_seconds = time.perf_counter() - fdm_start

    rows: list[dict[str, float]] = []
    for run_idx in range(cfg.runs):
        seed = cfg.first_seed + run_idx
        print(f"[run {run_idx + 1}/{cfg.runs}] seed={seed}", flush=True)
        set_seed(seed)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        obs = build_observations(h_fdm, seed, cfg.observation_noise_std_m)

        nn_model, nn_time, nn_peak_mb = train_neural_network(cfg, obs, device)
        nn_metrics = evaluate_model(nn_model, h_fdm, device, "nn")
        del nn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        set_seed(seed)
        pinn_model, pinn_time, pinn_peak_mb = train_pinn(cfg, obs, device)
        pinn_metrics = evaluate_model(pinn_model, h_fdm, device, "pinn")
        del pinn_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row: dict[str, float] = {
            "run": float(run_idx + 1),
            "seed": float(seed),
            "fdm_seconds": fdm_seconds,
            "nn_train_seconds": nn_time,
            "pinn_train_seconds": pinn_time,
            "nn_peak_cuda_mb": nn_peak_mb,
            "pinn_peak_cuda_mb": pinn_peak_mb,
            "rss_mb": float(process.memory_info().rss / (1024**2)),
            **nn_metrics,
            **pinn_metrics,
        }
        row["rmse_reduction_percent"] = (
            (row["nn_rmse_mean"] - row["pinn_rmse_mean"]) / row["nn_rmse_mean"] * 100.0
        )
        row["mae_reduction_percent"] = (
            (row["nn_mae_mean"] - row["pinn_mae_mean"]) / row["nn_mae_mean"] * 100.0
        )
        rows.append(row)
        print(
            f"  NN RMSE={row['nn_rmse_mean']:.6f}, PINN RMSE={row['pinn_rmse_mean']:.6f}, "
            f"reduction={row['rmse_reduction_percent']:.2f}%, "
            f"time NN/PINN={nn_time:.1f}/{pinn_time:.1f}s",
            flush=True,
        )
        write_csv(out_dir / "multiseed_metrics_partial.csv", rows)

    stats_out = paired_stats(rows)
    write_csv(out_dir / "multiseed_metrics.csv", rows)
    (out_dir / "summary_stats.json").write_text(
        json.dumps(stats_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "config": asdict(cfg),
        "domain": {
            "Lx_m": Lx,
            "Ly_m": Ly,
            "Nx": NX,
            "Ny": NY,
            "T_end_h": T_END,
            "dt_h": DT,
            "Nt": NT,
            "K_day_m_per_day": K_DAY,
            "K_m_per_h": K,
            "Ss": SS,
            "D_m2_per_h": D,
            "rx_plus_ry": D * DT / DX**2 + D * DT / DY**2,
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "numpy": np.__version__,
            "scipy": stats.__version__ if hasattr(stats, "__version__") else "see scipy package",
        },
        "files": {
            "script_sha256": sha256(Path(__file__)),
        },
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_summary_markdown(out_dir / "validation_summary.md", cfg, stats_out, rows)
    plot_validation(out_dir, rows)
    print(json.dumps(stats_out, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
