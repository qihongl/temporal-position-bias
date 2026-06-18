"""
Path, I/O, and Noise Utilities
===============================
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F


# ---- Path utilities ----

def param_dir(rho, sigma_m, d_context=64, seed=None, base='logs'):
    """Build nested directory path: {base}/r{rho}_s{sigma}_d{dim}/seed{seed}/"""
    folder = f'r{rho:.2f}_s{sigma_m:.2f}_d{d_context}'
    path = os.path.join(base, folder)
    if seed is not None:
        path = os.path.join(path, f'seed{seed}')
    return path


def log_dir(rho, sigma_m, d_context=64, seed=None, base='logs'):
    """Shorthand: param_dir with base='logs'."""
    return param_dir(rho, sigma_m, d_context, seed, base)


def figure_dir(rho, sigma_m, d_context=64, seed=None, base='figures'):
    """Shorthand: param_dir with base='figures'."""
    return param_dir(rho, sigma_m, d_context, seed, base)


def ensure_dir(path):
    """Create directory (and parents) if not present."""
    os.makedirs(path, exist_ok=True)


# ---- Noise helper ----

def apply_noise(c_q, sigma):
    """Apply Gaussian noise and re-normalize context vector.

    Args:
        c_q: (B, d) context tensor
        sigma: noise standard deviation; zero = no-op

    Returns:
        (B, d) noisy, L2-normalized context
    """
    if sigma > 0:
        c_q = c_q + sigma * torch.randn_like(c_q)
        c_q = F.normalize(c_q, dim=-1)
    return c_q


# ---- Serialization ----

def save_results(save_path, errors_dict, attn_data, rho, sigma_m, metadata=None):
    """Save evaluation errors, attention weights, and metadata as .npz files."""
    ensure_dir(save_path)

    # Errors
    error_path = os.path.join(save_path, 'errors.npz')
    np.savez(error_path,
             p20=errors_dict[0.20], p40=errors_dict[0.40],
             p60=errors_dict[0.60], p80=errors_dict[0.80])

    # Attention weights
    attn_path = os.path.join(save_path, 'attention.npz')
    attn_arrays = {}
    for pf, data in attn_data.items():
        label = f'p{int(pf*100):02d}'
        attn_arrays[label] = data['weights']
        attn_arrays[f'{label}_pred'] = np.array([data['pred']])
        attn_arrays[f'{label}_true_idx'] = np.array([data['true_idx']])
    np.savez(attn_path, **attn_arrays)

    # Metadata
    meta = {'rho': rho, 'sigma_m': sigma_m, **({} if metadata is None else metadata)}
    meta_path = os.path.join(save_path, 'metadata.npz')
    np.savez(meta_path, **{k: np.array([v]) if isinstance(v, (int, float)) else v
                            for k, v in meta.items()})


def load_results(load_path):
    """Load saved evaluation results from a run directory."""
    error_path = os.path.join(load_path, 'errors.npz')
    if not os.path.exists(error_path):
        raise FileNotFoundError(f"No results found at {error_path}")

    data = np.load(error_path)
    errors = {
        0.20: data['p20'], 0.40: data['p40'],
        0.60: data['p60'], 0.80: data['p80']
    }

    # Attention (optional)
    attn_path = os.path.join(load_path, 'attention.npz')
    attn_data = None
    if os.path.exists(attn_path):
        attn_raw = np.load(attn_path)
        attn_data = {}
        for pf in [0.20, 0.40, 0.60, 0.80]:
            label = f'p{int(pf*100):02d}'
            if label in attn_raw:
                attn_data[pf] = {
                    'weights': attn_raw[label],
                    'pred': float(attn_raw[f'{label}_pred'][0]),
                    'true_idx': int(attn_raw[f'{label}_true_idx'][0]),
                }

    # Metadata (optional)
    meta_path = os.path.join(load_path, 'metadata.npz')
    metadata = {}
    if os.path.exists(meta_path):
        meta_raw = np.load(meta_path)
        metadata = {k: v.item() if v.size == 1 else v for k, v in meta_raw.items()}

    return errors, attn_data, metadata


# ---- Run discovery ----

def discover_runs(base='logs'):
    """Find all completed runs by scanning the logs directory."""
    runs = []
    if not os.path.exists(base):
        return runs
    for param_dir in sorted(os.listdir(base)):
        param_path = os.path.join(base, param_dir)
        if not os.path.isdir(param_path):
            continue
        for seed_dir in sorted(os.listdir(param_path)):
            run_path = os.path.join(param_path, seed_dir)
            if os.path.exists(os.path.join(run_path, 'errors.npz')):
                runs.append(run_path)
    return runs


_RUN_PATH_RE = re.compile(r'r([\d.]+)_s([\d.]+)_d(\d+)/seed(\d+)')


def parse_run_path(run_path):
    """Extract rho, sigma_m, d, seed from a run path using regex.

    Matches paths like: logs/r0.95_s0.10_d64/seed42/
    """
    m = _RUN_PATH_RE.search(run_path)
    if not m:
        raise ValueError(f"Cannot parse run path: {run_path}")
    return {
        'rho': float(m.group(1)),
        'sigma_m': float(m.group(2)),
        'd': int(m.group(3)),
        'seed': int(m.group(4)),
    }
