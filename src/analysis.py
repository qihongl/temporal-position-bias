"""
Analysis Utilities
===================
Shared computation functions: summary statistics, human loss, MemToolbox fit.
"""

import numpy as np
from scipy.stats import vonmises
from scipy.optimize import minimize
from .config import TEST_POSITIONS, HUMAN_ERRORS


def compute_summary_stats(errors_dict):
    """Compute means, SEMs, and asymmetry from per-position error dict.

    Args:
        errors_dict: {position_fraction: np.array of signed errors (%)}

    Returns:
        means: list of mean errors at each position
        sems: list of standard errors of the mean
        asymmetry: |early| - |late| in percentage points
    """
    means = [np.mean(errors_dict[p]) for p in TEST_POSITIONS]
    sems = [np.std(errors_dict[p]) / np.sqrt(len(errors_dict[p])) for p in TEST_POSITIONS]
    early_mag = (abs(means[0]) + abs(means[1])) / 2
    late_mag = (abs(means[2]) + abs(means[3])) / 2
    asymmetry = early_mag - late_mag
    return means, sems, asymmetry


def compute_human_loss(means):
    """Compute L1 distance between model means and human benchmark.

    Args:
        means: list of 4 mean signed errors at [20%, 40%, 60%, 80%]

    Returns:
        float: sum of absolute deviations from human data
    """
    return float(np.sum(np.abs(np.array(HUMAN_ERRORS) - np.array(means))))


def fit_memtoolbox_mu(errors_pct):
    """Fit MemToolbox Orientation(WithBias(StandardMixtureModel)).

    Replicates the human analysis pipeline:
      1. Remove non-finite errors
      2. Clip to [-80, 80] (human timeline response bound)
      3. Transform: asind(error / 80)  (map to angular space)
      4. Remove non-finite post-transform
      5. MLE via scipy (MAP with flat priors)

    Returns μ in degrees (positive = overestimation bias).
    """
    errors_pct = errors_pct[np.isfinite(errors_pct)]
    errors_pct = np.clip(errors_pct, -80, 80)
    theta_deg = np.degrees(np.arcsin(errors_pct / 80))
    theta_deg = theta_deg[np.isfinite(theta_deg)]
    theta = np.radians(theta_deg)

    def nll(params):
        mu, log_kappa, logit_g = params
        kappa = np.exp(log_kappa)
        g = 1.0 / (1.0 + np.exp(-logit_g))
        g = np.clip(g, 1e-9, 1 - 1e-9)
        like = (1 - g) * vonmises.pdf(theta, kappa, loc=mu) + g / (2 * np.pi)
        return -np.sum(np.log(np.maximum(like, 1e-300)))

    res = minimize(nll, [0.0, np.log(10), 0.0],
                   bounds=[(-np.pi / 2, np.pi / 2),
                           (np.log(0.1), np.log(1e6)),
                           (-20, 20)],
                   method='L-BFGS-B', options={'maxiter': 1000})
    return np.degrees(res.x[0])


def load_model_mu(log_root, rho, sigma=0.05):
    """Load per-seed MemToolbox μ from saved run directories.

    Returns {position: [μ_seed1, μ_seed2, ...]}.
    """
    from .utils import load_results, discover_runs, parse_run_path
    positions = list(TEST_POSITIONS)
    seed_mu = {pf: [] for pf in positions}
    for run_path in discover_runs(log_root):
        info = parse_run_path(run_path)
        if abs(info['rho'] - rho) > 0.005 or abs(info['sigma_m'] - sigma) > 0.005:
            continue
        errors, _, _ = load_results(run_path)
        for pf in positions:
            seed_mu[pf].append(fit_memtoolbox_mu(errors[pf]))
    return seed_mu


def load_seed_means(log_root, rho, sigma=0.05):
    """Load per-seed mean signed errors and pooled trial data.

    Returns (seed_means, all_errors): both {position: list}.
    """
    from .utils import load_results, discover_runs, parse_run_path
    positions = list(TEST_POSITIONS)
    seed_means = {pf: [] for pf in positions}
    all_errors = {pf: [] for pf in positions}
    for run_path in discover_runs(log_root):
        info = parse_run_path(run_path)
        if abs(info['rho'] - rho) > 0.005 or abs(info['sigma_m'] - sigma) > 0.005:
            continue
        errors, _, _ = load_results(run_path)
        for pf in positions:
            seed_means[pf].append(np.mean(errors[pf]))
            all_errors[pf].extend(errors[pf].tolist())
    return seed_means, all_errors
