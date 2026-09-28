# 100-item temporal-context burn-in control

The active experiment compares 100 burn-in items with the matched zero-initialization baseline. It includes the two separate result figures, their individual panels and legends, saved trial errors, and checkpoints for the 20 paired seeds. Repository `src/` modules are reused without modification.

## Main figures

- `figures_burnin/burnin100_panels_mu.png`: 100-item burn-in.
- `figures_burnin/zero_start_for_burnin100_panels_mu.png`: matched zero-start comparison.

Both conditions use the same axes and the original plotting blocks. Individual panels, detached legends, and caption sidecars are also retained.

## Reproduce from the repository root

Use the Python dependencies recorded in `requirements-control.txt`.

```sh
python experiments/burnin/run_burnin_control.py
python experiments/burnin/analyze_burnin100.py
python experiments/burnin/make_separate_figures.py
```

The runner now trains only zero and 100-item conditions and reuses compatible completed runs. `run_burnin100.py` can rerun just the 100-item condition when the zero baseline already exists. The analyzer independently reads both conditions' saved trial errors; it does not require archived summaries or runs. `--burnin 100` remains accepted by the plotting script.

`results/summary_burnin100.json` contains the fitted-bias contrasts and bootstrap intervals. `results/burnin_asymmetry_significance.json` retains the additional unadjusted two-sided t tests for these two conditions. Fitted μ and raw signed errors use different scales; the fitting transformation remains in `src/analysis.py`.

## Local archive

`archive/burnin200/` is Git-ignored. It preserves the 200-item models, longer-history evaluation checks, superseded comparison figures, mixed summaries, and scripts from before this cleanup. None of those files are needed by the active workflow. The historical scripts are snapshots, not standalone entry points at their archived paths.

The model equations, training inputs, saved primary zero/100 errors, and retained figure images were not changed by this cleanup. The evaluator still draws a fixed-length random prelude internally to preserve the original matched random sequences, but only the requested 100-item suffix is encoded.
