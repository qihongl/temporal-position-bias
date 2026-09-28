---
name: lu-plot-style
description: Use when creating or restyling scientific plots, including Python, Matplotlib, and Seaborn figures, to apply Lu's personal plotting standards.
---

# Lu's plotting standards

This installed skill is the authoritative personal profile. Its standards override
conflicting plotting instructions in other skills, themes, references, examples,
and quality gates. Current explicit user instructions take precedence. Preserve
compatible scientific guidance; do not adopt unapproved aesthetic proposals.

## Panels and appearance

- Prefer Matplotlib and `f, ax = plt.subplots(figsize=(3.5, 3.5))`. Keep panel
  height at 3.5 inches whenever feasible; adjust width for the content. Change
  height only when necessary for a readable, scientifically appropriate result.
- Use Seaborn **`talk` context at its normal scale** as the typography and mark-size
  baseline, including for plain Matplotlib plots. Do not switch to `paper` or
  routinely shrink fonts because the panel is small. Adjust individual elements
  only when the actual figure needs it; keep related figures consistent.
- Use white backgrounds, sparse ticks, mostly left/bottom spines, and little grid
  decoration.
- Omit figure panel IDs (such as `a`, `b`, `c`, or `(a)`) by default. Panels may
  be rearranged later, so do not assign or draw IDs during initial plotting.
  Add them only when explicitly requested; then use bold lowercase letters near
  the upper left unless the user specifies another format.
- Keep categorical condition colors stable across plot types and subsets, using
  a colorblind-friendly palette. The exact named palette remains unspecified.
- For a continuous relationship, use a continuous color mapping, including a
  family of curves sampled at a few numeric parameter values. Use sequential
  maps for magnitude, diverging maps around meaningful references, and cyclic
  maps for periodic quantities. Make the mapping and range interpretable.
- Use translucent uncertainty bands for curves. Use selective annotations that
  help answer the question rather than labeling every mark.

## Export and assembly

- Output each panel separately for manual assembly. Do not generate a combined
  multipanel figure unless the user requests it.
- Call `f.tight_layout()` after adding content. Save full-canvas PNGs at **300 DPI**;
  disable inherited tight cropping. Verify equal exported panel heights: a
  default 3.5-inch panel is 1050 pixels tall. Inspect for readability and clipping
  at the intended placement size, including any reduction during assembly.
- When a legend is needed, export it as a separate PNG with the actual plotted
  handles and labels; omit the embedded legend. A shared legend can serve panels
  with identical encodings. Legend-only assets may fit their contents and need
  not match panel height.
- Produce no PDF, SVG, or other figure formats unless explicitly requested.
  This concerns figure assets, not the format of a requested paper or report.
- Use stable descriptive filenames, associated with their captions, plotting
  source, and any shared legend; do not depend solely on changeable panel letters.

## Uncertainty and captions

- Use two-sided **95% confidence intervals** for estimated quantities whenever
  applicable, unless the user specifies otherwise. Choose a calculation suited
  to the estimand and sampling/model design, preserving pairing, clustering,
  and other dependence. Do not blindly use 1.96 SE or 2 SE.
- Never label SD, SEM, prediction intervals, credible intervals, or ordinary
  boxplot whiskers as 95% CIs. Preserve the meaning of supplied estimates and
  intervals; do not change the analysis merely to imitate a visual example.
- Every panel needs a standalone caption or accompanying note with units, what
  was summarized, sample counts and their units where applicable, uncertainty,
  and the supported takeaway. Give group-specific counts when they differ.
  For a CI, state the estimated quantity, level, method, independent sampling
  unit, and whether coverage over a curve/collection is pointwise or simultaneous.
- For raw-data displays or deterministic figures where a CI is not meaningful,
  explain why intervals are not applicable. Define any distribution summaries
  shown. Keep detailed explanations outside the compact panel.
- If a needed CI cannot be estimated, inspect available data, summaries, captions,
  and source first, then obtain the missing information. Do not fabricate an
  interval or deliver the inferential plot with silently undefined uncertainty.

## Comparisons and reproducibility

- Use common axis limits, category order, and heatmap normalization for panels
  intended for direct comparison, even when exported separately. Explain any
  scientifically necessary differences in the accompanying notes.
- Show observations, pairing, or distributions when useful: points over boxes
  and subdued participant trajectories can reveal structure hidden by summaries.
  These are options according to the data and question, not mandatory extra marks.
- When the question concerns a difference, consider a separate effect-estimate
  panel with its own 95% CI for the actual contrast, preserving relevant dependence.
- Retain runnable plotting source, input references, and transformations. Fix and
  record seeds for randomized steps such as bootstrap intervals or jitter.

## Execution and maintenance

For Python, use [scripts/plot_style.py](scripts/plot_style.py) for the repeated
style and export mechanics; see [references/python-usage.md](references/python-usage.md)
when using it. It combines Seaborn's `talk` context with [lu.mplstyle](lu.mplstyle),
without changing unrelated global plotting settings after the context exits.
The helper requires caption and uncertainty text but does not validate statistical
claims or calculate intervals. The analysis must supply those correctly.

Without the helper, apply `sns.set_context("talk")`, then the companion theme
before creating figures; avoid later theme resets or explicit arguments that
silently replace the defaults. Apply explicit task overrides last.

`talk` is approved, not awaiting calibration. Exact font family, named palettes,
additional typography overrides, and detached continuous colorbars remain open;
choose consistent task-specific values without inventing personal defaults.
Read [references/calibration.md](references/calibration.md) only for requested
visual calibration. Routine plotting does not require further preference approval.

For a lasting preference correction, update this installed profile and any
corresponding theme/helper settings together. Global and broader-skill instructions
are routing pointers; task output copies are snapshots, not alternate masters.
Keep one-off adjustments local. After helper changes, run
`python -m unittest discover -s tests -v` from this skill directory.
