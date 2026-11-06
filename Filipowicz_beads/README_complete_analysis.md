# Complete HMM beads analysis

Open `Filipowicz_beads_complete_analysis_2026.ipynb` and run all cells in order. It works with the repository root or `Filipowicz_beads` as the working directory. Use the Python environment containing the repository's dependencies, including `ndd`.

The notebook analyzes HMM_inst and HMM_noinst only: 141 participants, each with Low and High hazard sessions (282 sessions). It preserves the seven-bead history and response/target alignment used in the reference notebooks, while using the correct binary response field, `Resp01`. It fits the optimal, one-back, and subjective-stay models separately for every participant and hazard condition, with the reference 25-start procedure for the subjective-stay model.

The notebook includes:

- Data validation and participant/session audit.
- Sequence-specific empirical full and one-back IB bounds; theoretical bounds and empirical softmax curves.
- NSB complexity and predictive information, 10,000-resample confidence intervals, information efficiency, and bound classification.
- BIC comparisons, fitted parameters, optimizer diagnostics, and empirical-to-model relative entropy.
- Low-to-High arrows for every participant in the reference orange–blue direction-based style, showing the low-hazard full and one-back bounds; separate instruction-group panels, paired changes and Wilcoxon tests, and explicit filtering rules.
- Example psychometrics grouped by unique seven-bead sequence, with reference opacity weighting and IB-predicted curves; matching participant IB points and bootstrap intervals.
- Efficiency-versus-subjective-hazard curves, exploratory instruction-group contrasts, and session-order sensitivity summaries.

Full-resolution figures, tables, provenance, and intermediate caches are written to `complete_analysis_results/`. Caches include input values, numerical settings, and the shared utility source hash. Pickle caches should be treated as trusted local files. Delete the cache directory to force recomputation. Input data and reference notebooks are not overwritten.

`filipowicz_analysis_helpers.py` accelerates the reference binary bootstrap by retaining count-based NSB entropy estimates in persistent worker processes. The implementation preserves the reference's seeds, trial resampling, and marginal-cardinality behavior, including constant responders. Execution uses a Linux/fork worker pool. Verify parity from the repository root with:

```bash
python -m unittest discover -s Filipowicz_beads -p 'test_analysis_helpers.py'
```

For a quick execution check using one participant per instruction group and reduced numerical resolution:

```bash
FILIPOWICZ_SMOKE=1 jupyter nbconvert --execute --to notebook \
  --output /tmp/Filipowicz_smoke.ipynb \
  Filipowicz_beads/Filipowicz_beads_complete_analysis_2026.ipynb
```

Smoke results are saved separately and must not be used for inference.

Interpretation follows the reference procedures: trial bootstrap intervals omit serial dependence; efficiency intervals condition on complexity and the estimated bound; BIC comparisons use the fitted sample; and subjective-stay fitting retains the reference's unusually small finite-difference step (`eps=1e-19`). The notebook exports convergence diagnostics and flags complexity outside the empirical bound's interpolation range. The hazard contrast is not a learning contrast, and instruction-group comparisons are exploratory.
