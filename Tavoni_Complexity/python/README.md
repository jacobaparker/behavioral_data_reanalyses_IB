Python data and model analyses
=============================

The generated tables are in `../outputs/`; the quickest entry point is
`../Python_Analysis.ipynb`. No MATLAB installation is needed. These are new
subject fits using the manuscript equations. The repository's `.mat` outputs
contain optimization results for simulated processes, not the subject fits.

Run from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-python.txt
PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m mhi.pipeline --workers 4
.venv/bin/python -m pytest -q
```

`--output /path`, `--seed 20261107`, `--workers 1`, and
`--gaussian-prior-sd 150` are available. The run is deterministic for a given
seed and package versions; `outputs/provenance.json` records versions, source
archive hashes, model-code hash, and settings. Original archives are read in
place without extraction or modification. The generated `.csv.gz` files are
portable alternatives to the typed `.parquet` files.
`requirements-python.lock.txt` records the exact installed package versions.
The notebook requires a Jupyter environment with a kernel containing these
packages; IPython display is supplied by Jupyter. Its seven computation cells
were also exercised in plain Python with display replaced by a file/table check.

Tables and loading
------------------

```python
from pathlib import Path
import pandas as pd

out = Path("outputs")
gaussian_subject_conditions = pd.read_parquet(out / "gaussian_subject_conditions.parquet")
bernoulli_subject_conditions = pd.read_parquet(out / "bernoulli_subject_conditions.parquet")
gaussian_trials = pd.read_parquet(out / "gaussian_trials.parquet")
bernoulli_trials = pd.read_parquet(out / "bernoulli_trials.parquet")

# The 53-subject Bernoulli cohort described in the paper:
b_summary = bernoulli_subject_conditions.query("completed_experiment")
b_trials = bernoulli_trials.query("completed_experiment")

# One subject-condition's predictions and all its fit statistics:
g_fits = pd.read_parquet(out / "gaussian_model_fits.parquet")
example = g_fits.query("subject_id == 'G_varR_001' and condition_id == 'varR_h0.1_R4'")
```

| File stem | Row definition | Rows |
| --- | --- | ---: |
| `gaussian_subject_conditions` | subject × condition | 507 |
| `bernoulli_subject_conditions` | subject × started condition | 217 |
| `gaussian_trials` | response trial, including training/invalid responses | 202,800 |
| `bernoulli_trials` | prediction trial, including partial sessions | 64,242 |
| `gaussian_model_fits` | subject × condition × model | 4,056 |
| `bernoulli_model_fits` | complete block × model | 642 |
| `bernoulli_subject_inventory` | every subject file, even without task trials | 69 |
| `bernoulli_stimulus_audit` | supplied generative sequence | 400 |
| `bernoulli_statistics_sensitivity` | flagged complete block × model using empirical sequence statistics | 159 |

The 53 complete Bernoulli participants contribute 212 summary rows and 63,600
trial rows. Two partial sessions each contain an additional complete block;
these blocks are also fitted, with `completed_experiment=False`. Three partial
blocks are retained without fits. Thirteen files contain no task responses;
they appear in the inventory. Subjects 56 and 68 have truncated final CSV
records; complete preceding records are recovered by strict CSV parsing and the
truncation is recorded. Unstarted conditions are not invented as summary rows.

Keys, coding, and dictionary
---------------------------

* Join summary and trials on `subject_id, condition_id`. Add `trial_in_block`
  for a unique trial key. `block_index` and `trial_in_block` are one-based.
* Gaussian `G_varR_###` and `G_varh_###` identify sorted source sessions, with
  the original session identifier in `source_subject_id`. Bernoulli `B_###`
  corresponds directly to `Subject#.csv`. IDs identify sessions within each
  experiment, and do not establish cross-experiment participant matches.
* `task`: `gaussian_estimation` or `bernoulli_prediction`.
* `hazard_rate`, `noise_ratio`: nominal task parameters. For Gaussian tasks
  `noise_ratio=R=observation_sd/prior_sd`; the confusing raw column `Sigma0`
  actually contains R. `prior_mean=2500`; `prior_sd=150` is inferred from the
  raw generative distributions (residual SD ≈ R×150), and is configurable.
  For Bernoulli, `reliability=p`, `noise_ratio=2*(1-p)`.
* Gaussian `stimulus`, `response`, `latent_state` are the displayed card,
  submitted estimate, and current generative deck mean, in original units.
* Bernoulli `response` is predicted bead color: black=1, white=0. Verified key
  mapping: left-arrow 37=black, right-arrow 39=white. In the source sequences,
  bead/urn code 1=black and 2=white. Original sequence codes are also retained.
* Bernoulli row t is the choice **before bead t is revealed**: `stimulus` is
  bead t−1, and is missing on the first trial (`NoChoice` screen). `target_bead`
  and `latent_state` are bead t and its urn. `stimulus_urn` is the previous urn.
  This convention prevents accidentally treating a prediction as an estimate
  of an already observed bead. `target_black_probability` uses the nominal
  emission reliability conditional on the target urn; it is suspect for H40P86.
* `sequence_id` identifies the supplied stimulus stream, not a participant or
  condition. Gaussian raw `Block_Column` is this sequence ID. Bernoulli
  `Script_Picker` selects the stream and `Block_Order` defines condition order;
  raw `Block_Type` stays fixed across blocks and must not be used as the
  trial's actual condition. Condition identities are reconstructed from order
  and validated against observed sequence/feedback images.
* `is_training`: first 40 of each 400 Gaussian trials; false for Bernoulli task
  trials (tutorial screens are excluded). Models see all stimuli from block
  start, including training and trials with excluded responses. Model state
  resets at block boundaries; invalid response trials do not break the stream.
* `valid_response`: finite response and RT≥100 ms; Gaussian additionally
  requires 0≤response≤5000. `included_in_fit` excludes Gaussian training and
  incomplete Bernoulli blocks. `exclusion_reason` gives a reason. Original
  response/RT values and excluded trials remain available.
* `is_change_point`: observed latent-state change; false at block start, where
  `change_point_observed=False`. Rounded identical Gaussian re-draws cannot be
  distinguished from no change, so empirical hazard is approximate.
* `source_file`, `stimulus_source_file`, `trial_index`, and `internal_node_id`
  trace reconstructed records back to the archives.
* Trial `prediction__MODEL` columns contain every deterministic model output;
  `best_model_prediction` selects the output for the summary's `best_model`.
  Outputs use card units for Gaussian and expected black-bead indicator [0,1]
  for Bernoulli. They do not contain sampled decision noise. Partial Bernoulli
  blocks have model predictions but no selected best model.
* Summary `subject_mse` is performance against the task target, not fit error:
  Gaussian estimate versus latent mean; Bernoulli response versus target bead.
  `subject_rmse`, `subject_mae`, `subject_normalized_mse` are Gaussian measures;
  Bernoulli `accuracy` is proportion correct, and `subject_mse=1-accuracy`.
  `bayesian_task_mse` evaluates Bayesian outputs against the same task targets
  on the same valid trials. `subject_relative_inaccuracy` is subject MSE divided
  by Bayesian task MSE, minus one. Bernoulli uses nominal generative statistics.
* `empirical_hazard_rate` and `empirical_reliability` in summaries describe
  the retained block; stimulus audit statistics use all 500 supplied sequence
  rows. Gaussian empirical SDs use all 400 stimulus/latent pairs, not responses.
* Summary `MODEL__METRIC` columns contain all model goodness-of-fit measures;
  the corresponding long fit table has one model per row.
* Fit `sse`, `mse`, `rmse`, `mae`, and `r2` compare model output to **subject
  responses** on `included_in_fit` trials. R² can be negative; it is missing
  for constant responses. `decision_noise_sd` is the profiled Gaussian
  residual SD. `log_likelihood`, `aic`, and conventional `bic` include decision
  noise among `n_parameters`. Smaller SSE/AIC/BIC is better.
* `paper_log_evidence_score=-BIC` implements the paper's Eq. 40 score;
  `bic_log_evidence_approx=-BIC/2` is the usual log-evidence approximation.
  `delta_bic=BIC-min(BIC)`; `bic_weight` normalizes exp(−delta_bic/2) within a
  subject-condition. These weights do not implement the paper's group-level
  variational Bayesian analysis.
* `best_model` minimizes Gaussian BIC or Bernoulli SSE, following the primary
  comparisons in the paper. `best_model_by_bic` is also reported for both.
  `best_model_bic` is a numeric BIC value. `bic_gap` is the best-to-second-best
  BIC gap; small gaps mean weak separation. `selection_tied_models` records
  exact numerical ties; alphabetical order resolves ties deterministically.
* `parameters_json` lists learning rates for Memoryless/Delta-Rule models or
  window lengths for Sliding-Window models. Mixture rates are descending;
  windows are ascending. Named `learning_rate_1/2` and `window_1/2` columns
  also appear in the Gaussian long fit table.
* `algorithmic_complexity` counts arithmetic/memory operations using unit
  costs and Supplementary Table 1; it excludes the shared reflexive constant.
  Bayesian asymptotic complexity is infinite. This is distinct from parameter
  counts and BIC. Fitted long windows can have large complexity and be weakly
  identified by these short blocks: see `window_exceeds_block` and
  `parameter_at_search_boundary`. Complexity order can depend on window size.
* `optimizer_success`, `optimizer_method`, `nfev`, and
  `decision_noise_at_upper_bound` document fitting behavior.

Models and reconstruction choices
---------------------------------

Gaussian models: Evidence, Prior, Memoryless, Delta Rule, Sliding Window,
Mixture of two Delta Rules, Mixture of two Sliding Windows, exact Bayesian
run-length inference. These are the eight main-text models. The supplementary
Kalman filter and HGF are outside this main hierarchy and are not fitted here.

The Gaussian likelihood is the manuscript's independent Gaussian decision
noise model, with decision SD capped at 500. Learning rates range from 0 to 1.
Sliding-window lengths range from 1 to 1,000,000 and imply learning rates
1/(R²+r). The single fractional window is optimized analytically over all
integer-length segments and the long-window tail. Delta Rule uses a dense
rate grid with bounded local refinement. Mixtures use seeded differential
evolution plus multiple Powell refinements and explicitly include the best
single-unit model as a candidate. Returned optimizer convergence does not
prove a global optimum for the mixtures.

The exact Bayesian filter uses all possible run lengths with log-domain
normalization. Mixture likelihoods use updated unit means, as in the supplied
MATLAB and Eq. 22. For window start-up, unavailable observations before the
block are padded by the prior mean. This explicit choice allows identical fit
trials across all windows; the supplied simulation routines instead discard a
window-dependent burn-in. Models receive the 40 Gaussian training stimuli as
warm-up, but training responses are never fitted. Exact zero learning rates
are handled as frozen prior units.

Bernoulli models use the Bayesian log-odds update and its two documented
approximations. If L is the posterior log-odds of the black urn, its prior
log-odds on the next trial is:

* Bayesian: logit(h+(1−2h)·sigmoid(L)).
* Constant Prior: sign(L)·log((1−h)/h), with sign(0)=0.
* Leaky Accumulator: (1−2h)·L.

On observing a bead, add ±log(p/(1−p)) to those prior log-odds. The predicted
next bead indicator is (1−p)+(2p−1)·sigmoid(prior log-odds). These three models
have no free inference parameters at fixed task statistics. The primary
selection minimizes SSE; the additional Gaussian residual-noise likelihood
uses one fitted decision SD and provides comparable AIC/BIC scores. It is a
Gaussian density score on binary responses, not a Bernoulli choice likelihood
or an estimated choice policy. The source repository supplies no Bernoulli
subject-fit implementation, so response-scale and initialization choices are
explicit reconstructions rather than verified copies of unpublished code.

Data discrepancy requiring attention
-----------------------------------

Every supplied H40P86 stimulus file has realized statistics consistent with
h≈0.05 and p≈0.98, instead of the nominal h=0.40 and p=0.86. The behavioral
stimulus and outcome images agree with those files. This is an archive/log
finding; it does not establish why the discrepancy occurred or whether it
affected the published analyses.

`nominal_statistics_suspect=True` flags these trials/summary rows. Primary
fits preserve nominal values, with no silent correction. The separate
`bernoulli_statistics_sensitivity` table and trial
`sensitivity_prediction__MODEL` columns use h and p estimated from the full
500-trial supplied sequence. The summary contains `sensitivity_best_model`
and the parameters used. This is an exploratory sensitivity analysis with
noisy sequence-statistic estimates, not proof of the intended generator.
For analyses requiring trustworthy nominal task statistics, filter this
condition explicitly; do not interpret it as a validated high-hazard condition.

Verification and sources
------------------------

`outputs/sanity_checks.json` contains data integrity checks, independent
stimulus/feedback alignment, recomputed residual scores, optimizer status,
normalized weights, and model nesting. Tests independently check the MATLAB
probability-domain Bayesian recursion, direct mixture calculations, fractional
window optimization, a two-state Bernoulli HMM, causal stimulus timing, limiting
cases, synthetic learning-rate recovery, and source-data reconstruction.
Example plots select three subjects per experiment reproducibly from seed
20261107; Gaussian panels show 120 valid test trials, and Bernoulli panels show
20-trial rolling response/model means. All reported fit metrics are in-sample;
no cross-validation or group-level model selection is implied. No subjects are
removed using the manuscript's optional 75th-percentile performance cutoff or
the zero-adaptivity exclusion for integration-kernel analyses.
The completed run passed all 26 pipeline checks and all 15 tests. The
`outputs/dataframe_schema.json` file records every exported column's type and
missing-value count. `bernoulli_statistics_sensitivity.png` shows how predictions
change under the empirical-statistics sensitivity analysis for H40P86.

Primary references:

* Local manuscript and `codes/MSE_Estim_*.m`, `codes/AC_*_estim.m`.
* [Tavoni et al. (2022), publisher](https://www.nature.com/articles/s41562-022-01357-z).
* [Supplementary Information, including operation counts](https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs41562-022-01357-z/MediaObjects/41562_2022_1357_MOESM1_ESM.pdf).
* [Glaze et al. (2015), Eqs. 1–3](https://elifesciences.org/articles/08825),
  cited by Tavoni et al. for the Bernoulli approximation models.
