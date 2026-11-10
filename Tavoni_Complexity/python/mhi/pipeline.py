"""Reproducible subject-by-condition exports, fits, plots, and checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from .data import load_gaussian, load_bernoulli
from .models import (fit_gaussian, bernoulli_predictions, goodness_of_fit,
                     gaussian_complexity, GAUSSIAN_MODELS, BERNOULLI_MODELS)


def _gaussian_worker(job):
    sid, cond, x, y, mask, h, R, sd, seed = job
    fits = fit_gaussian((x-2500)/sd, (y-2500)/sd, mask, h, R, seed)
    rows, predictions = [], {}
    for model, fit in fits.items():
        prediction = fit.pop("prediction")*sd+2500
        predictions[model] = prediction
        params = fit.pop("parameters")
        row = dict(subject_id=sid, condition_id=cond, model=model,
                   parameters_json=json.dumps(params), **fit,
                   window_exceeds_block=bool("window" in model and max(params, default=0) > len(x)),
                   parameter_at_search_boundary=bool(
                       ("window" in model and any(p <= 1+1e-6 or p >= 1e6-1 for p in params)) or
                       ("window" not in model and any(p <= 1e-6 or p >= 1-1e-6 for p in params))),
                   algorithmic_complexity=gaussian_complexity(model, params),
                   **goodness_of_fit(y[mask], prediction[mask], len(params)+1, 500))
        if "window" in model:
            for i, p in enumerate(params, 1):
                row[f"window_{i}"] = p
                row[f"learning_rate_{i}"] = 1/(R*R+p)
        else:
            for i, p in enumerate(params, 1):
                row[f"learning_rate_{i}"] = p
        rows.append(row)
    return sid, cond, rows, predictions


def _model_weights(fits):
    fits = fits.copy()
    group = fits.groupby(["subject_id", "condition_id"])
    fits["delta_bic"] = fits.bic-group.bic.transform("min")
    weights = np.exp(-fits.delta_bic/2)
    fits["bic_weight"] = weights/weights.groupby([fits.subject_id, fits.condition_id]).transform("sum")
    return fits


def _base_summaries(d, experiment):
    rows = []
    for (sid, cond), g in d.groupby(["subject_id", "condition_id"], sort=True):
        use = g[g.included_in_fit]
        # Descriptive performance is also available for partial blocks.
        valid = g[g.valid_response & ~g.is_training]
        first = g.iloc[0]
        row = {key: first[key] for key in ["subject_id", "source_subject_id", "task", "condition_id",
                                           "block_index", "sequence_id", "source_file", "hazard_rate", "noise_ratio"]}
        row.update(n_trials=len(g), n_fit_trials=len(use), n_valid_responses=len(valid),
                   n_training_trials=int(g.is_training.sum()),
                   n_excluded_test_trials=int((~g.is_training & ~g.valid_response).sum()),
                   median_rt_ms=float(valid.rt_ms.median()) if len(valid) else np.nan,
                   response_mean=float(valid.response.mean()) if len(valid) else np.nan,
                   response_sd=float(valid.response.std()) if len(valid) else np.nan,
                   empirical_hazard_rate=float(g.is_change_point.sum()/max(1, g.change_point_observed.sum())),
                   fit_status="fitted" if len(use) >= 10 else "incomplete_or_insufficient_data")
        if experiment == "gaussian":
            row.update(cohort=first.cohort, prior_mean=first.prior_mean, prior_sd=first.prior_sd,
                       observation_sd=first.observation_sd, complete_block=len(g) == 400,
                       completed_experiment=True,
                       subject_mse=float(valid.squared_error.mean()),
                       subject_rmse=float(np.sqrt(valid.squared_error.mean())),
                       subject_mae=float(valid.absolute_error.mean()),
                       subject_normalized_mse=float(valid.squared_error.mean()/first.prior_sd**2),
                       empirical_observation_sd=float(np.std(g.stimulus-g.latent_state)),
                       empirical_latent_mean=float(g.latent_state.mean()),
                       empirical_latent_sd=float(g.latent_state.std()))
            benchmark = float(np.mean((valid.prediction__bayesian-valid.latent_state)**2))
            row["bayesian_task_mse"] = benchmark
            row["subject_relative_inaccuracy"] = (row["subject_mse"]-benchmark)/benchmark if benchmark > 0 else np.nan
        else:
            row.update(reliability=first.reliability, nominal_statistics_suspect=bool(first.nominal_statistics_suspect),
                       complete_block=bool(first.complete_block),
                       completed_experiment=bool(first.completed_experiment),
                       source_csv_truncated=bool(first.source_csv_truncated),
                       accuracy=float(valid.correct.astype(float).mean()) if len(valid) else np.nan,
                       empirical_reliability=float(np.mean(g.target_urn_raw == g.target_bead_raw)),
                       stimulus_source_file=first.stimulus_source_file)
            row["subject_mse"] = float(np.mean((valid.response-valid.target_bead)**2)) if len(valid) else np.nan
            benchmark = float(np.mean((valid.prediction__bayesian-valid.target_bead)**2)) if len(valid) else np.nan
            row["bayesian_task_mse"] = benchmark
            row["subject_relative_inaccuracy"] = (row["subject_mse"]-benchmark)/benchmark if benchmark > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_fits(summary, fits, criterion):
    metrics = ["sse", "mse", "rmse", "mae", "r2", "log_likelihood", "aic", "bic", "delta_bic",
               "bic_weight", "decision_noise_sd", "n_parameters", "parameters_json",
               "optimizer_success", "algorithmic_complexity", "decision_noise_at_upper_bound",
               "window_exceeds_block", "parameter_at_search_boundary"]
    available = [m for m in metrics if m in fits]
    wide = fits.pivot(index=["subject_id", "condition_id"], columns="model", values=available)
    wide.columns = [f"{model}__{metric}" for metric, model in wide.columns]
    out = summary.merge(wide.reset_index(), on=["subject_id", "condition_id"], how="left", validate="one_to_one")
    best = []
    for (sid, cond), g in fits.groupby(["subject_id", "condition_id"]):
        # Alphabetical model order makes numerical ties deterministic and visible.
        g = g.sort_values([criterion, "model"], kind="stable")
        first = g.iloc[0]
        bic_order = g.sort_values("bic")
        best.append(dict(subject_id=sid, condition_id=cond, best_model=first.model,
                         selection_criterion=criterion, best_model_score=first[criterion],
                         best_model_rmse=first.rmse, best_model_r2=first.r2,
                         best_model_bic_weight=first.bic_weight,
                         best_model_algorithmic_complexity=first.get("algorithmic_complexity", np.nan),
                         best_model_parameters_json=first.parameters_json,
                         best_model_bic=float(first.bic), best_model_sse=float(first.sse),
                         best_model_by_bic=bic_order.iloc[0].model,
                         bic_gap=float(bic_order.bic.iloc[1]-bic_order.bic.iloc[0]),
                         selection_tied_models=";".join(g.loc[np.isclose(g[criterion], first[criterion], rtol=0, atol=1e-8), "model"])))
    return out.merge(pd.DataFrame(best), on=["subject_id", "condition_id"], how="left", validate="one_to_one")


def fit_all_gaussian(d, seed, workers):
    jobs, indices = [], {}
    for (sid, cond), g in d.groupby(["subject_id", "condition_id"], sort=True):
        g = g.sort_values("trial_in_block")
        first = g.iloc[0]
        indices[sid, cond] = g.index.to_numpy()
        jobs.append((sid, cond, g.stimulus.to_numpy(float), g.response.to_numpy(float),
                     g.included_in_fit.to_numpy(bool), float(first.hazard_rate), float(first.noise_ratio),
                     float(first.prior_sd), seed+len(jobs)))
    rows = []
    start = time.monotonic()
    for model in GAUSSIAN_MODELS:
        d[f"prediction__{model}"] = np.nan
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = executor.map(_gaussian_worker, jobs) if executor else map(_gaussian_worker, jobs)
        for i, (sid, cond, result, predictions) in enumerate(results, 1):
            rows.extend(result)
            for model, pred in predictions.items():
                d.loc[indices[sid, cond], f"prediction__{model}"] = pred
            if i % 25 == 0 or i == len(jobs):
                print(f"Gaussian fits: {i}/{len(jobs)} subject-conditions ({time.monotonic()-start:.1f}s)", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return _model_weights(pd.DataFrame(rows))


def fit_all_bernoulli(d, audit):
    rows, sensitivity = [], []
    stats_by_path = audit.set_index("source_file")
    for model in BERNOULLI_MODELS:
        d[f"prediction__{model}"] = np.nan
    for (sid, cond), g in d.groupby(["subject_id", "condition_id"], sort=True):
        g = g.sort_values("trial_in_block")
        f = g.iloc[0]
        predictions = bernoulli_predictions(g.target_bead.to_numpy(), f.hazard_rate, f.reliability)
        mask = g.included_in_fit.to_numpy(bool)
        for model, pred in predictions.items():
            d.loc[g.index, f"prediction__{model}"] = pred
            if mask.sum() < 10:
                continue
            rows.append(dict(subject_id=sid, condition_id=cond, model=model, parameters_json="[]",
                             optimizer_success=True, optimizer_method="fixed_environment_parameters",
                             **goodness_of_fit(g.response.to_numpy(float)[mask], pred[mask], 1, np.inf)))
        if f.nominal_statistics_suspect and mask.sum() >= 10:
            # Sensitivity uses all 500 rows of the supplied generative sequence,
            # not subjects' choices; the nominal fit remains the primary output.
            stat = stats_by_path.loc[f.stimulus_source_file]
            h, p = float(stat.empirical_hazard_rate), float(stat.empirical_reliability)
            for model, pred in bernoulli_predictions(g.target_bead.to_numpy(), h, p).items():
                sensitivity.append(dict(subject_id=sid, condition_id=cond, model=model,
                                        sensitivity_hazard_rate=h, sensitivity_reliability=p,
                                        **goodness_of_fit(g.response.to_numpy(float)[mask], pred[mask], 1, np.inf)))
                d.loc[g.index, f"sensitivity_prediction__{model}"] = pred
    return _model_weights(pd.DataFrame(rows)), pd.DataFrame(sensitivity)


def validate(gaussian, bernoulli, inventory, audit, gf, bf):
    checks = []
    def check(name, passed, details=""):
        checks.append(dict(check=name, passed=bool(passed), details=str(details)))
    keys = ["subject_id", "condition_id", "trial_in_block"]
    for name, d in [("gaussian", gaussian), ("bernoulli", bernoulli)]:
        check(f"{name}_unique_trial_keys", not d.duplicated(keys).any())
        check(f"{name}_finite_stimuli_and_latents", np.isfinite(d.latent_state).all() and
              np.isfinite(d.stimulus[d.trial_in_block > 1]).all())
        check(f"{name}_chronological_blocks", d.groupby(["subject_id", "condition_id"]).trial_in_block.apply(
            lambda s: np.array_equal(s.to_numpy(), np.arange(1, len(s)+1))).all())
    counts = gaussian.groupby(["subject_id", "condition_id"]).size()
    check("gaussian_169_subjects_507_blocks", gaussian.subject_id.nunique() == 169 and len(counts) == 507)
    check("gaussian_400_trials_40_training", (counts == 400).all() and gaussian.is_training.sum() == 40*507)
    check("gaussian_three_conditions_per_subject", (gaussian.groupby("subject_id").condition_id.nunique() == 3).all())
    check("bernoulli_53_complete_subjects", inventory.completed_experiment.sum() == 53,
          f"{inventory.completed_experiment.sum()} complete; {len(inventory)} subject files")
    full = bernoulli[bernoulli.completed_experiment]
    check("bernoulli_complete_subjects_4x300", len(full) == 53*4*300 and
          (full.groupby(["subject_id", "condition_id"]).size() == 300).all())
    feedback = bernoulli.target_from_feedback.notna()
    check("bernoulli_feedback_matches_sequence", (bernoulli.loc[feedback, "target_from_feedback"] ==
                                                 bernoulli.loc[feedback, "target_bead"]).all(),
          f"{feedback.sum()} independently reconstructed targets")
    choice = bernoulli.choice_image_response.notna() & bernoulli.response.notna()
    check("bernoulli_keys_match_choice_images", (bernoulli.loc[choice, "choice_image_response"] ==
                                                 bernoulli.loc[choice, "response"]).all())
    prev = bernoulli.groupby(["subject_id", "condition_id"]).target_bead.shift()
    check("bernoulli_display_is_previous_bead", (bernoulli.loc[prev.notna(), "stimulus"] == prev.dropna()).all())
    for name, fits, trials, models in [("gaussian", gf, gaussian, GAUSSIAN_MODELS),
                                      ("bernoulli", bf, bernoulli, BERNOULLI_MODELS)]:
        check(f"{name}_all_model_fits_present", (fits.groupby(["subject_id", "condition_id"]).size() == len(models)).all())
        check(f"{name}_finite_fit_metrics", np.isfinite(fits[["sse", "bic", "log_likelihood"]].to_numpy()).all())
        check(f"{name}_bic_weights_normalized", np.allclose(fits.groupby(["subject_id", "condition_id"]).bic_weight.sum(), 1))
        check(f"{name}_optimizer_success", fits.optimizer_success.all(), f"{int((~fits.optimizer_success).sum())} unsuccessful terminations")
        errors = []
        for (sid, cond), g in trials.groupby(["subject_id", "condition_id"]):
            use = g[g.included_in_fit]
            if len(use) < 10:
                continue
            ff = fits[(fits.subject_id == sid) & (fits.condition_id == cond)].set_index("model")
            for model in models:
                sse = np.sum((use.response-use[f"prediction__{model}"])**2)
                errors.append(abs(sse-ff.loc[model, "sse"])/max(1, sse))
        check(f"{name}_sse_recomputed_from_trials", max(errors) < 1e-10)
    pivot = gf.pivot(index=["subject_id", "condition_id"], columns="model", values="sse")
    check("gaussian_nested_delta_mixture", (pivot.mixture_delta_rules <= pivot.delta_rule+1e-5).all())
    check("gaussian_nested_window_mixture", (pivot.mixture_sliding_windows <= pivot.sliding_window+1e-5).all())
    warnings = [
        "H40P86 archive has realized h≈0.05,p≈0.98, inconsistent with nominal h=0.40,p=0.86; nominal fits flagged, empirical-statistics sensitivity exported.",
        "Gaussian prior SD=150 is inferred from raw generative residuals; change via --gaussian-prior-sd for sensitivity analysis.",
        "Fits and model-selection metrics are in-sample; BIC weights are descriptive within-subject weights, not group variational Bayes probabilities.",
        "The original subject-fitting code/outputs are absent. These are new fits using the documented equations, not a claim of bitwise reproduction.",
        "Partial Bernoulli blocks retained but not fitted; complete blocks from partial sessions have completed_experiment=False.",
        "Windows exceeding the 400-trial block are weakly identified; window_exceeds_block and parameter_at_search_boundary flags accompany fits.",
    ]
    return {"checks": checks, "all_checks_passed": all(c["passed"] for c in checks), "warnings": warnings}


def plots(out, gaussian, bernoulli, gs, bs, seed):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mhi-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    selected = {}
    for name, trials, summary in [("gaussian", gaussian, gs), ("bernoulli", bernoulli, bs)]:
        subjects = sorted(summary.loc[summary.completed_experiment, "subject_id"].unique())
        picked = rng.choice(subjects, size=3, replace=False).tolist()
        selected[name] = picked
        ncond = 3 if name == "gaussian" else 4
        fig, axes = plt.subplots(3, ncond, figsize=(5*ncond, 8), squeeze=False, constrained_layout=True)
        for i, sid in enumerate(picked):
            for j, (_, s) in enumerate(summary[summary.subject_id == sid].sort_values("block_index").iterrows()):
                ax = axes[i, j]
                g = trials[(trials.subject_id == sid) & (trials.condition_id == s.condition_id)]
                pred = g[f"prediction__{s.best_model}"]
                if name == "gaussian":
                    use = g[g.included_in_fit].iloc[:120]
                    t = use.trial_in_block
                    ax.scatter(t, use.stimulus, s=6, c="0.7", alpha=.7, label="Stimulus")
                    ax.plot(t, use.latent_state, color="black", lw=1, label="Latent mean")
                    ax.plot(t, use.response, color="#3575ad", lw=.8, alpha=.7, label="Response")
                    ax.plot(t, pred.loc[use.index], color="#d34d37", lw=1.5, label="Best model")
                    ax.set_ylabel("Card/deck number")
                else:
                    valid = g.included_in_fit
                    smooth_y = g.response.where(valid).rolling(20, min_periods=10).mean()
                    smooth_p = pred.where(valid).rolling(20, min_periods=10).mean()
                    ax.plot(g.trial_in_block, smooth_y, color="#3575ad", label="Response (20-trial mean)")
                    ax.plot(g.trial_in_block, smooth_p, color="#d34d37", label="Model (20-trial mean)")
                    ax.plot(g.trial_in_block, g.target_black_probability, color="0.5", lw=.5, alpha=.5,
                            label="Nominal target P(black)")
                    ax.set_ylim(-.05, 1.05)
                    ax.set_ylabel("Black choice / expected bead")
                flag = " [statistics mismatch]" if s.get("nominal_statistics_suspect", False) else ""
                ax.set_title(f"{sid} · {s.condition_id}{flag}\n{s.best_model}", fontsize=9)
                ax.set_xlabel("Trial in block")
        axes[0, 0].legend(fontsize=7, loc="best")
        fig.savefig(out/f"{name}_example_subjects.png", dpi=160)
        plt.close(fig)
    # Behavior is shown for individual subjects; this aggregate plot checks scale.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, s, metric, ylabel in [(axes[0], gs, "subject_rmse", "Subject RMSE"),
                                   (axes[1], bs[bs.completed_experiment], "accuracy", "Subject accuracy")]:
        groups = list(s.groupby("condition_id", sort=True))
        for j, (condition, g) in enumerate(groups):
            ax.scatter(j+rng.uniform(-.12, .12, len(g)), g[metric], s=12, alpha=.5)
        ax.set_xticks(range(len(groups)), [k for k, _ in groups], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
    fig.savefig(out/"behavior_by_condition.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), constrained_layout=True)
    for ax, sid in zip(axes, selected["bernoulli"]):
        s = bs[(bs.subject_id == sid) & (bs.condition_id == "H40P86")].iloc[0]
        g = bernoulli[(bernoulli.subject_id == sid) & (bernoulli.condition_id == "H40P86")]
        valid = g.included_in_fit
        for series, color, label in [
            (g.response, "#3575ad", "Response"),
            (g[f"prediction__{s.best_model}"], "#d34d37", "Best nominal model"),
            (g[f"sensitivity_prediction__{s.sensitivity_best_model}"], "#289165", "Best empirical-statistics model")]:
            ax.plot(g.trial_in_block, series.where(valid).rolling(20, min_periods=10).mean(),
                    color=color, label=label)
        ax.set_title(f"{sid} · H40P86", fontsize=10)
        ax.set_xlabel("Trial in block")
        ax.set_ylabel("20-trial mean black choice / prediction")
        ax.set_ylim(-.05, 1.05)
    axes[0].legend(fontsize=7)
    fig.savefig(out/"bernoulli_statistics_sensitivity.png", dpi=160)
    plt.close(fig)
    return selected


def _save(df, out, name):
    df.to_parquet(out/f"{name}.parquet", index=False)
    df.to_csv(out/f"{name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--gaussian-prior-sd", type=float, default=150)
    args = parser.parse_args(argv)
    out = args.output or args.root/"outputs"
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    gaussian = load_gaussian(args.root, args.gaussian_prior_sd)
    bernoulli, inventory, audit = load_bernoulli(args.root)
    print(f"Loaded {len(gaussian):,} Gaussian and {len(bernoulli):,} Bernoulli trials; "
          f"{inventory.completed_experiment.sum()} complete Bernoulli subjects.", flush=True)
    gf = fit_all_gaussian(gaussian, args.seed, args.workers)
    bf, sensitivity = fit_all_bernoulli(bernoulli, audit)
    gs = _attach_fits(_base_summaries(gaussian, "gaussian"), gf, "bic")
    bs = _attach_fits(_base_summaries(bernoulli, "bernoulli"), bf, "sse")
    for summary, trials in [(gs, gaussian), (bs, bernoulli)]:
        mapping = summary.set_index(["subject_id", "condition_id"]).best_model
        trials["best_model"] = [mapping.loc[(s, c)] for s, c in zip(trials.subject_id, trials.condition_id)]
        trials["best_model_prediction"] = [trials.loc[i, f"prediction__{m}"] if isinstance(m, str) else np.nan
                                             for i, m in zip(trials.index, trials.best_model)]
    sensitivity = _model_weights(sensitivity)
    alt_best = sensitivity.loc[sensitivity.groupby(["subject_id", "condition_id"]).sse.idxmin()]
    bs = bs.merge(alt_best[["subject_id", "condition_id", "model", "sensitivity_hazard_rate", "sensitivity_reliability"]]
                  .rename(columns={"model": "sensitivity_best_model"}), on=["subject_id", "condition_id"],
                  how="left", validate="one_to_one")
    report = validate(gaussian, bernoulli, inventory, audit, gf, bf)
    report["random_example_subjects"] = plots(out, gaussian, bernoulli, gs, bs, args.seed)
    tables = dict(gaussian_subject_conditions=gs, bernoulli_subject_conditions=bs,
                  gaussian_trials=gaussian, bernoulli_trials=bernoulli,
                  gaussian_model_fits=gf, bernoulli_model_fits=bf,
                  bernoulli_subject_inventory=inventory, bernoulli_stimulus_audit=audit,
                  bernoulli_statistics_sensitivity=sensitivity)
    for name, df in tables.items():
        _save(df, out, name)
    schema = {name: {column: dict(dtype=str(df[column].dtype), missing=int(df[column].isna().sum()))
                     for column in df.columns} for name, df in tables.items()}
    (out/"dataframe_schema.json").write_text(json.dumps(schema, indent=2)+"\n")
    report["table_shapes"] = {name: list(df.shape) for name, df in tables.items()}
    report["best_model_counts"] = {name: df.best_model.value_counts().to_dict()
                                    for name, df in [("gaussian", gs), ("bernoulli_complete_subjects", bs[bs.completed_experiment])]}
    report["elapsed_seconds"] = time.monotonic()-start
    (out/"sanity_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    provenance = dict(seed=args.seed, gaussian_prior_sd=args.gaussian_prior_sd,
                      python=sys.version, packages={p: importlib.metadata.version(p) for p in
                            ["numpy", "scipy", "pandas", "numba", "matplotlib", "pyarrow"]},
                      source_sha256={str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [args.root/"data/Gaussian_task.tar.gz", args.root/"data/Bernoulli_task.tar.gz"]},
                      model_code_sha256=hashlib.sha256(Path(__file__).with_name("models.py").read_bytes()).hexdigest(),
                      python_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in sorted(Path(__file__).parent.glob("*.py"))},
                      gaussian_models=list(GAUSSIAN_MODELS), bernoulli_models=list(BERNOULLI_MODELS))
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "table_shapes": report["table_shapes"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    if not report["all_checks_passed"]:
        for check in report["checks"]:
            if not check["passed"]:
                print("FAILED CHECK:", check)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
