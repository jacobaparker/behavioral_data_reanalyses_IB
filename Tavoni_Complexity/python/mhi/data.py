"""Read archives without extracting or modifying the source data."""
import csv
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_gaussian(root, prior_sd=150.0):
    """Return all 400 trials/block, retaining training and invalid responses.

    The generative scale 150 is inferred from the observation residuals and
    change-point draws in the archive; it is configurable, not given in readme.
    """
    frames = []
    with tarfile.open(Path(root) / "data/Gaussian_task.tar.gz") as tar:
        for cohort in ("varR", "varh"):
            source = f"Gaussian_task/Data_{cohort}.dat"
            d = pd.read_csv(tar.extractfile(source))
            d = d.sort_values(["uniqueid", "trial_index"], kind="stable")
            identities = sorted(d.uniqueid.unique())
            d["subject_id"] = d.uniqueid.map({s: f"G_{cohort}_{i:03d}"
                                                for i, s in enumerate(identities, 1)})
            d["source_subject_id"] = d.uniqueid
            d["cohort"] = cohort
            d["task"] = "gaussian_estimation"
            d["source_file"] = source
            d["sequence_id"] = d.Block_Column.astype(int)
            d["hazard_rate"] = d.Hazard.astype(float)
            d["noise_ratio"] = d.Sigma0.astype(float)
            d["condition_id"] = [f"{cohort}_h{h:g}_R{r:g}" for h, r in
                                      zip(d.hazard_rate, d.noise_ratio)]
            d["block_index"] = d.groupby("subject_id").cumcount() // 400 + 1
            d["trial_in_block"] = d.groupby(["subject_id", "block_index"]).cumcount() + 1
            d["is_training"] = d.trial_in_block <= 40
            d["stimulus"] = pd.to_numeric(d.ObservationActual, errors="coerce")
            d["response"] = pd.to_numeric(d.response, errors="coerce")
            d["latent_state"] = pd.to_numeric(d.GenerativeMean, errors="coerce")
            d["prior_mean"] = 2500.0
            d["prior_sd"] = float(prior_sd)
            d["observation_sd"] = d.noise_ratio * prior_sd
            d["rt_ms"] = pd.to_numeric(d.rt, errors="coerce")
            d["valid_response"] = d.response.between(0, 5000) & (d.rt_ms >= 100)
            d["included_in_fit"] = d.valid_response & ~d.is_training
            d["exclusion_reason"] = np.select(
                [~np.isfinite(d.response), ~d.response.between(0, 5000),
                 ~np.isfinite(d.rt_ms), d.rt_ms < 100, d.is_training],
                ["missing_response", "response_out_of_range", "missing_rt",
                 "rt_below_100ms", "training"], default="")
            d["squared_error"] = (d.response - d.latent_state) ** 2
            d["absolute_error"] = abs(d.response - d.latent_state)
            group = d.groupby(["subject_id", "block_index"], sort=False)
            d["is_change_point"] = group.latent_state.diff().ne(0)
            d.loc[d.trial_in_block == 1, "is_change_point"] = False
            d["change_point_observed"] = d.trial_in_block > 1
            keep = ["subject_id", "source_subject_id", "task", "cohort", "condition_id",
                    "block_index", "trial_in_block", "is_training", "stimulus", "response",
                    "latent_state", "hazard_rate", "noise_ratio", "prior_mean", "prior_sd",
                    "observation_sd", "rt_ms", "valid_response", "included_in_fit",
                    "exclusion_reason", "squared_error", "absolute_error", "is_change_point",
                    "change_point_observed", "sequence_id", "source_file", "trial_index",
                    "internal_node_id"]
            frames.append(d[keep])
    return pd.concat(frames, ignore_index=True)


def _read_subject(tar, name):
    """Recover complete CSV records before a truncated final quoted record.

    Strict parsing prevents the incomplete final record being interpreted as data.
    No demographic/survey content is retained in the exported tables.
    """
    reader = csv.DictReader(io.StringIO(tar.extractfile(name).read().decode("utf-8")), strict=True)
    rows, error = [], ""
    try:
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                error = "incomplete_csv_record"
                break
            rows.append(row)
    except csv.Error as exc:
        error = str(exc)
    return rows, error


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def load_bernoulli(root):
    """Return trials, inventory of all subject files, and stimulus-file audit.

    Trial t is a prediction BEFORE sequence bead t is revealed. The displayed
    bead is t-1 (missing for t=1), and the target latent urn is sequence urn t.
    Bead/urn 1 denotes black, 2 white, verified against choice/feedback images.
    """
    trials, inventory, stimulus_audit = [], [], []
    with tarfile.open(Path(root) / "data/Bernoulli_task.tar.gz") as tar:
        sequences = {}
        for member in tar.getmembers():
            if re.search(r"/stimuli/H\d+P\d+/H\d+P\d+\.csv$", member.name):
                seq = pd.read_csv(tar.extractfile(member))
                sequences[member.name] = seq
                cond = member.name.split("/")[-2]
                stimulus_audit.append(dict(
                    condition_id=cond, source_file=member.name, n_trials=len(seq),
                    nominal_hazard_rate=int(cond[1:3])/100,
                    nominal_reliability=int(cond[4:6])/100,
                    empirical_hazard_rate=float(np.mean(np.diff(seq.Urn) != 0)),
                    empirical_reliability=float(np.mean(seq.Urn == seq.Bead))))
        names = sorted((m.name for m in tar.getmembers()
                        if re.search(r"/subjects/Subject\d+\.csv$", m.name)),
                       key=lambda n: int(re.search(r"Subject(\d+)", n)[1]))
        for name in names:
            rows, csv_error = _read_subject(tar, name)
            sid = f"B_{int(re.search(r'Subject(\d+)', name)[1]):03d}"
            order = rows[0]["Block_Order"].split(",")
            script = int(rows[0]["Script_Picker"])
            counts = []
            for b, label in enumerate(order, 1):
                cond = label.removesuffix("Actual")
                h, p = int(cond[1:3])/100, int(cond[4:6])/100
                path = f"Bernoulli_task/stimuli/{cond}/{cond}{script}.csv"
                seq = sequences[path]
                prefix = f"0.0-{4+3*(b-1)}.0-"
                events = {}
                for row in rows:
                    if row["internal_node_id"].startswith(prefix):
                        stage, t = map(int, row["internal_node_id"][len(prefix):].split("."))
                        events[t, stage] = row
                prediction_events = sorted(t for t, stage in events if stage == 0)
                counts.append(len(prediction_events))
                for t in prediction_events:
                    row = events[t, 0]
                    key = _number(row["key_press"])
                    response = {37: 1.0, 39: 0.0}.get(key, np.nan)
                    target = int(seq.Bead.iloc[t] == 1)
                    latent = int(seq.Urn.iloc[t] == 1)
                    rt = _number(row["rt"])
                    feedback = events.get((t, 2))
                    choice = events.get((t, 1))
                    feedback_correct = ("Correct" in feedback["stimulus"]) if feedback else None
                    image_choice = (int("Black" in choice["stimulus"]) if choice else np.nan)
                    # Infer target independently from selected-color and outcome images.
                    target_from_feedback = (int(image_choice if feedback_correct else 1-image_choice)
                                            if feedback and choice else np.nan)
                    stimulus = (int("NewTrialBC" in row["stimulus"]) if t else np.nan)
                    valid = bool(np.isfinite(response) and rt >= 100)
                    complete_block = len(prediction_events) == 300
                    reason = ("missing_or_invalid_response" if not np.isfinite(response) else
                              "missing_rt" if not np.isfinite(rt) else
                              "rt_below_100ms" if rt < 100 else
                              "incomplete_block" if not complete_block else "")
                    trials.append(dict(
                        subject_id=sid, source_subject_id=Path(name).stem, task="bernoulli_prediction",
                        condition_id=cond, block_index=b, trial_in_block=t+1,
                        stimulus=stimulus, response=response, latent_state=latent,
                        target_bead=target, stimulus_urn=(int(seq.Urn.iloc[t-1] == 1) if t else np.nan),
                        target_urn_raw=int(seq.Urn.iloc[t]), target_bead_raw=int(seq.Bead.iloc[t]),
                        hazard_rate=h, reliability=p, noise_ratio=2*(1-p),
                        nominal_statistics_suspect=cond == "H40P86",
                        target_black_probability=p if latent else 1-p,
                        rt_ms=rt, key_press=key, stimulus_image=row["stimulus"],
                        valid_response=valid, complete_block=complete_block,
                        included_in_fit=valid and complete_block,
                        exclusion_reason=reason, is_training=False,
                        correct=(bool(response == target) if np.isfinite(response) else None),
                        feedback_correct=feedback_correct, choice_image_response=image_choice,
                        target_from_feedback=target_from_feedback,
                        is_change_point=(bool(seq.Urn.iloc[t] != seq.Urn.iloc[t-1]) if t else False),
                        change_point_observed=t > 0, sequence_id=script, source_file=name,
                        stimulus_source_file=path, trial_index=int(row["trial_index"]),
                        internal_node_id=row["internal_node_id"], source_csv_truncated=bool(csv_error)))
            inventory.append(dict(subject_id=sid, source_file=name, sequence_id=script,
                                  n_trials=sum(counts), n_complete_blocks=sum(n == 300 for n in counts),
                                  completed_experiment=counts == [300]*4, csv_error=csv_error,
                                  n_parsed_event_records=len(rows)))
    d = pd.DataFrame(trials)
    inv = pd.DataFrame(inventory)
    d = d.merge(inv[["subject_id", "completed_experiment"]], on="subject_id", validate="many_to_one")
    return d, inv, pd.DataFrame(stimulus_audit)
