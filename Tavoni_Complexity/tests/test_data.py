from pathlib import Path
import numpy as np
import pytest
from mhi.data import load_gaussian, load_bernoulli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load_gaussian(ROOT), load_bernoulli(ROOT)


def test_gaussian_subject_identity_training_and_exclusions(data):
    g, _ = data
    assert len(g) == 202800
    assert g.subject_id.nunique() == 169
    assert (g.groupby(["subject_id", "condition_id"]).size() == 400).all()
    assert g.is_training.sum() == 507*40
    assert not g.loc[g.is_training, "included_in_fit"].any()
    assert not g.loc[g.rt_ms < 100, "included_in_fit"].any()
    assert not g.loc[g.response.isna(), "included_in_fit"].any()
    assert set(g.loc[g.cohort == "varR", "noise_ratio"].unique()) == {.8, 4., .01}


def test_bernoulli_completion_and_truncation(data):
    _, (b, inventory, audit) = data
    assert len(b) == 64242
    assert len(inventory) == 69
    assert inventory.completed_experiment.sum() == 53
    assert set(inventory.loc[inventory.csv_error != "", "subject_id"]) == {"B_056", "B_068"}
    assert b.loc[b.subject_id == "B_068"].shape[0] == 320
    assert not b.loc[~b.complete_block, "included_in_fit"].any()
    assert len(audit) == 400
    suspect = audit[audit.condition_id == "H40P86"]
    assert suspect.empirical_hazard_rate.mean() < .07
    assert suspect.empirical_reliability.mean() > .97


def test_bernoulli_independent_feedback_and_alignment(data):
    _, (b, _, _) = data
    observed = b.target_from_feedback.notna()
    np.testing.assert_array_equal(b.loc[observed, "target_bead"], b.loc[observed, "target_from_feedback"])
    previous = b.groupby(["subject_id", "condition_id"]).target_bead.shift()
    np.testing.assert_array_equal(b.loc[previous.notna(), "stimulus"], previous.dropna())
    assert b.loc[b.trial_in_block == 1, "stimulus"].isna().all()
    assert b[b.subject_id == "B_001"].condition_id.drop_duplicates().tolist() == ["H05P57", "H05P98", "H05P86", "H40P86"]
