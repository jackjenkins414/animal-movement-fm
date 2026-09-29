# GenAI reference: I listed what the rotation evaluator must guarantee and GenAI wrote these tests from that, on synthetic data only.

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_rotation_sealed_eval as R
from experiments.data import bird_duplicate_check as D

T0 = pd.Timestamp("2021-01-01")
PAYLOAD = R.RT.load_registered_folds()


# --------------------------------------------------------------------------- seal

FREEZE_TEXT = (
    "# Rotation\n\n## 5. Readings\n\ntext\n\n## 6. Pre-run predictions\n\nFilled by me.\n\n"
    "## 7. As trained\n\n198 of 198 runs, 493 SU.\n\n## 8. Limitations\n\n- one\n"
)


class _Shown:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


def test_sections_are_found_by_number():
    assert R.section_body(FREEZE_TEXT, 6) == "Filled by me."
    assert R.section_body(FREEZE_TEXT, 7) == "198 of 198 runs, 493 SU."
    with pytest.raises(SystemExit, match="no section 9"):
        R.section_body(FREEZE_TEXT, 9)


@pytest.mark.parametrize("body", ["", "WILL FILL", "to fill after training", "JACK TO WRITE", "Jack  to write",
                                  "**DRAFT** 198 of 198"])
def test_a_placeholder_or_empty_section_is_refused(body):
    with pytest.raises(SystemExit):
        R.check_filled(6, body)


def test_the_seal_reads_both_sections_from_head_and_restores_the_tier2_freeze(monkeypatch, tmp_path):
    seen = {}

    def committed():
        seen["freeze"] = R.S2.FREEZE
        return {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40}

    original = R.S2.FREEZE
    monkeypatch.setattr(R.S2, "assert_freeze_committed", committed)
    monkeypatch.setattr(R.E4, "git", lambda *args: _Shown(FREEZE_TEXT))
    monkeypatch.setattr(R, "MANIFEST", tmp_path / "manifest.json")
    attestation = R.assert_sealed()
    assert seen["freeze"] == R.FREEZE and R.FREEZE.parent.name == "preregistration"
    assert R.S2.FREEZE == original
    assert attestation["sections_filled"] == [6, 7]


@pytest.mark.parametrize("section, old", [(6, "Filled by me."), (7, "198 of 198 runs, 493 SU.")])
def test_the_seal_refuses_a_committed_placeholder_in_6_or_7(monkeypatch, tmp_path, section, old):
    monkeypatch.setattr(R.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(R.E4, "git", lambda *args: _Shown(FREEZE_TEXT.replace(old, "WILL FILL")))
    monkeypatch.setattr(R, "MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(SystemExit, match=f"sec {section} still says"):
        R.assert_sealed()


def test_the_seal_refuses_a_different_blob_than_the_manifest(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attestation": {"freeze_blob": "a" * 40}}))
    monkeypatch.setattr(R.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b" * 40})
    monkeypatch.setattr(R.E4, "git", lambda *args: _Shown(FREEZE_TEXT))
    monkeypatch.setattr(R, "MANIFEST", manifest)
    with pytest.raises(SystemExit, match="--report instead"):
        R.assert_sealed()


# --------------------------------------------------------------------------- scoring rules

def _fixes(stem, individual, partition, n, start=0, species=None, lon0=0.0):
    return pd.DataFrame({
        "stem": stem, "species": species or stem, "individual": individual, "partition": partition,
        "timestamp": [T0 + pd.Timedelta(hours=6 * (start + i)) for i in range(n)],
        "lon": [lon0 + 0.01 * (start + i) for i in range(n)], "lat": 10.0,
    })


def _corpus(*frames):
    return pd.concat(frames, ignore_index=True)


def test_identity_drops_at_100_shared_fixes_with_another_species_and_not_at_99():
    fixes = _corpus(
        _fixes("ardea_alba", "t1", "test", 200),
        _fixes("ardea_alba", "t2", "test", 200, start=1000),
        _fixes("mycteria_americana", "o1", "train", 100),  # 100 exact with t1
        _fixes("megaptera_novaeangliae", "o2", "unassigned", 99, start=1000),  # 99 with t2, unassigned
    )
    drops = R.rule_drops(fixes, ["ardea_alba"])["ardea_alba"]
    assert drops["test_birds"] == 2 and drops["dropped"] == ["t1"] and drops["scored"] == 1
    assert drops["identity"] == [{"test": "t1", "other": "mycteria_americana:o1", "shared": 100}]


def test_identity_counts_one_individual_at_a_time():
    fixes = _corpus(
        _fixes("ardea_alba", "t1", "test", 200),
        _fixes("mycteria_americana", "o1", "train", 60),
        _fixes("mycteria_americana", "o2", "val", 60, start=60),  # 120 in total, never 100 with one individual
    )
    assert R.rule_drops(fixes, ["ardea_alba"])["ardea_alba"]["dropped"] == []


def test_the_stork_extract_is_the_same_species_and_never_triggers_identity():
    fixes = _corpus(
        _fixes("ciconia_ciconia", "t1", "test", 300),
        _fixes("stork", "t1", "test", 300, species="ciconia_ciconia"),
    )
    drops = R.rule_drops(fixes, ["ciconia_ciconia"])["ciconia_ciconia"]
    assert drops["test_birds"] == 1 and drops["dropped"] == []  # the stork file's own test rows are not scored


def test_val_test_drops_at_100_with_one_validation_bird_of_the_same_species():
    fixes = _corpus(
        _fixes("gyps_himalayensis", "t1", "test", 150),
        _fixes("gyps_himalayensis", "t2", "test", 150, start=1000),
        _fixes("gyps_himalayensis", "v1", "val", 100),
        _fixes("gyps_himalayensis", "v2", "val", 99, start=1000),
        _fixes("gyps_himalayensis", "r1", "train", 150, start=1000),  # training copies are not this rule
    )
    drops = R.rule_drops(fixes, ["gyps_himalayensis"])["gyps_himalayensis"]
    assert drops["dropped"] == ["t1"]
    assert drops["val_test"] == [{"test": "t1", "val": "v1", "shared": 100}]


def test_scored_counts_are_checked_against_the_freeze():
    drops = {
        "ciconia_ciconia": {"test_birds": 105, "scored": 100, "dropped": list("abcde")},
        "morus_bassanus": {"test_birds": 10, "scored": 9, "dropped": ["x1"]},
        "larus_fuscus": {"test_birds": 10, "scored": 10, "dropped": []},
    }
    mismatches = R.scored_mismatches(drops)
    assert len(mismatches) == 1 and "morus_bassanus" in mismatches[0] and "x1" in mismatches[0]


def test_eligibility_reports_scored_birds_below_the_floor():
    drops = {"ardea_alba": {"dropped": ["t1"]}}
    floor = R.eligibility(drops, {"ardea_alba": {"t1": 5, "t2": 24, "t3": 400}})["ardea_alba"]
    assert floor == {"scored": 2, "below_floor": ["t2"], "min_windows": 24}


def test_load_fixes_keeps_unassigned_birds_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(D.SPW, "PROCESSED", tmp_path)
    monkeypatch.setattr(D.SPW, "SPLITS", tmp_path)
    (tmp_path / "toy_split_seed42.json").write_text(json.dumps({"train": [1], "val": [], "test": [3]}))
    rows = pd.concat([_fixes("toy", i, "", 4) for i in ("1", "3", "99")], ignore_index=True)
    rows.rename(columns={"individual": "individual_id"})[["individual_id", *D.FIX]].to_parquet(
        tmp_path / "toy_segments.parquet")
    default, _ = D.load_fixes("toy")
    assert sorted(default["individual"].unique()) == ["1", "3"]
    kept, _ = D.load_fixes("toy", keep_unassigned=True)
    assert kept.groupby("individual")["partition"].first().to_dict() == {"1": "train", "3": "test", "99": "unassigned"}


def test_load_fixes_reads_a_species_without_a_manifest_as_unassigned(tmp_path, monkeypatch):
    monkeypatch.setattr(D.SPW, "PROCESSED", tmp_path)
    monkeypatch.setattr(D.SPW, "SPLITS", tmp_path)
    _fixes("whale", "w1", "", 3).rename(columns={"individual": "individual_id"})[["individual_id", *D.FIX]].to_parquet(
        tmp_path / "whale_segments.parquet")
    with pytest.raises(FileNotFoundError):
        D.load_fixes("whale")
    kept, _ = D.load_fixes("whale", keep_unassigned=True)
    assert set(kept["partition"]) == {"unassigned"}


# --------------------------------------------------------------------------- frames and guards

def _prediction(ids, n=30, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for individual in ids:
        for i in range(n):
            rows.append({"individual_id": individual, "timestamp": T0 + pd.Timedelta(hours=6 * i),
                         "target_lon": 0.1 * i, "target_lat": 1.0, "error_m": 0.0})
    frame = pd.DataFrame(rows)
    frame["error_m"] = (rng.gamma(2.0, 500.0, size=len(frame)) + 200.0) * scale
    return frame


def test_drops_leave_p_and_every_arm_aligned():
    persistence = _prediction([101, 102, 103])
    frames = {"P": persistence.copy(), "Bb_f1_s0": _prediction([101, 102, 103], seed=1)}
    kept_p, kept = R.drop_scored(persistence, frames, ["102"])
    assert set(kept_p["individual_id"]) == {101, 103}
    for frame in kept.values():
        assert (frame["individual_id"].to_numpy() == kept_p["individual_id"].to_numpy()).all()


def test_test_frame_refuses_fold_0_and_unknown_birds():
    for bird in (*R.FOLD0, "grus_grus"):
        with pytest.raises(SystemExit, match="not a rotation held-out bird"):
            R.test_frame(bird)


def test_only_the_bird_s_own_fold_and_own_specialists_are_predicted():
    records = [
        {"run_id": "Bb_f5_s0", "arm": "Bb", "fold": 5, "species": None},
        {"run_id": "Bb_f6_s0", "arm": "Bb", "fold": 6, "species": None},
        {"run_id": "LSTMS_gyps_himalayensis_s0", "arm": "LSTMS", "fold": 5, "species": "gyps_himalayensis"},
        {"run_id": "LSTMS_milvus_milvus_s0", "arm": "LSTMS", "fold": 5, "species": "milvus_milvus"},
    ]
    chosen = {r["run_id"] for r in R.own_records("gyps_himalayensis", records)}
    assert chosen == {"Bb_f5_s0", "LSTMS_gyps_himalayensis_s0"}


def test_a_prediction_from_another_fold_s_model_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PREDICTIONS", tmp_path)
    records = {"Bb_f6_s0": {"run_id": "Bb_f6_s0", "arm": "Bb", "fold": 6, "species": None, "seed": 0}}
    monkeypatch.setattr(R, "record_map", lambda records_=None: records)
    frame = _prediction([1, 2])
    frame.to_parquet(tmp_path / "gyps_himalayensis__P.parquet")
    frame.to_parquet(tmp_path / "gyps_himalayensis__Bb_f6_s0.parquet")
    with pytest.raises(SystemExit, match="outside its own fold"):
        R.rotation_frames("gyps_himalayensis")


def test_seeds_up_to_keeps_seeds_0_to_2_and_p():
    records = {f"Bb_s{s}": {"arm": "Bb", "seed": s} for s in range(5)}
    frames = {tag: None for tag in [*records, "P"]}
    kept = R.seeds_up_to(frames, records, R.parse_tag, 2)
    assert set(kept) == {"Bb_s0", "Bb_s1", "Bb_s2", "P"}


# --------------------------------------------------------------------------- statistic

BOUNDARY = 1.0  # every persistence error is above it, so every window is travel


def _trio():
    ids = list(range(1, 7))
    persistence = _prediction(ids, seed=0)
    return persistence, _prediction(ids, seed=1), _prediction(ids, seed=2)


def test_the_metre_interval_is_the_registered_paired_bootstrap_exactly():
    persistence, a, b = _trio()
    ours = R.Resampler(persistence, BOUNDARY).skill_interval("A", a, "B", b)["difference_m"]
    theirs = R.E3.paired_delta(a, b, persistence, BOUNDARY, ("A", "B"))["series"]["travel_p90_m"]["difference_m"]
    assert ours["lo"] == theirs["lo"] and ours["hi"] == theirs["hi"]


def test_skill_is_one_minus_model_over_persistence_on_every_resample():
    persistence, _, _ = _trio()
    better = persistence.assign(error_m=persistence["error_m"] * 0.8)
    boot = R.Resampler(persistence, BOUNDARY).skill_interval("A", better, "P", persistence)
    assert boot["skill"]["A"]["lo"] == pytest.approx(0.2) and boot["skill"]["A"]["hi"] == pytest.approx(0.2)
    assert boot["skill"]["P"] == {"lo": 0.0, "hi": 0.0}
    assert boot["skill_difference"]["lo"] == pytest.approx(0.2)


def test_fewer_than_two_individuals_gives_no_interval():
    one = _prediction([1])
    assert "error" in R.Resampler(one, BOUNDARY).skill_interval("A", one, "P", one)


# --------------------------------------------------------------------------- readings

@pytest.mark.parametrize("gap, label", [(0.05, "within"), (-0.05, "within"), (0.0501, "above"), (-0.0501, "below"),
                                        (float("nan"), "not scored")])
def test_bands_count_0_05_as_within(gap, label):
    assert R.band(gap) == label


def _report(gaps: dict[str, float]):
    return {"species": {bird: {"readings": {"gaps": {family: {"skill_difference": gaps[bird]} for family in R.FAMILIES}}}
                        for bird in R.BIRDS}}


def test_band_counts_over_30_and_27():
    gaps = {bird: (0.1 if bird in R.FOLD0 else -0.1) for bird in R.BIRDS}
    counts = R.band_counts(_report(gaps))
    assert counts["LSTM_30"] == {"above": 3, "within": 0, "below": 27, "not scored": 0}
    assert counts["XGB_27"] == {"above": 0, "within": 0, "below": 27, "not scored": 0}


def test_spearman_of_the_gap_against_training_windows():
    windows = {bird: i + 1 for i, bird in enumerate(R.BIRDS)}
    gaps = {bird: -float(i) for i, bird in enumerate(R.BIRDS)}
    result = R.spearman_windows(_report(gaps), windows)
    assert result["LSTM"] == {"n": 30, "rho": pytest.approx(-1.0)}


def test_the_rbar_join_reads_stork_as_white_stork_and_refuses_a_missing_row():
    table = pd.DataFrame({"species": [R.SPW.TIER4_PROFILE_KEY.get(b, b) for b in R.BIRDS],
                          "rbar_corr": np.linspace(0.1, 0.4, 30)})
    assert "white_stork" in set(table["species"])
    arms = {arm: {"skill": {"travel": {"p90_m": 0.0}}} for arm in R.ARMS}
    report = {"species": {bird: {"arms": {arm: {"skill": {"travel": {"p90_m": i * 0.01}}} for arm in R.ARMS}}
                          for i, bird in enumerate(R.BIRDS)}}
    join = R.rbar_join(report, table)
    assert join["rho"]["LSTMS"] == {"n": 30, "rho": pytest.approx(1.0)}
    with pytest.raises(SystemExit, match="white_stork"):
        R.rbar_join(report, table[table["species"] != "white_stork"])


# --------------------------------------------------------------------------- runs and exclusions

def _clean_records():
    counts, exclusions = PAYLOAD["train_window_counts"], PAYLOAD["training_exclusions"]
    records = {}
    for run in R.RT.PLAN:
        names = [run.species] if run.species else PAYLOAD["pools"][str(run.fold)]
        record = {"run_id": run.run_id, "arm": run.arm,
                  "excluded_individuals": {n: exclusions[n] for n in names if n in exclusions}}
        if R.RT.ARMS[run.arm].get("xgb"):
            record["training_windows"] = R.RT.XGB_POOLED_DRAWS if not run.species else counts[run.species]
        else:
            record["pool_windows"] = sum(counts[n] for n in names)
        records[run.run_id] = record
    return records


def test_exclusions_pass_on_clean_records():
    assert R.assert_training_exclusions(PAYLOAD, _clean_records())["runs"] == 198


def test_exclusions_catch_a_wrong_window_count_or_exclusion_list():
    records = _clean_records()
    records["Bb_f1_s0"]["pool_windows"] += 1
    with pytest.raises(SystemExit, match="Bb_f1_s0: pool_windows"):
        R.assert_training_exclusions(PAYLOAD, records)
    records = _clean_records()
    records["LSTMS_ciconia_ciconia_s0"]["excluded_individuals"] = {}
    with pytest.raises(SystemExit, match="LSTMS_ciconia_ciconia_s0: record names"):
        R.assert_training_exclusions(PAYLOAD, records)


def test_completeness_names_a_missing_run_with_its_fold(monkeypatch, tmp_path):
    monkeypatch.setattr(R.RT, "RESULTS", tmp_path)
    gap = R.completeness({})
    assert not gap["complete"] and gap["n_planned"] == 198
    assert "Bb_f1_s0 (fold 1): no record" in gap["problems"]
