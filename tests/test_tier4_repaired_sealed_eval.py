# GenAI reference: tests for tier4_repaired_sealed_eval.py, written by GenAI from my outline which was (will add)
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.tier4_repaired_sealed_eval as V

R, T, S2, E4, DG = V.R, V.T, V.S2, V.E4, V.DG  # the modules the evaluator imported
GULL, WOLF, NES = V.HELD_OUT


# ---------------------------------------------------------------------------------------------------- seal
def test_the_seal_points_both_checks_at_this_freeze_and_restores_them(monkeypatch):
    saved = S2.FREEZE, E4.FREEZE
    seen = {}
    monkeypatch.setattr(S2, "assert_freeze_committed",
                        lambda: seen.setdefault("s2", S2.FREEZE) and {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(E4, "assert_as_trained_filled", lambda: seen.setdefault("e4", E4.FREEZE) and "line\nline")
    monkeypatch.setattr(V, "MANIFEST", V.SEALED / "no_such_manifest.json")
    attestation = V.assert_sealed()
    assert seen == {"s2": V.FREEZE, "e4": V.FREEZE} and attestation["as_trained_lines"] == 2
    assert (S2.FREEZE, E4.FREEZE) == saved


def test_the_globals_are_restored_when_a_check_fails(monkeypatch):
    saved = S2.FREEZE, E4.FREEZE

    def refuse():
        raise SystemExit("FAIL: not committed")

    monkeypatch.setattr(S2, "assert_freeze_committed", refuse)
    with pytest.raises(SystemExit, match="not committed"):
        V.assert_sealed()
    assert (S2.FREEZE, E4.FREEZE) == saved


def test_predict_never_opens_a_test_partition_when_the_seal_fails(monkeypatch):
    def refuse():
        raise SystemExit("FAIL: sec 7 still says TO FILL")

    monkeypatch.setattr(V, "assert_sealed", refuse)
    monkeypatch.setattr(S2, "test_frame", lambda species: pytest.fail("a test partition was opened"))
    monkeypatch.setattr(V, "checked_runs", lambda: pytest.fail("the seal must come first"))
    with pytest.raises(SystemExit, match="TO FILL"):
        V.run_predictions("cpu")


def test_a_manifest_attested_at_another_blob_is_refused(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attestation": {"freeze_blob": "a" * 40}}))
    monkeypatch.setattr(V, "MANIFEST", manifest)
    monkeypatch.setattr(S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b" * 40})
    monkeypatch.setattr(E4, "assert_as_trained_filled", lambda: "filled")
    with pytest.raises(SystemExit, match="attested at blob"):
        V.assert_sealed()


# ---------------------------------------------------------------------------------------------------- files
def test_the_new_runs_and_read_arms_on_each_species():
    assert [len(V.new_runs(s)) for s in (GULL, WOLF, NES)] == [15, 20, 20]
    labels = {s: sorted({label for label, *_ in V.read_runs(s)}) for s in (GULL, WOLF, NES)}
    assert labels[GULL] == sorted(["B", "B9", "B26", "LSTMB26", "XGBB26", "LSTMS"])
    assert labels[WOLF] == sorted(labels[GULL] + ["ZdT"])
    assert labels[NES] == sorted(labels[GULL] + ["BdM"])
    for species in (GULL, WOLF, NES):
        runs = V.read_runs(species)
        assert all(sorted(seed for l, _, _, seed in runs if l == label) == list(R.SEEDS) for label in labels[species])
        assert {run_id for l, _, run_id, _ in runs if l == "LSTMS"} == {f"LSTMS_{species}_s{s}" for s in R.SEEDS}


def test_comparator_files_are_read_from_their_own_tiers_sealed_predictions():
    files = V.reused_files(NES)
    assert files["P"] == V.SOURCE_PREDICTIONS["2"] / f"{NES}__P.parquet"
    assert files["B_s0"] == V.SOURCE_PREDICTIONS["2"] / f"{NES}__B_s0.parquet"
    assert files["B9_s3"] == V.SOURCE_PREDICTIONS["3"] / f"{NES}__B9_s3.parquet"
    assert files[f"LSTMS_{NES}_s1"] == V.SOURCE_PREDICTIONS["3"] / f"{NES}__LSTMS_{NES}_s1.parquet"
    assert files[f"BdM_{NES}_s4"] == V.SOURCE_PREDICTIONS["4"] / f"{NES}__BdM_{NES}_s4.parquet"
    assert files["XGBB26_s2"] == V.SOURCE_PREDICTIONS["4"] / f"{NES}__XGBB26_s2.parquet"
    assert len(files) == 1 + 7 * 5
    assert not set(files) & {run.run_id for run in R.PLAN}
    assert all(V.PREDICTIONS not in path.parents for path in files.values())


def test_the_report_order_and_the_comparisons_each_species_carries():
    assert V.labels_on(GULL) == ["P", "B3F", "B9F", "B26F", "B", "B9", "B26", "LSTMB26", "XGBB26", "LSTMS"]
    assert "BdTF" in V.labels_on(WOLF) and "ZdT" in V.labels_on(WOLF) and "BdM" not in V.labels_on(WOLF)
    assert V.pairs_on(WOLF) == [("B9F", "B3F"), ("B26F", "B9F"), ("B26F", "B3F"), ("BdTF", "B26F"), ("BdTF", "ZdT")]
    assert V.pairs_on(NES) == [("B3F", "B"), ("B9F", "B9"), ("B26F", "B26"), ("BdMF", "BdM"), ("B26F", "LSTMB26"),
                               ("B26F", "XGBB26"), ("B9F", "B3F"), ("B26F", "B9F"), ("B26F", "B3F"), ("BdMF", "B26F")]


class _Frame:
    """Writes a partial file, then fails, as a walltime kill mid-write would leave it."""

    def to_parquet(self, path, index=False):
        open(path, "wb").write(b"half")
        raise KeyboardInterrupt


def test_a_prediction_file_is_written_whole_or_not_at_all(tmp_path):
    out = tmp_path / f"{NES}__B26F_s0.parquet"
    with pytest.raises(KeyboardInterrupt):
        V.write_atomic(_Frame(), out)
    assert list(tmp_path.iterdir()) == []
    V.write_atomic(pd.DataFrame({"a": [1]}), out)
    assert [p.name for p in tmp_path.iterdir()] == [out.name] and pd.read_parquet(out)["a"].tolist() == [1]


def test_cached_predictions_are_skipped_and_no_model_is_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(S2, "test_frame", lambda species: pd.DataFrame({"individual_id": [1, 2]}))
    monkeypatch.setattr(DG, "load_model", lambda *args: pytest.fail("a cached run was rescored"))
    for run in V.new_runs(NES):
        V.prediction_path(NES, run.run_id).write_bytes(b"")
    written = V.predict_species(NES, "cpu", False, 11, {})
    assert written == [f"{NES}__{run.run_id}.parquet" for run in V.new_runs(NES)]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frames_refuse_a_changed_comparator_file_and_a_stray_prediction(tmp_path, monkeypatch):
    p = tmp_path / f"{NES}__P.parquet"
    pd.DataFrame({"individual_id": [1, 1], "timestamp": [0, 1], "target_lon": [0.0, 0.0], "target_lat": [0.0, 0.0],
                  "error_m": [1.0, 2.0]}).to_parquet(p)
    own = tmp_path / "own"
    own.mkdir()
    monkeypatch.setattr(V, "PREDICTIONS", own)
    monkeypatch.setattr(V, "reused_files", lambda species: {"P": p})
    monkeypatch.setattr(V, "read_runs", lambda species: [])
    monkeypatch.setattr(V, "new_runs", lambda species: [])
    manifest = {"reused": {NES: {"P": _sha(p)}}}
    persistence, frames, parsed = V.load_frames(NES, manifest)
    assert list(frames) == ["P"] and parsed["P"]["arm"] == "P"
    (own / f"{NES}__B26_s0.parquet").write_bytes(b"")
    with pytest.raises(SystemExit, match="does not score"):
        V.load_frames(NES, manifest)
    (own / f"{NES}__B26_s0.parquet").unlink()
    with pytest.raises(SystemExit, match="sha256"):
        V.load_frames(NES, {"reused": {NES: {"P": "0" * 64}}})


# ---------------------------------------------------------------------------------------------------- readings
@pytest.mark.parametrize("rule, difference, expected", [
    ("above", 0.05, False), ("above", 0.0501, True), ("above", -0.2, False),
    ("within", 0.05, True), ("within", -0.05, True), ("within", 0.0501, False), ("within", -0.0501, False),
    ("above", float("nan"), None), ("within", None, None),
])
def test_met_reads_the_point_estimate_against_the_noise_scale(rule, difference, expected):
    assert V.met(rule, difference) is expected


def test_the_verdict_needs_every_comparison():
    assert V.verdict([True, True]) == "met"
    assert V.verdict([True, False, True]) == "not met (2 of 3 met)"
    assert V.verdict([True, None]) == "not scored"


def test_the_five_expectations_are_quoted_from_the_freeze():
    bullets = V.sec6_bullets()
    assert len(bullets) == len(V.EXPECTATIONS) == 5
    assert "lifts each arm above its old-encoder twin" in bullets[0] and "wolf stays flat" in bullets[4]
    assert [e["rule"] for e in V.EXPECTATIONS] == ["above", "within", "within", "above", "within"]
    assert len(V.EXPECTATIONS[0]["pairs"]) == 7 and len(V.EXPECTATIONS[2]["pairs"]) == 9


def test_the_tie_back_is_skill_here_minus_each_arms_own_report():
    here = {"species": {NES: {"labels": ["P", "B", "B26"],
                              "arms": {"B": {"skill": {"travel": {"p90_m": 0.10}}},
                                       "B26": {"skill": {"travel": {"p90_m": 0.20}}}}}}}
    reports = {"2": {"species": {NES: {"comparators": {"B": {"persistence_skill": {"travel_p90_m": 0.09}}}}}},
               "3": {"species": {}},
               "4": {"species": {NES: {"arms": {"B26": {"skill": {"travel": {"p90_m": 0.20}}}}}}}}
    tie = V.tie_back(here, reports)
    assert tie["arms"]["B"]["max_abs_diff"] == pytest.approx(0.01) and tie["arms"]["B"]["tier"] == "2"
    assert tie["arms"]["B26"]["max_abs_diff"] == 0.0 and tie["arms"]["B26"]["n"] == 1
    assert tie["arms"]["ZdT"]["n"] == 0 and tie["arms"]["ZdT"]["max_abs_diff"] is None


# ---------------------------------------------------------------------------------------------------- end to end
FACTORS = {"B3F": 0.85, "B9F": 0.86, "B26F": 0.80, "BdMF": 0.70, "B": 1.0, "B9": 1.0, "B26": 0.99, "BdM": 0.98,
           "LSTMB26": 0.82, "XGBB26": 0.83, "LSTMS": 0.75}  # model error = factor x P error, so skill = 1 - factor


def _synthetic(species):
    """P and every arm on one species: four individuals of 40 windows, all travel; seeds spread around the factor."""
    ids = np.repeat([1, 2, 3, 4], 40)
    error = 5_000.0 + 100.0 * np.arange(len(ids)) % 3_000
    base = {"individual_id": ids, "timestamp": np.arange(len(ids)), "target_lon": 0.0, "target_lat": 0.0}
    frames = {"P": pd.DataFrame(base | {"error_m": error})}
    parsed = {"P": {"arm": "P", "seed": None, "k": None, "draw": None}}
    for label in V.labels_on(species)[1:]:
        for seed in R.SEEDS:
            tag = f"{label}_s{seed}"
            frames[tag] = pd.DataFrame(base | {"error_m": error * FACTORS[label] * (1 + 0.01 * (seed - 2))})
            parsed[tag] = {"arm": label, "seed": seed, "k": None, "draw": None}
    return frames["P"], frames, parsed


def test_one_species_block_scores_every_arm_on_one_set_of_resamples(monkeypatch):
    monkeypatch.setattr(S2, "N_BOOTSTRAP", 200)
    block = V.species_block(NES, *_synthetic(NES))
    for label in V.labels_on(NES)[1:]:
        assert block["against_p"][label]["skill_difference"] == pytest.approx(1 - FACTORS[label])
        assert block["arms"][label]["n_seeds"] == 5
    assert block["comparisons"]["BdMF - B26F"]["skill_difference"] == pytest.approx(0.10)
    # the same draws behind every row: B26F's skill interval is identical in each comparison it enters
    intervals = [block["against_p"]["B26F"]["skill_bootstrap"]["skill"]["B26F"]] + [
        block["comparisons"][key]["skill_bootstrap"]["skill"]["B26F"]
        for key in ("B26F - B26", "B26F - LSTMB26", "BdMF - B26F")]
    assert all(interval == intervals[0] for interval in intervals)
    assert "difference_m" in block["comparisons"]["B26F - B26"]["skill_bootstrap"]

    report = {"species": {NES: block}}
    readings = V.expectations(report)
    assert readings[3]["verdict"] == "met"  # BdMF - B26F on NES = +0.10
    assert readings[0]["verdict"] == "not scored"  # the gull is not in this report
    nes_rows = {row["comparison"]: row for row in readings[0]["rows"] if row["species"] == NES}
    assert nes_rows["B26F - B26"]["met"] is True and nes_rows["B9F - B9"]["met"] is True
    assert [row["alongside"] for row in readings[1]["rows"]] == [False, False, True, True]


def test_a_species_missing_a_seed_is_refused(monkeypatch):
    persistence, frames, parsed = _synthetic(NES)
    del frames["B26_s3"], parsed["B26_s3"]
    with pytest.raises(SystemExit, match="B26 has seeds"):
        V.species_block(NES, persistence, frames, parsed)


def test_the_markdown_carries_each_expectation_its_rows_and_the_skill_table(tmp_path, monkeypatch):
    monkeypatch.setattr(S2, "N_BOOTSTRAP", 200)
    block = V.species_block(NES, *_synthetic(NES))
    report = {"species": {NES: block}, "attestation": {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40},
              "registered_by": "preregistration/tier4_repaired_exploratory_freeze.md", "switches": R.SWITCHES,
              "completeness": {"n_clean": 25, "n_planned": 25}, "evaluation_floor": 25, "n_bootstrap": 200,
              "bootstrap_seed": 42, "noise_scale": 0.05, "tf32": {"predict": {}}}
    report["expectations"] = V.expectations(report)
    report["tie_back"] = V.tie_back(report, {"2": {"species": {}}, "3": {"species": {}}, "4": {"species": {}}})
    path = tmp_path / "report.md"
    V.write_markdown(report, path)
    text = path.read_text()
    assert "### 4. met" in text and "### 1. not scored" in text
    assert "| BdMF - B26F | NES | +0.1000 |" in text and "| B26F - XGBB26 | NES |" in text and "alongside" in text
    assert "| B26F | +0.2000 [" in text and "| ZdT | — |" in text
