# GenAI reference: tests for bird_rotation_repaired_sealed_eval.py, written by GenAI from my spec, which i checked. 
from __future__ import annotations

import copy
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_rotation_repaired_sealed_eval as V

D, RS, RD, R4, S2, E4, BE = V.D, V.RS, V.RD, V.R4, V.S2, V.E4, V.BE
GULL = BE.GULL
STORK = "ciconia_ciconia"
FILLED = "# x\n\n## 6. Expectations\n\n- a\n- b\n- c\n\n## 7. As trained\n\n| runs | 60 of 60 |\n\n## 8. Limitations\n"


def _git(text):
    return lambda *args: subprocess.CompletedProcess(args, 0, stdout=text, stderr="")


# ---------------------------------------------------------------------------------------------------- seal
def test_the_seal_accepts_a_committed_freeze_with_secs_6_and_7_filled(monkeypatch):
    monkeypatch.setattr(V, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(E4, "git", _git(FILLED))
    monkeypatch.setattr(V, "MANIFEST", V.SEALED / "no_such_manifest.json")
    assert V.assert_sealed()["sections_filled"] == [6, 7]


@pytest.mark.parametrize("text, match", [
    (FILLED.replace("| runs | 60 of 60 |", "To fill after training, before the sealed evaluation."), "sec 7 still says"),
    (FILLED.replace("| runs | 60 of 60 |", ""), "sec 7 is empty"),
    (FILLED.replace("- a\n- b\n- c", "- draft"), "sec 6 still says"),
    (FILLED.replace("7. As trained", "7. Runs"), "not headed 'As trained'"),
])
def test_the_seal_refuses_an_unfilled_or_misheaded_section(monkeypatch, text, match):
    monkeypatch.setattr(V, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(E4, "git", _git(text))
    with pytest.raises(SystemExit, match=match):
        V.assert_sealed()


def test_the_committed_freeze_today_is_refused_because_sec_7_is_a_placeholder(monkeypatch):
    text = V.FREEZE.read_text()
    monkeypatch.setattr(V, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(E4, "git", _git(text))
    with pytest.raises(SystemExit, match="sec 7 still says 'To fill'"):
        V.assert_sealed()


def test_the_freeze_check_is_pointed_at_this_freeze_and_restored(monkeypatch):
    saved, seen = S2.FREEZE, {}
    monkeypatch.setattr(S2, "assert_freeze_committed", lambda: seen.setdefault("f", S2.FREEZE) and {})
    V.freeze_committed()
    assert seen["f"] == V.FREEZE and S2.FREEZE == saved


def test_a_manifest_attested_at_another_blob_is_refused(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attestation": {"freeze_blob": "a" * 40}}))
    monkeypatch.setattr(V, "MANIFEST", manifest)
    monkeypatch.setattr(V, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b" * 40})
    monkeypatch.setattr(E4, "git", _git(FILLED))
    with pytest.raises(SystemExit, match="attested at blob"):
        V.assert_sealed()


def test_predict_never_opens_a_test_partition_when_the_seal_fails(monkeypatch):
    def refuse():
        raise SystemExit("FAIL: sealed")

    monkeypatch.setattr(V, "assert_sealed", refuse)
    monkeypatch.setattr(V, "test_frame", lambda bird: pytest.fail("a test partition was opened"))
    monkeypatch.setattr(V, "checked_runs", lambda: pytest.fail("the seal must come first"))
    with pytest.raises(SystemExit, match="sealed"):
        V.run_predictions("cpu")


def test_unclean_runs_or_originals_are_refused(monkeypatch):
    monkeypatch.setattr(RD, "read_records", lambda: {})
    monkeypatch.setattr(RD, "completeness", lambda *a: {"complete": False, "problems": ["BbF6_f0_s0: no record"]})
    with pytest.raises(SystemExit, match="missing or unclean"):
        V.checked_runs()
    monkeypatch.setattr(RD, "completeness", lambda *a: {"complete": True, "problems": []})
    monkeypatch.setattr(RD, "original_check", lambda *a: ["Bb_s0: record missing"])
    with pytest.raises(SystemExit, match="originals differ"):
        V.checked_runs()


# ---------------------------------------------------------------------------------------------------- plan and files
def test_the_60_runs_give_180_prediction_files_each_on_its_own_folds_birds():
    pairs = [(bird, run) for bird in V.BIRDS for run in V.new_runs(bird)]
    assert len(pairs) == 180 and len({(b, r.run_id) for b, r in pairs}) == 180
    assert all(V.FOLD_OF[bird] == run.fold for bird, run in pairs)
    assert {r.run_id for r in V.new_runs(GULL)} == {f"{a}_f0_s{s}" for a in V.NEW for s in (0, 1, 2)}
    assert V.prediction_path(GULL, "BbF6_f0_s0") == V.PREDICTIONS / "larus_fuscus__BbF6_f0_s0.parquet"
    assert V.PREDICTIONS.parent.name == "birds_rotation_repaired_sealed"


def test_the_read_arms_follow_the_plans_and_their_sealed_homes():
    rotation = V.read_runs(STORK)
    assert rotation["P"]["path"] == RS.PREDICTIONS / f"{STORK}__P.parquet"
    seeds = {arm: sorted(e["seed"] for e in rotation.values() if e["arm"] == arm) for arm in V.ORIGINALS}
    assert seeds == {"Bb": [0, 1, 2], "LSTMBb": [0, 1, 2], "XGBBb": [0], "Zb": [0, 1, 2], "LSTMS": [0, 1, 2],
                     "XGBS": [0]}
    assert all(e["path"].parent == RS.PREDICTIONS for e in rotation.values())

    gull = V.read_runs(GULL)
    assert gull["P"]["path"] == S2.PREDICTIONS / f"{GULL}__P.parquet"
    assert gull["Bb_s4"]["path"] == BE.PREDICTIONS / f"{GULL}__Bb_s4.parquet"
    assert gull["tier3:LSTMS_larus_fuscus_s0"]["path"] == V.E3.PREDICTIONS / f"{GULL}__LSTMS_larus_fuscus_s0.parquet"
    assert sorted(e["seed"] for e in gull.values() if e["arm"] == "XGBBb") == [0, 1, 2, 3, 4]
    goose = V.read_runs("anser_albifrons")
    assert goose["P"]["path"].parent == BE.PREDICTIONS and "LSTMS_anser_albifrons_s4" in goose
    assert sum(len(V.read_runs(bird)) for bird in V.BIRDS) == 486


def test_the_read_tags_must_equal_the_rotation_reports():
    report = {"species": {bird: {"arms": {arm: {"tags": {str(i): tag for i, tag in enumerate(
        sorted(t for t, e in V.read_runs(bird).items() if e["arm"] == arm))}} for arm in V.ORIGINALS}}
        for bird in V.BIRDS}}
    assert V.tag_problems(report) == []
    report["species"][STORK]["arms"]["Bb"]["tags"].pop("0")
    assert any("ciconia_ciconia Bb: tags" in p for p in V.tag_problems(report))


def test_the_fold_0_seed_rule_matches_at_0_2_and_publishes_at_five():
    matched, published = V.expected_seeds(GULL), V.expected_seeds(GULL, published=True)
    assert matched["Bb"] == {0, 1, 2} and matched["XGBBb"] == {0} and matched["XGBS"] == {0}
    assert matched["BbF6"] == matched["ZbF6"] == {0, 1, 2}
    assert published["Bb"] == published["XGBBb"] == published["LSTMS"] == {0, 1, 2, 3, 4}
    assert published["BbF6"] == {0, 1, 2}
    assert V.expected_seeds(STORK) == V.expected_seeds(STORK, published=True)


def test_the_rotations_drops_are_checked_against_its_freeze_and_report():
    rules = {b: {"test_birds": RS.SCORED_IN_FREEZE.get(b, (10, 10))[0], "scored": RS.SCORED_IN_FREEZE.get(b, (10, 10))[1],
                 "dropped": [f"x{i}" for i in range(np.subtract(*RS.SCORED_IN_FREEZE.get(b, (10, 10))))]}
             for b in V.BIRDS}
    report = {"species": {b: {"dropped": list(rules[b]["dropped"])} for b in V.BIRDS}}
    assert V.drop_problems(rules, report) == []
    report["species"][STORK]["dropped"] = []
    rules[GULL]["dropped"] = ["1"]
    problems = V.drop_problems(rules, report)
    assert any(p.startswith(STORK) for p in problems) and any("fold 0 drops" in p for p in problems)


def test_a_prediction_file_is_written_whole_or_not_at_all(tmp_path):
    out = tmp_path / "x.parquet"
    R4.write_atomic(pd.DataFrame({"a": [1]}), out)
    assert out.exists() and not list(tmp_path.glob("*.partial"))


def test_cached_predictions_are_skipped_and_no_partition_or_model_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path)
    for run in V.new_runs(GULL):
        V.prediction_path(GULL, run.run_id).write_bytes(b"x")
    monkeypatch.setattr(V, "test_frame", lambda bird: pytest.fail("the partition was read for cached runs"))
    monkeypatch.setattr(RD, "load_model", lambda *a: pytest.fail("a cached run was rescored"))
    assert len(V.predict_bird(GULL, "cpu", False, 11, {})) == 6


def test_a_run_whose_pool_held_the_bird_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(V, "test_frame", lambda bird: pd.DataFrame({"individual_id": [1]}))
    monkeypatch.setattr(RD, "load_model", lambda *a: (None, {"pool_species": [GULL]}))
    with pytest.raises(SystemExit, match="pool held"):
        V.predict_bird(GULL, "cpu", False, 11, {})


def _six_hourly_frame(n_individuals=2, n_fixes=20):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_individuals):
        lon = 4.0 + np.cumsum(rng.normal(0, 0.01, n_fixes))
        lat = 52.0 + np.cumsum(rng.normal(0, 0.01, n_fixes))
        stamps = pd.date_range("2020-06-01", periods=n_fixes, freq="6h")
        rows.append(pd.DataFrame({"individual_id": f"bird{i}", "timestamp": stamps, "lon": lon, "lat": lat}))
    return pd.concat(rows, ignore_index=True)


def test_the_predictor_scores_both_repaired_arms_through_the_diagnostics_loader(tmp_path, monkeypatch):
    """The --predict path on real checkpoints: RD.load_model's checks, then MovementFMPredictor on the unwrapped model,
    which passes `species` and reads ZbF6's forecast from its (forecast, reconstruction) tuple as the diagnostic did."""
    import torch

    from animal_movement_fm.eval.fm_predictor import MovementFMPredictor
    from tests.test_bird_rotation_repaired_diagnostic import _checkpoint, _record

    results, predictions = tmp_path / "results", tmp_path / "predictions"
    results.mkdir()
    predictions.mkdir()
    monkeypatch.setattr(D, "RESULTS", results)
    monkeypatch.setattr(V, "PREDICTIONS", predictions)
    monkeypatch.setattr(V, "test_frame", lambda bird: _six_hourly_frame())
    scored = {run.arm: run for run in V.new_runs(GULL) if run.seed == 0}
    assert set(scored) == {"BbF6", "ZbF6"}
    for run in V.new_runs(GULL):
        if run not in scored.values():
            V.prediction_path(GULL, run.run_id).write_bytes(b"x")  # cached: only the two seed 0 runs are scored
    for run in scored.values():
        record = _record(run, results)
        (results / f"{run.run_id}.json").write_text(json.dumps(record))
        torch.manual_seed(run.seed)
        _checkpoint(results, run.run_id, record["config"], dict(D.SWITCHES))

    assert len(V.predict_bird(GULL, "cpu", False, 11, RD.D.load_payloads())) == 6
    for run in scored.values():
        frame = pd.read_parquet(V.prediction_path(GULL, run.run_id))
        assert len(frame) == 2 * (20 - V.CONTEXT_LENGTH)
        assert np.isfinite(frame[["pred_lon", "pred_lat", "error_m"]].to_numpy()).all()

    x = torch.randn(5, V.CONTEXT_LENGTH - 1, 12)
    for arm, run in scored.items():
        entry = next(e for e in RD.entries(run.fold) if e["run_id"] == run.run_id)
        wrapped, _ = RD.load_model(entry, 11, RD.D.load_payloads())
        model = V.movement_fm(wrapped)
        with torch.no_grad():
            assert isinstance(model(x), tuple) == (arm == "ZbF6")
            forecast = wrapped(x).numpy().astype(float)
        np.testing.assert_allclose(MovementFMPredictor(model, device="cpu")._forward(x.numpy(), None), forecast)
    with pytest.raises(SystemExit, match="Forecast around a MovementFM"):
        V.movement_fm(model)


# ---------------------------------------------------------------------------------------------------- synthetic bird
FACTORS = {"BbF6": 0.85, "ZbF6": 0.90, "Bb": 0.99, "Zb": 1.0, "LSTMBb": 0.82, "XGBBb": 0.83, "LSTMS": 0.75,
           "XGBS": 0.78}  # model error = factor x P error, so skill = 1 - factor; published seeds 3-4 are worse


def _synthetic(bird, published_shift=0.05):
    ids = np.repeat([1, 2, 3, 4], 40)
    error = 5_000.0 + 100.0 * np.arange(len(ids)) % 3_000
    base = {"individual_id": ids, "timestamp": np.arange(len(ids)), "target_lon": 0.0, "target_lat": 0.0}
    frames = {"P": pd.DataFrame(base | {"error_m": error})}
    parsed = {"P": {"arm": "P", "seed": None, "k": None, "draw": None}}
    seeds = V.expected_seeds(bird, published=V.FOLD_OF[bird] == 0)
    for arm, arm_seeds in seeds.items():
        for seed in arm_seeds:
            tag = f"{arm}_s{seed}"
            factor = FACTORS[arm] * (1 + 0.01 * (seed - 1)) + (published_shift if seed > 2 else 0)
            frames[tag] = pd.DataFrame(base | {"error_m": error * factor})
            parsed[tag] = {"arm": arm, "seed": seed, "k": None, "draw": None}
    return frames["P"], frames, parsed


def test_matched_and_published_blocks_differ_only_in_fold_0s_originals():
    persistence, frames, parsed = _synthetic(GULL)
    matched_frames, matched = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL), "m")
    _, published = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL, True), "p")
    assert not any(tag.endswith(("_s3", "_s4")) for tag in matched_frames)
    assert matched["Bb"]["n_seeds"] == 3 and published["Bb"]["n_seeds"] == 5 and matched["XGBBb"]["n_seeds"] == 1
    assert V.E3.travel_skill(matched, "Bb") == pytest.approx(1 - 0.99)
    assert V.E3.travel_skill(published, "Bb") < V.E3.travel_skill(matched, "Bb")
    assert V.E3.travel_skill(published, "BbF6") == V.E3.travel_skill(matched, "BbF6")


def test_a_bird_missing_a_seed_is_refused():
    persistence, frames, parsed = _synthetic(STORK)
    del frames["Zb_s2"]
    with pytest.raises(SystemExit, match="Zb has seeds"):
        V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(STORK), STORK)


def test_one_set_of_resamples_behind_every_comparison(monkeypatch):
    monkeypatch.setattr(S2, "N_BOOTSTRAP", 200)
    persistence, frames, parsed = _synthetic(STORK)
    chosen, arms = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(STORK), STORK)
    block = V.compare_all(arms, chosen, persistence, 100.0)
    assert set(block["against_p"]) == set(V.LABELS[1:])
    assert set(block["comparisons"]) == {V.name(l, r) for _, l, r in V.SEC4 if r != "P"}
    assert V.reading(block, "BbF6 - P")["skill_difference"] == pytest.approx(0.15)
    assert V.reading(block, "ZbF6 - BbF6")["skill_difference"] == pytest.approx(-0.05)
    intervals = [block["against_p"]["BbF6"]["skill_bootstrap"]["skill"]["BbF6"]] + [
        block["comparisons"][c]["skill_bootstrap"]["skill"]["BbF6"] for c in ("BbF6 - Bb", "BbF6 - LSTMBb")]
    assert all(i == intervals[0] for i in intervals)
    assert "difference_m" in block["comparisons"]["LSTMBb - LSTMS"]["skill_bootstrap"]


# ---------------------------------------------------------------------------------------------------- tie-back
def _sealed_from(primary, matched=None):
    sealed = {"arms": {arm: {"headline": primary[arm]["headline"], "skill": primary[arm]["skill"]} for arm in RS.ARMS},
              "readings": {"gaps": {f: {"skill_difference": V._gap(primary, *pair)} for f, pair in V.FAMILIES.items()}},
              "n_test_individuals_scored": 4, "n_test_windows": 160}
    sealed = copy.deepcopy(sealed)
    if matched is not None:
        sealed["seeds_0_2"] = {"skill": {arm: V.E3.travel_skill(matched, arm) for arm in V.ORIGINALS},
                               "gaps": {f: V._gap(matched, *pair) for f, pair in V.FAMILIES.items()}}
    return sealed


def test_the_tie_back_passes_on_equal_reports_and_fails_on_any_difference():
    persistence, frames, parsed = _synthetic(GULL)
    _, matched = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL), "m")
    _, published = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL, True), "p")
    counts = {"n_test_individuals_scored": 4, "n_test_windows": 160}
    sealed = _sealed_from(published, matched)
    tie = V.tie_back(GULL, published, matched, sealed, counts)
    assert tie["problems"] == [] and tie["max_abs_skill_diff"] == 0.0

    sealed["arms"]["LSTMS"]["headline"]["travel"]["p90_m"] *= 1 + 1e-9
    sealed["seeds_0_2"]["gaps"]["XGB"] += 1e-6
    problems = V.tie_back(GULL, published, matched, sealed, counts | {"n_test_windows": 159})["problems"]
    assert any("LSTMS travel p90_m" in p for p in problems)
    assert any("XGB seeds 0-2 gap" in p for p in problems)
    assert any("n_test_windows" in p for p in problems)


def test_the_report_refuses_before_any_bootstrap_when_the_tie_back_fails(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"rotation": {"report_sha256": "r", "manifest_sha256": "m"}}))
    monkeypatch.setattr(V, "MANIFEST", manifest)
    monkeypatch.setattr(V, "rotation_inputs", lambda: {"report_sha256": "r", "manifest_sha256": "m", "rules": {},
                                                       "report": {"species": {b: {} for b in V.BIRDS}}})
    monkeypatch.setattr(V, "checked_runs", lambda: ({}, 11, {}))
    monkeypatch.setattr(V.RT, "check_boundaries", lambda: {b: 1.0 for b in V.ROTATION})
    monkeypatch.setattr(BE, "check_boundaries", lambda: {b: 1.0 for b in V.FOLD0})
    monkeypatch.setattr(V, "bird_inputs", lambda *a: {})
    monkeypatch.setattr(V, "check_bird", lambda bird, *a: {"problems": [f"{bird}: off"], "max_abs_skill_diff": 1.0,
                                                           "n_arms": 1})
    monkeypatch.setattr(V, "compare_all", lambda *a: pytest.fail("the bootstrap ran before the tie-back"))
    with pytest.raises(SystemExit, match="differ from the rotation's sealed report"):
        V.build_report()


def test_the_report_refuses_a_rotation_report_changed_after_predict(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"rotation": {"report_sha256": "old", "manifest_sha256": "m"}}))
    monkeypatch.setattr(V, "MANIFEST", manifest)
    monkeypatch.setattr(V, "rotation_inputs", lambda: {"report_sha256": "new", "manifest_sha256": "m"})
    with pytest.raises(SystemExit, match="changed after --predict"):
        V.build_report()


# ---------------------------------------------------------------------------------------------------- frames
def test_frames_refuse_a_changed_read_file_and_a_stray_prediction(tmp_path, monkeypatch):
    persistence, frames, _ = _synthetic(STORK)
    p = tmp_path / "P.parquet"
    persistence.to_parquet(p, index=False)
    own = tmp_path / "own"
    own.mkdir()
    monkeypatch.setattr(V, "PREDICTIONS", own)
    monkeypatch.setattr(V, "read_runs", lambda bird: {"P": {"path": p, "arm": "P", "seed": None}})
    monkeypatch.setattr(V, "new_runs", lambda bird: [])
    with pytest.raises(SystemExit, match="differs from the sha256"):
        V.load_frames(STORK, {"reused": {STORK: {"P": "0" * 64}}})
    good = {"reused": {STORK: {"P": R4.sha256(p)}}}
    assert len(V.load_frames(STORK, good)[0]) == len(persistence)
    frames["Bb_s0"].to_parquet(own / f"{STORK}__Bb_f2_s0.parquet", index=False)
    with pytest.raises(SystemExit, match="does not score"):
        V.load_frames(STORK, good)


# ---------------------------------------------------------------------------------------------------- across birds, sec 6
def test_the_bird_sets_and_fold_6():
    sizes = {k: len(v) for k, v in V.BIRD_SETS.items()}
    assert sizes == {"all": 30, "not_fold0": 27, "not_fold6": 27, "not_fold0_or_fold6": 24}
    assert not set(V.BIRD_SETS["not_fold6"]) & set(RS.RT.FOLDS[6])
    assert not set(V.BIRD_SETS["not_fold0"]) & set(V.FOLD0)


def _report(values: dict[str, list[float]]) -> dict:
    """A report whose matched comparisons take `values[comparison][i]` on bird i (others 0), and published ones +9."""
    species = {}
    for i, bird in enumerate(V.BIRDS):
        block = {"against_p": {}, "comparisons": {}}
        for comparison in V.COMPARISON_NAMES:
            value = {"skill_difference": values.get(comparison, [0.0] * 30)[i]}
            left, right = comparison.split(" - ")
            (block["against_p"].__setitem__(left, value) if right == "P"
             else block["comparisons"].__setitem__(comparison, value))
        species[bird] = {"matched": block}
        if V.FOLD_OF[bird] == 0:
            species[bird]["published"] = {"against_p": {}, "comparisons": {
                c: {"skill_difference": 9.0} for c in V.COMPARISON_NAMES}}
    return {"species": species}


def test_across_birds_counts_medians_and_sets_use_matched_seeds_only():
    values = {"BbF6 - Bb": [0.10] * 16 + [0.0] * 10 + [-0.10] * 4}
    across = V.across_birds(_report(values))["BbF6 - Bb"]
    assert across["all"] | {} == {"n_birds": 30, "n_scored": 30, "median": pytest.approx(0.10), "above": 16,
                                  "within": 10, "below": 4, "not scored": 0, "negative": 4}
    assert across["not_fold0"]["above"] == 13  # fold 0 is the first three birds
    assert across["not_fold6"]["n_birds"] == 27 and across["not_fold0_or_fold6"]["n_birds"] == 24


def test_sec_6_reads_more_than_15_of_30_and_medians_below_minus_0_05():
    bullets = ["one", "two", "three"]
    values = {"BbF6 - Bb": [0.10] * 16 + [0.0] * 14, "ZbF6 - Zb": [0.10] * 15 + [0.0] * 15,
              "BbF6 - LSTMBb": [-0.06] * 30, "BbF6 - XGBBb": [-0.05] * 30,
              "ZbF6 - BbF6": [-0.001] * 16 + [0.01] * 14}
    report = _report(values)
    report["across_birds"] = V.across_birds(report)
    readings = V.expectations(report, bullets)
    first = {row["comparison"]: row for row in readings[0]["rows"]}
    assert first["BbF6 - Bb"]["met"] is True and first["ZbF6 - Zb"]["met"] is False  # 16 > 15; 15 is not
    assert readings[0]["verdict"] == "not met (1 of 2 met)"
    second = {row["comparison"]: row for row in readings[1]["rows"]}
    assert second["BbF6 - LSTMBb"]["met"] is True and second["BbF6 - XGBBb"]["met"] is False  # -0.05 is not below
    assert readings[2]["verdict"] == "met" and readings[2]["rows"][0]["value"] == 16  # below 0, not below -0.05
    assert readings[2]["rows"][0]["elsewhere"]["not_fold0"] == 13


def test_sec_6_is_not_scored_if_any_bird_is_missing():
    values = {"ZbF6 - BbF6": [-0.1] * 29 + [float("nan")]}
    report = _report(values)
    report["across_birds"] = V.across_birds(report)
    assert V.expectations(report, ["a", "b", "c"])[2]["verdict"] == "not scored"


def test_the_three_expectations_are_quoted_from_the_freeze():
    bullets = V.sec6_bullets()
    assert len(bullets) == 3 and bullets[0].startswith("Both repairs help")
    assert "ZbF6 − BbF6 is negative" in bullets[2]


# ---------------------------------------------------------------------------------------------------- figure 5.3
def test_the_figure_rows_carry_every_arm_and_contrast_with_fits_and_files(monkeypatch):
    monkeypatch.setattr(S2, "N_BOOTSTRAP", 200)
    persistence, frames, parsed = _synthetic(GULL)
    inputs = {"persistence": persistence, "parsed": parsed, "dropped": [],
              "counts": {"n_test_individuals_scored": 4, "n_test_windows": 160},
              "matched": V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL), "m"),
              "published": V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(GULL, True), "p")}
    monkeypatch.setattr(V, "read_runs", lambda bird: {})
    monkeypatch.setattr(V.RD, "pool_names", lambda fold, payloads: ["a"] * 22)
    entry = V.bird_entry(GULL, inputs, 100.0, {}, {"new": {GULL: {"BbF6_s0": "abc"}}})
    report = {"study": "birds_rotation_repaired", "registered_by": "r", "attestation": {}, "rotation": {"attestation": {}},
              "species": {GULL: entry}}
    figure = V.figure_rows(report)
    rows = figure["rows"]
    per_set = len(V.LABELS) - 1 + sum(r != "P" for _, _, r in V.SEC4)
    assert len(rows) == 2 * per_set
    keys = {"species", "fold", "carried", "seed_set", "n_source_species", "n_test_individuals", "n_test_windows",
            "comparison", "arm_or_family", "estimate", "ci95_lo", "ci95_hi", "left_n_fits", "right_n_fits",
            "provenance"}
    assert all(keys <= set(row) for row in rows)
    matched = {(r["comparison"], r["arm_or_family"]): r for r in rows if r["seed_set"] == "matched"}
    assert matched["skill_vs_p", "BbF6"]["estimate"] == pytest.approx(0.15)
    assert matched["skill_vs_p", "Bb"]["left_n_fits"] == 3 and matched["specialist_gap", "XGB"]["right_n_fits"] == 1
    assert matched["contrast", "BbF6 - Bb"]["sec4"] == 2
    assert matched["skill_vs_p", "BbF6"]["provenance"]["BbF6"][0]["sha256"] == "abc"
    published = [r for r in rows if r["seed_set"] == "published" and r["arm_or_family"] == "Bb"]
    assert published[0]["left_n_fits"] == 5 and all(r["carried"] for r in rows)
    assert json.loads(json.dumps(figure, default=str))["figure"] == "5.3"


def test_a_full_synthetic_report_writes_its_markdown_and_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(S2, "N_BOOTSTRAP", 50)
    monkeypatch.setattr(V, "read_runs", lambda bird: {})
    monkeypatch.setattr(V.RD, "pool_names", lambda fold, payloads: ["a"] * 27)
    entries = {}
    for bird in V.BIRDS:
        persistence, frames, parsed = _synthetic(bird)
        inputs = {"persistence": persistence, "parsed": parsed, "dropped": [],
                  "counts": {"n_test_individuals_scored": 4, "n_test_windows": 160},
                  "matched": V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(bird), bird)}
        if V.FOLD_OF[bird] == 0:
            inputs["published"] = V.summarise(persistence, frames, parsed, 100.0, V.expected_seeds(bird, True), bird)
        entries[bird] = V.bird_entry(bird, inputs, 100.0, {}, {})
    report = {"study": "birds_rotation_repaired", "registered_by": "preregistration/x.md",
              "attestation": {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40}, "rotation": {"attestation": {}},
              "switches": D.SWITCHES, "completeness": {"n_clean": 60, "n_planned": 60}, "evaluation_floor": 25,
              "n_bootstrap": 50, "bootstrap_seed": 42, "noise_scale": 0.05, "most": 15, "tf32": {"predict": {}},
              "species": entries, "tie_back": {"against": "r.json", "max_abs_skill_diff": 0.0}}
    report["across_birds"] = V.across_birds(report)
    report["expectations"] = V.expectations(report)
    path = tmp_path / "report.md"
    V.write_markdown(report, path)
    text = path.read_text()
    # every bird: BbF6 - Bb +0.14 and ZbF6 - Zb +0.10 (above), BbF6 - LSTMBb = -0.03, ZbF6 - BbF6 = -0.05 (below 0)
    assert "### 6.1 met" in text and "### 6.2 not met (0 of 2 met)" in text and "### 6.3 met" in text
    assert "| BbF6 - Bb | 30 | 30 | +0.1400 | 30 | 0 | 0 | 0 |" in text
    assert "## Fold 0 with the originals' published five seeds" in text
    assert text.count("| 0 | *L. fuscus* |") == 2
    figure = V.figure_rows(report)
    assert len(figure["rows"]) == 33 * (len(V.LABELS) - 1 + sum(r != "P" for _, _, r in V.SEC4))
    json.dumps(figure, default=str)
