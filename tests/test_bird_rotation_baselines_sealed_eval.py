# GenAI reference: I listed what the baselines evaluator must guarantee and GenAI wrote these tests from that, on synthetic data only.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_rotation_baselines_sealed_eval as B
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.models.arima import ARIMA
from animal_movement_fm.models.averaging import DisplacementAveraging

T0 = pd.Timestamp("2021-01-01")


def track(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "individual_id": "a",
        "bin_index": np.arange(n),
        "timestamp": T0 + pd.to_timedelta(6 * np.arange(n), unit="h"),
        "lon": 10 + np.cumsum(rng.normal(0, 0.01, n)),
        "lat": 50 + np.cumsum(rng.normal(0, 0.01, n)),
    })


# Baselines and seal
def test_baselines_are_the_bird_study_s_unchanged():
    models = dict(B.baseline_models())
    assert list(models) == ["V", "ARIMA", "AVG3", "AVG10"]
    assert isinstance(models["ARIMA"], ARIMA) and models["ARIMA"].order == (2, 0, 2)
    assert models["ARIMA"].min_history == 10
    assert [models[tag].k for tag in ("V", "AVG3", "AVG10")] == [1, 3, 10]
    assert all(isinstance(models[tag], DisplacementAveraging) for tag in ("V", "AVG3", "AVG10"))
    assert set(B.ARMS) == set(B.BE.TRAINING_FREE)


def test_predict_refuses_before_any_test_read_without_the_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "FREEZE", tmp_path / "missing_freeze.md")

    def opened(*_):
        raise AssertionError("a test partition was opened")

    monkeypatch.setattr(B.BR, "test_frame", opened)
    with pytest.raises(SystemExit, match="does not exist"):
        B.run_predictions(workers=1)


def test_the_seal_points_the_tier2_check_at_this_freeze_and_restores_it(monkeypatch, tmp_path):
    seen = {}

    def committed():
        seen["freeze"] = B.S2.FREEZE
        return {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40}

    original = B.S2.FREEZE
    monkeypatch.setattr(B.S2, "assert_freeze_committed", committed)
    monkeypatch.setattr(B, "MANIFEST", tmp_path / "manifest.json")
    B.assert_sealed()
    assert seen["freeze"] == B.FREEZE and B.FREEZE.name == "birds_rotation_baselines_exploratory_freeze.md"
    assert B.S2.FREEZE == original


def test_a_manifest_from_another_freeze_blob_is_refused(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"attestation": {"freeze_blob": "' + "a" * 40 + '"}}')
    monkeypatch.setattr(B, "MANIFEST", manifest)
    monkeypatch.setattr(B.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40})
    with pytest.raises(SystemExit, match="Rebuild the report"):
        B.assert_sealed()


# P identity
def test_p_identity_passes_on_the_same_windows_and_fails_on_any_change():
    frame = track()
    sealed = evaluate(B.persistence_model(), frame, context_length=CONTEXT_LENGTH, show_progress=False)
    sealed = sealed.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)
    again = evaluate(B.persistence_model(), frame, context_length=CONTEXT_LENGTH, show_progress=False)
    assert len(sealed) > 0 and B.p_identity(again, sealed)["identical"] is True
    changed = again.copy()
    changed.loc[0, "error_m"] += 1e-9
    assert B.p_identity(changed, sealed)["identical"] is False
    assert B.p_identity(again.iloc[1:], sealed)["identical"] is False


# Frames and checks
def test_tags_and_fold0_sources():
    assert B.parse_tag("ARIMA", {}) == {"arm": "ARIMA", "seed": None, "k": None, "draw": None}
    assert B.parse_tag("LSTMS_ciconia_ciconia_s0", {}) is None
    gull = B.fold0_paths(B.BE.GULL)
    assert gull["V"].parent == B.S2.PREDICTIONS and gull["ARIMA"].parent == B.E3.PREDICTIONS
    goose = B.fold0_paths("anser_albifrons")
    assert set(goose) == set(B.ARMS) and {p.parent for p in goose.values()} == {B.BE.PREDICTIONS}
    assert not set(B.FOLD0) & set(B.ROTATION)


def headline(value: float) -> dict:
    return {stratum: {name: value for name in B.E3.STATISTICS} for stratum in (*B.E3.STRATA, "all")}


def test_check_against_refuses_any_statistic_that_differs_from_the_sealed_report():
    arms = {"P": {"headline": headline(100.0)}}
    B.check_against("same", arms, {"P": {"headline": headline(100.0)}}, ["P"])
    with pytest.raises(SystemExit, match="differs"):
        B.check_against("P differs", arms, {"P": {"headline": headline(100.0 + 1e-6)}}, ["P"])
    with pytest.raises(SystemExit):
        B.check_against("missing", arms, {}, ["P"])


def test_bands_use_the_noise_scale():
    assert [B.band(v) for v in (0.06, 0.05, -0.05, -0.06, float("nan"))] == [
        "above", "within", "within", "below", "not scored"]


# Scheduling
def test_schedule_submits_largest_first_and_projects_the_makespan():
    order, makespan = B.schedule({"a": 10.0, "b": 6.0, "c": 5.0, "d": 4.0}, workers=2)
    assert order == ["a", "b", "c", "d"]
    assert makespan == 14.0
    assert B.schedule({"a": 10.0, "b": 6.0}, workers=8)[1] == 10.0


# Worker and report entry, end to end on synthetic birds
BIRD = "anser_anser"


def tracks() -> pd.DataFrame:
    return pd.concat([track(40, seed).assign(individual_id=name) for seed, name in enumerate("abc")],
                     ignore_index=True)


@pytest.fixture(scope="module")
def predicted(tmp_path_factory):
    """One synthetic rotation bird predicted by the worker, with its sealed P written as the rotation wrote it."""
    root = tmp_path_factory.mktemp("baselines")
    frame = tracks()
    sealed = root / "P.parquet"
    evaluate(B.persistence_model(), frame, context_length=CONTEXT_LENGTH, show_progress=False).to_parquet(
        sealed, index=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(B.BR, "test_frame", lambda bird: frame)
        patch.setattr(B, "rotation_p_path", lambda bird: sealed)
        patch.setattr(B, "PREDICTIONS", root)
        first = B.predict_bird(BIRD, overwrite=False)
        second = B.predict_bird(BIRD, overwrite=False)
    return root, sealed, first, second


def test_the_worker_checks_p_then_writes_aligned_baselines_once(predicted):
    root, _, first, second = predicted
    assert first["p_identity"]["identical"] is True
    assert sorted(path.name for path in root.glob(f"{BIRD}__*.parquet")) == sorted(
        f"{BIRD}__{tag}.parquet" for tag in B.BASELINES)
    assert set(first["seconds"]) == {*B.BASELINES, "total"}
    assert all(name.endswith("(exists, skipped)") for name in second["written"])


def test_the_worker_writes_nothing_when_p_differs_from_the_sealed_file(monkeypatch, tmp_path):
    sealed = tmp_path / "P.parquet"
    evaluate(B.persistence_model(), track(40, 7), context_length=CONTEXT_LENGTH, show_progress=False).to_parquet(
        sealed, index=False)
    out = tmp_path / "predictions"
    out.mkdir()
    monkeypatch.setattr(B.BR, "test_frame", lambda bird: tracks())
    monkeypatch.setattr(B, "rotation_p_path", lambda bird: sealed)
    monkeypatch.setattr(B, "PREDICTIONS", out)
    with pytest.raises(SystemExit, match="P recomputed differs"):
        B.predict_bird(BIRD, overwrite=False)
    assert not list(out.iterdir())


def sealed_rotation_entry(arms: dict) -> dict:
    stored = {arm: {key: arms["V"][key] for key in ("n_seeds", "headline", "skill")} for arm in B.SPECIALISTS}
    return {"arms": {"P": {"headline": arms["P"]["headline"]}, **stored}, "readings": {}}


def test_a_bird_entry_scores_every_baseline_against_p_with_intervals(predicted, monkeypatch):
    root, sealed, _, _ = predicted
    monkeypatch.setattr(B, "rotation_p_path", lambda bird: sealed)
    monkeypatch.setattr(B, "PREDICTIONS", root)
    rules = {BIRD: {"dropped": []}}
    arms, *_ = B.score_bird(BIRD, rules, 100.0)
    entry = B.bird_entry(BIRD, rules, 100.0, sealed_rotation_entry(arms), None)
    assert list(entry["arms"]) == list(B.ARMS) and entry["n_test_individuals_scored"] == 3
    for arm in B.BASELINES:
        bands = entry["arms"][arm]["intervals"]
        assert all(bands[series] is not None for series in B.MT.INTERVALS), arm
        assert entry["against_p"][arm]["left"] == arm and entry["against_p"][arm]["right"] == "P"
    assert entry["arms"]["P"]["intervals"]["travel_p90_skill"]["lo"] == 0.0
    assert set(entry["specialists"]) == set(B.SPECIALISTS)
    assert set(entry["arima_fallback_rate"]) == {"lon_axis", "lat_axis", "both_axes", "either_axis"}

    dropped = B.bird_entry(BIRD, {BIRD: {"dropped": ["b"]}}, 100.0,
                           sealed_rotation_entry(B.score_bird(BIRD, {BIRD: {"dropped": ["b"]}}, 100.0)[0]), None)
    assert dropped["n_test_individuals_scored"] == 2 and dropped["dropped"] == ["b"]

    with pytest.raises(SystemExit, match="P differs from the rotation report"):
        B.bird_entry(BIRD, {BIRD: {"dropped": ["b"]}}, 100.0, sealed_rotation_entry(arms), None)

    monkeypatch.setattr(B, "BIRDS", (BIRD,))
    monkeypatch.setattr(B, "ROTATION", (BIRD,))
    report = {"registered_by": "x", "evaluation_floor": 25, "n_bootstrap": 2000, "bootstrap_seed": 42,
              "species": {BIRD: entry}}
    report["across_birds"] = B.across_birds(report)
    assert report["across_birds"]["30 birds"]["arms"]["V"]["travel_p90"]["n"] == 1
    path = root / "report.md"
    B.write_markdown(report, path)
    assert "| LSTMS (rotation) |" in path.read_text()


def test_a_spawned_worker_imports_the_evaluator():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(B.schedule, {"a": 2.0, "b": 1.0}, 1).result() == (["a", "b"], 3.0)


# Diagnostic
def test_diagnostic_reads_no_test_target(monkeypatch, capsys):
    def opened(*_args, **_kwargs):
        raise AssertionError("a test partition or prediction file was read")

    monkeypatch.setattr(B.BR, "test_frame", opened)
    monkeypatch.setattr(B.pd, "read_parquet", opened)
    monkeypatch.setattr(B.E3, "load_frame", opened)
    monkeypatch.setattr(B, "rotation_manifest", lambda: {"scoring_rules": {b: {"dropped": []} for b in B.BIRDS}})
    monkeypatch.setattr(B, "rotation_report", lambda: {"species": {b: {"n_test_windows": 90} for b in B.BIRDS}})
    monkeypatch.setattr(B, "boundaries", lambda: {b: 1.0 for b in B.BIRDS})
    monkeypatch.setattr(B, "n_rows", lambda path: 100 if "baselines" not in str(path) else None)
    monkeypatch.setattr(B, "arima_probe", lambda bird, windows: {
        "bird": bird, "partition": "train", "individuals": 1, "windows": windows, "seconds": 1.0,
        "s_per_window": 0.1, "host": "h", "processor": "p"})
    B.diagnostic(workers=4, probe_bird="ciconia_ciconia", probe_windows=10)
    out = capsys.readouterr().out
    assert "27 of 27 sealed P files present, 2,700 windows" in out
    assert "15 of 15 sealed files present" in out
    assert "| test-partition reads | none" in out
