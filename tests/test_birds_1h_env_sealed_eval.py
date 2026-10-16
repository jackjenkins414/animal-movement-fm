# GenAI reference: I specified these tests for the env study's sealed evaluator (contrast arithmetic and sign, intervals
# from shared resamples, the refusals, a diagnostic that never opens test, the 1 h parts unchanged) and GenAI wrote them.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_1h_env_sealed_eval as V
from animal_movement_fm.data import era5

E, env, b1h, RS, S2, E3 = V.E, V.env, V.b1h, V.RS, V.S2, V.E3
TEST_WINDOWS, ENV_WINDOWS, HORIZON_WINDOWS = E.test_windows, env.env_windows, b1h.horizon_windows  # before the fixture
T0 = pd.Timestamp("2021-01-01", tz="UTC")
BOUNDARY = 1.0  # every persistence error is above it, so every window is travel


def _refuse(*args, **kwargs):
    raise AssertionError("a test reached a test-partition reader")


@pytest.fixture(autouse=True)
def _no_test_partition(monkeypatch):
    """Every reader of test-bird fixes, ids or windows raises, so no test here can open a test partition."""
    for owner, name in ((E, "test_windows"), (E, "scored_ids"), (E, "scoring_state"), (E.RS, "load_corpus"),
                        (E.RS, "test_frame"), (env, "env_windows"), (b1h, "horizon_windows")):
        monkeypatch.setattr(owner, name, _refuse)


# --------------------------------------------------------------------------- 1 h study untouched
def test_the_1h_parts_and_plan_are_unchanged_and_bound_restores_the_shared_globals():
    assert E.PARTS == env.MAIN_PARTS == {"main": (0, 269), "ablation": (270, 629)}
    env.check_main_study()
    assert len(V.MAIN_RUNS) == 90 and {r.arm for r in V.MAIN_RUNS} == {"Bb", "LSTMBb", "XGBBb"}
    assert all(r.context == E.CORE and not r.species for r in V.MAIN_RUNS)
    before = (RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR)
    with pytest.raises(RuntimeError), V.bound():
        assert (RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR) == (V.FREEZE, V.MANIFEST, 150, 150)
        raise RuntimeError
    assert (RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR) == before
    assert V.FREEZE.name == "birds_1h_env_exploratory_freeze.md" and V.SEALED.name == "birds_1h_env_sealed"


# --------------------------------------------------------------------------- statistic
SCALES = {"BbE": 0.7, "BbR": 0.9, "LSTMBbE": 0.87, "LSTMBbR": 0.9, "XGBBbE": 0.96, "XGBBb": 0.9, "Bb": 0.88,
          "LSTMBb": 0.92}  # model error / persistence error on every window, so skill = 1 - scale exactly


def _persistence(n_individuals=6, n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"individual_id": f"i{k}", "timestamp": T0 + pd.Timedelta(hours=i), "target_lon": 0.1 * i,
             "target_lat": 1.0} for k in range(n_individuals) for i in range(n)]
    frame = pd.DataFrame(rows)
    frame["error_m"] = rng.gamma(2.0, 500.0, size=len(frame)) + 200.0
    return frame


def _cell(scales=SCALES, noise=0.0):
    """P and every arm's frames on the same windows; neural arms at three seeds, the median seed at the arm's scale."""
    persistence = _persistence()
    rng = np.random.default_rng(7)
    frames, records = {"P": persistence}, {}
    for arm, scale in scales.items():
        seeds = (0,) if arm.startswith("XGB") else (0, 1, 2)
        for seed in seeds:
            tag = f"{arm}_s{seed}"
            factor = scale * (1 + 0.01 * (seed - 1) * (len(seeds) > 1))
            jitter = 1 + noise * rng.normal(size=len(persistence))
            frames[tag] = persistence.assign(error_m=persistence["error_m"] * factor * jitter)
            records[tag] = {"arm": arm, "seed": seed}
    expected = {arm: ({0} if arm.startswith("XGB") else {0, 1, 2}) for arm in scales}
    return persistence, frames, records, expected


def test_env_effects_contrasts_and_hardware_have_the_registered_arithmetic_and_sign():
    persistence, frames, records, expected = _cell()
    arms, readings = V.cell_readings(persistence, frames, records, BOUNDARY, expected, "synthetic")
    skill = {arm: 1 - scale for arm, scale in SCALES.items()}
    assert readings["against_p"] == pytest.approx(skill)
    effect = {family: skill[a] - skill[b] for family, (a, b) in V.EFFECTS.items()}
    assert effect == pytest.approx({"Bb": 0.2, "LSTMBb": 0.03, "XGBBb": -0.06})
    for family, value in effect.items():
        reading = readings["env_effect"][family]
        assert (reading["left"], reading["right"]) == V.EFFECTS[family]
        assert reading["skill_difference"] == pytest.approx(value)
        boot = reading["skill_bootstrap"]["skill_difference"]
        assert boot["lo"] == pytest.approx(value) and boot["hi"] == pytest.approx(value)
    assert [RS.band(readings["env_effect"][f]["skill_difference"]) for f in V.EFFECTS] == ["above", "within", "below"]
    # (BbE - BbR) - (LSTMBbE - LSTMBbR) = 0.2 - 0.03; (BbE - BbR) - (XGBBbE - XGB-Bb) = 0.2 + 0.06
    for name, value in (("Bb_vs_LSTMBb", 0.17), ("Bb_vs_XGBBb", 0.26)):
        reading = readings["contrasts"][name]
        assert reading["difference"] == pytest.approx(value) and reading["band"] == "above"
        boot = reading["skill_bootstrap"]["difference"]
        assert boot["lo"] == pytest.approx(value) and boot["hi"] == pytest.approx(value) and boot["excludes_zero"]
    assert readings["contrasts"]["Bb_vs_LSTMBb"]["label"] == "(BbE - BbR) - (LSTMBbE - LSTMBbR)"
    assert readings["contrasts"]["Bb_vs_XGBBb"]["label"] == "(BbE - BbR) - (XGBBbE - XGBBb)"
    # hardware: the rerun minus the main study's run, a difference only
    assert readings["hardware"]["Bb"]["skill_difference"] == pytest.approx(skill["BbR"] - skill["Bb"])
    assert readings["hardware"]["LSTMBb"]["skill_difference"] == pytest.approx(skill["LSTMBbR"] - skill["LSTMBb"])
    assert "band" not in readings["hardware"]["Bb"]


def test_a_transformer_that_gains_less_gives_negative_contrasts():
    scales = SCALES | {"BbE": 0.9, "BbR": 0.9}  # Bb gains nothing
    persistence, frames, records, expected = _cell(scales)
    _, readings = V.cell_readings(persistence, frames, records, BOUNDARY, expected, "synthetic")
    assert readings["contrasts"]["Bb_vs_LSTMBb"]["difference"] == pytest.approx(-0.03)
    assert readings["contrasts"]["Bb_vs_XGBBb"]["difference"] == pytest.approx(0.06)
    assert readings["contrasts"]["Bb_vs_LSTMBb"]["band"] == "within"


def test_contrast_intervals_come_from_the_same_resamples_as_the_env_effects():
    persistence, frames, records, expected = _cell(noise=0.3)
    with V.bound():
        arms = RS.arm_blocks(E3.summarise_all(frames, persistence, BOUNDARY, records, parse=E.parse_tag))
        resampler = RS.Resampler(persistence, BOUNDARY)
        effects = {f: RS.compare(a, b, arms, frames, persistence, BOUNDARY, resampler) for f, (a, b) in V.EFFECTS.items()}
        cached = dict(resampler._cache)
        result = V.contrast("Bb", "LSTMBb", arms, frames, resampler)
    assert resampler._cache.keys() == cached.keys() and all(resampler._cache[k] is cached[k] for k in cached)
    s = {arm: 1 - cached[arm] / resampler.reference for arm in ("BbE", "BbR", "LSTMBbE", "LSTMBbR")}
    for family in ("Bb", "LSTMBb"):
        a, b = V.EFFECTS[family]
        assert effects[family]["skill_bootstrap"]["skill_difference"]["lo"] == np.nanpercentile(s[a] - s[b], 2.5)
    values = (s["BbE"] - s["BbR"]) - (s["LSTMBbE"] - s["LSTMBbR"])
    boot = result["skill_bootstrap"]
    assert (boot["n_resamples"], boot["seed"]) == (2000, 42)
    assert boot["difference"]["lo"] == np.nanpercentile(values, 2.5)
    assert boot["difference"]["hi"] == np.nanpercentile(values, 97.5)
    assert boot["difference"]["lo"] < result["difference"] < boot["difference"]["hi"]


def test_a_cell_missing_a_seed_is_refused():
    persistence, frames, records, expected = _cell()
    del frames["LSTMBbE_s2"], records["LSTMBbE_s2"]
    with pytest.raises(SystemExit, match=r"LSTMBbE has seeds \[0, 1\], expected \[0, 1, 2\]"):
        V.cell_readings(persistence, frames, records, BOUNDARY, expected, "synthetic")


def test_across_birds_medians_bands_and_interval_counts():
    def cell(bb, lstm, scored=True):
        if not scored:
            return {"scored": False}
        reading = {"skill_difference": bb, "skill_bootstrap": {"skill_difference": {"excludes_zero": bb > 0.1}}}
        other = {"skill_difference": lstm, "skill_bootstrap": {"skill_difference": {"excludes_zero": False}}}
        return {"scored": True, "readings": {
            "against_p": {arm: 0.1 for arm in V.ARMS[1:]},
            "env_effect": {"Bb": reading, "LSTMBb": other, "XGBBb": other},
            "contrasts": {name: {"difference": bb - lstm, "skill_bootstrap": {"difference": {"excludes_zero": True}}}
                          for name in V.CONTRASTS},
            "hardware": {"Bb": other, "LSTMBb": other}}}

    report = {"species": {
        "a": {"horizons": {"1": cell(0.2, 0.0), "6": cell(0.2, 0.0), "24": cell(0.2, 0.0)}},
        "b": {"horizons": {"1": cell(0.0, 0.0), "6": cell(0.0, 0.0), "24": cell(0.0, 0.0, scored=False)}},
        "c": {"horizons": {"1": cell(-0.1, 0.0), "6": cell(-0.1, 0.0), "24": cell(-0.1, 0.0)}},
    }}
    across = V.across_birds(report)
    assert across["common_set"] == ["a", "c"]
    one = across["horizons"]["1"]
    assert one["env_effect"]["Bb"]["scored"]["difference"] == {"n": 3, "median": 0.0}
    assert one["env_effect"]["Bb"]["scored"]["bands"] == {"above": 1, "within": 1, "below": 1, "not scored": 0}
    assert one["env_effect"]["Bb"]["scored"]["intervals_excluding_zero"] == 1
    assert one["env_effect"]["Bb"]["common"]["difference"] == {"n": 2, "median": pytest.approx(0.05)}
    assert one["contrasts"]["Bb_vs_LSTMBb"]["scored"]["intervals_excluding_zero"] == 3
    assert set(one["hardware"]["Bb"]["scored"]) == {"difference"}  # no bands: a difference only
    assert across["horizons"]["24"]["scored"] == ["a", "c"]


# --------------------------------------------------------------------------- records, files and guards
def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """All 150 env records and the 90 main-study records, clean, with their checkpoints, boosters and env files."""
    payload = b1h.load_registered()
    monkeypatch.setattr(env, "RESULTS", tmp_path / "env_results")
    monkeypatch.setattr(env, "ENV_DIR", tmp_path / "env")
    monkeypatch.setattr(b1h, "RESULTS", tmp_path / "main_results")
    for directory in (env.RESULTS, env.ENV_DIR, b1h.RESULTS):
        directory.mkdir()
    for bird in V.BIRDS:
        env.env_path(bird).write_bytes(f"era5 {bird}".encode())
    digests = {bird: _sha(env.env_path(bird)) for bird in V.BIRDS}

    def common(run, names):
        record = {"run_id": run.run_id, "fold": run.fold, "seed": run.seed, "horizon": run.horizon or None,
                  "context": E.CORE, "pool_species": names, "excluded_individuals": b1h.RT.excluded_in(names, payload),
                  "opened_test_partition": False}
        if run.xgb:
            for part in ("east.ubj", "north.ubj", "meta.json"):
                (results(run) / f"{run.run_id}_xgb_{part}").write_text("{}")
            return record | {"allow_missing": False, "training_windows": b1h.XGB_POOLED_DRAWS}
        (results(run) / f"{run.run_id}_reported.pt").write_bytes(b"")
        return record | {"applied_updates": b1h.REGISTERED_STEPS, "config": {"steps": b1h.REGISTERED_STEPS},
                         "skipped_steps": 0, "withheld_fraction": 0, "nonfinite_steps": [],
                         "reported_checkpoint": f"{run.run_id}_reported.pt",
                         "pool_windows": sum(b1h.TRAIN_WINDOWS[n][0] for n in names)}

    def results(run):
        return env.RESULTS if isinstance(run, env.EnvRun) else b1h.RESULTS

    for index, run in enumerate(env.PLAN):
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        record = common(run, names) | {
            "plan_index": index, "arm": run.arm, "tier": env.TAG, "env": run.env, "queue": env.queue_of(index),
            "env_channels": env.ENV_CHANNELS if run.env else 0, "env_clip_sd": era5.CLIP_SD if run.env else None,
            "env_stats": ({"channels": list(era5.ENV_CHANNELS[:14]), "mean": [0.0] * 14, "sd": [1.0] * 14}
                          if run.env else None),
            "env_files": {n: digests[n] for n in names} if run.env else None,
        }
        if not run.xgb:
            record["tf32"] = {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}
            record["device"] = "NVIDIA H200"
        (env.RESULTS / f"{run.run_id}.json").write_text(json.dumps(record))
    for run in V.MAIN_RUNS:
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        record = common(run, names) | {"plan_index": E.PLAN_INDEX[run.run_id], "arm": run.arm, "tier": "birds_1h"}
        (b1h.RESULTS / f"{run.run_id}.json").write_text(json.dumps(record))
    return payload


def _rewrite(run_id, **changes):
    path = env.RESULTS / f"{run_id}.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def test_clean_records_pass_every_guard(clean):
    records = V.read_env_records()
    assert len(records) == 150 and V.env_completeness(records)["complete"]
    main_records, gaps, exclusions, digests = V.checked_state(records, clean)
    assert len(main_records) == 90 and gaps["main"]["complete"] and len(digests) == 30
    assert exclusions["env"]["runs"] == 150 and exclusions["main"]["runs"] == 90


@pytest.mark.parametrize("run_id, changes, message", [
    ("BbE_f3_s1", {"env_clip_sd": 5.0}, "env_clip_sd 5.0, expected 10.0"),
    ("XGBBbE_f2_h6", {"env_clip_sd": None}, "env_clip_sd None, expected 10.0"),
    ("BbR_f0_s0", {"env_clip_sd": 10.0}, "env_clip_sd 10.0, expected None"),
    ("LSTMBbE_f1_s2", {"tf32": {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False}}, "tf32"),
    ("LSTMBbR_f4_s0", {"applied_updates": 19_999}, "applied 19999 of 20000"),
    ("XGBBbE_f5_h1", {"opened_test_partition": True}, "opened_test_partition"),
])
def test_an_unclean_env_run_or_another_clip_is_refused(clean, run_id, changes, message):
    _rewrite(run_id, **changes)
    with pytest.raises(SystemExit, match=f"{run_id}.*{message}"):
        V.checked_state(V.read_env_records(), clean)


def test_a_missing_env_or_main_run_is_refused(clean):
    (env.RESULTS / "LSTMBbE_f9_s2.json").unlink()
    (b1h.RESULTS / "XGBBb_f4_h24.json").unlink()
    with pytest.raises(SystemExit, match=r"(?s)LSTMBbE_f9_s2 \(index 59\): no record.*XGBBb_f4_h24.*no record"):
        V.checked_state(V.read_env_records(), clean)


def test_an_env_file_whose_sha256_differs_from_the_records_is_refused(clean):
    bird = V.BIRDS[4]
    env.env_path(bird).write_bytes(b"changed")
    with pytest.raises(SystemExit, match=f"(?s)env files differ.*{bird}: sha256"):
        V.checked_state(V.read_env_records(), clean)


def test_records_that_disagree_on_a_sha256_or_never_hold_a_bird_are_refused(clean):
    records = V.read_env_records()
    bird = next(iter(records["BbE_f0_s0"]["env_files"]))
    records["BbE_f0_s0"]["env_files"][bird] = "0" * 64
    _, problems = V.env_file_problems(records)
    assert problems == [f"{bird}: records hold 2 different sha256"]
    _, problems = V.env_file_problems({})
    assert len(problems) == 30 and all("no env record holds its sha256" in p for p in problems)


def test_a_checkpoint_at_another_clip_than_its_record_is_refused(clean, monkeypatch):
    records = V.read_env_records()
    run = env.PLAN[0]
    record = records[run.run_id] | {"horizon_k": [1.0, 2.0, 3.0]}
    state = {"horizon_k": [1.0, 2.0, 3.0], "context": E.CORE, "env_channels": 15, "env_stats": record["env_stats"],
             "env_clip_sd": 10.0}

    class Model:
        def to(self, device):
            return self

    monkeypatch.setattr(E3, "load_reported", lambda results, run_id: (Model(), dict(state)))
    assert len(V.load_env_models([run], {run.run_id: record}, "cpu")[0]) == 1
    state["env_clip_sd"] = 5.0
    with pytest.raises(SystemExit, match="env_clip_sd"):
        V.load_env_models([run], {run.run_id: record}, "cpu")


def test_the_evaluator_refuses_before_the_env_freeze_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(V, "FREEZE", tmp_path / "birds_1h_env_exploratory_freeze.md")
    monkeypatch.setattr(V, "MANIFEST", tmp_path / "manifest.json")
    for call in (lambda: V.run_predictions("cpu"), V.diagnostic):
        with pytest.raises(SystemExit, match="does not exist"):
            call()


def test_predictions_check_the_seal_before_records_or_test_data(monkeypatch):
    order = []

    def seal():
        order.append("seal")
        raise SystemExit("sealed")

    monkeypatch.setattr(V, "assert_sealed", seal)
    monkeypatch.setattr(V, "read_env_records", lambda: order.append("records"))
    with pytest.raises(SystemExit, match="sealed"):
        V.run_predictions("cpu")
    assert order == ["seal"]


def test_predictions_refuse_an_unclean_run_or_a_changed_env_file_before_any_test_read(clean, monkeypatch):
    monkeypatch.setattr(V, "assert_sealed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    _rewrite("BbE_f6_s0", env_clip_sd=None)
    with pytest.raises(SystemExit, match="BbE_f6_s0.*env_clip_sd"):
        V.run_predictions("cpu")
    _rewrite("BbE_f6_s0", env_clip_sd=era5.CLIP_SD)
    env.env_path(V.BIRDS[0]).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="env files differ"):
        V.run_predictions("cpu")
    env.env_path(V.BIRDS[0]).write_bytes(f"era5 {V.BIRDS[0]}".encode())
    with pytest.raises(AssertionError, match="test-partition reader"):  # every guard passed; scoring reads test next
        V.run_predictions("cpu")


def _committed(monkeypatch):
    monkeypatch.setattr(V, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(RS, "committed_freeze", lambda: "## 6. Predictions\n\nmine\n\n## 7. As trained\n\n150\n")


def test_the_diagnostic_passes_on_clean_records_and_never_opens_test(clean, monkeypatch, capsys):
    _committed(monkeypatch)
    V.diagnostic()  # the autouse fixture raises if any test-partition reader is reached
    out = capsys.readouterr().out
    assert "150 of 150 recorded, 150 clean" in out and "90 env runs at ±10 SD" in out
    assert "30 birds; sha256 as recorded" in out and "90 of 90 clean" in out and "TF32 off on 120" in out
    assert "test-partition reads | none" in out


def test_the_diagnostic_fails_on_another_clip_and_still_never_opens_test(clean, monkeypatch, capsys):
    _committed(monkeypatch)
    _rewrite("XGBBbE_f9_h24", env_clip_sd=8.0)
    with pytest.raises(SystemExit, match="diagnostic found a problem"):
        V.diagnostic()
    out = capsys.readouterr().out
    assert "XGBBbE_f9_h24" in out and "env_clip_sd 8.0" in out and "89 env runs at ±10 SD" in out


def test_saved_predictions_from_outside_the_fold_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path)
    bird = V.BIRDS[0]
    other = next(r for r in env.PLAN if r.fold != V.FOLD_OF[bird])
    for tag in ("P", other.run_id):
        _persistence(1, 2).to_parquet(V.prediction_path(bird, tag))
    with pytest.raises(SystemExit, match=f"outside its own fold.*{other.run_id}"):
        V.load_tables(bird)


# --------------------------------------------------------------------------- prediction wiring (synthetic test bird)
class _Net:
    """A stand-in neural model: records its inputs and predicts zero displacement."""

    def __init__(self):
        self.seen = []

    def __call__(self, x, observed=None):
        import torch

        self.seen.append(x.detach().cpu().numpy())
        return torch.zeros(x.shape[0], len(V.HORIZONS), 2)


class _Booster:
    def __init__(self):
        self.seen = []

    def predict(self, x):
        self.seen.append(x)
        return np.zeros((len(x), 2))


def _synthetic_bird(root, stem, n_bins=80):
    rng = np.random.default_rng(3)
    rows = [{"segment_id": f"c_{stem}_s0", "individual_id": f"c_{stem}", "bin_index": b,
             "timestamp": pd.Timestamp("2020-03-01", tz="UTC") + pd.Timedelta(seconds=3600 * b + int(rng.integers(0, 3000))),
             "lon": 10.0 + 0.004 * b + rng.normal(0, 3e-4), "lat": 50.0 + 0.002 * b + rng.normal(0, 3e-4)}
            for b in range(n_bins)]
    segments = pd.DataFrame(rows)
    segments.to_parquet(root / f"{stem}_segments.parquet")
    values = rng.normal(50.0, 10.0, size=(len(segments), len(era5.RAW))).astype(np.float32)
    frame = segments[["segment_id", "bin_index"]].assign(**{v: values[:, i] for i, v in enumerate(era5.RAW)})
    frame.to_parquet(env.env_path(stem), index=False)


def test_predict_bird_scores_plain_and_env_arms_on_the_same_windows(tmp_path, monkeypatch):
    bird = V.BIRDS[0]
    monkeypatch.setattr(b1h, "SEGMENTS", tmp_path)
    monkeypatch.setattr(env, "ENV_DIR", tmp_path / "env")
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path / "predictions")
    env.ENV_DIR.mkdir()
    V.PREDICTIONS.mkdir()
    _synthetic_bird(tmp_path, bird)
    monkeypatch.setattr(E, "test_windows", TEST_WINDOWS)  # this test's "test bird" is synthetic
    monkeypatch.setattr(env, "env_windows", ENV_WINDOWS)
    monkeypatch.setattr(b1h, "horizon_windows", HORIZON_WINDOWS)
    monkeypatch.setattr(E, "scored_ids", lambda stem, dropped: [f"c_{stem}"])

    pool = [b for b in V.BIRDS if b != bird]
    tight = {"channels": list(era5.ENV_CHANNELS[:14]), "mean": [50.0] * 14, "sd": [0.5] * 14}  # |z| up to ~100
    loose = tight | {"sd": [1e4] * 14}
    records, main_records, models = {}, {}, {}
    for run in V.own_env_runs(bird):
        stats = (tight if run.seed != 1 else loose) if run.env else None  # two stats groups among the env runs
        records[run.run_id] = {"pool_species": pool, "env_stats": stats}
        models[run.run_id] = _Booster() if run.xgb else _Net()
    for run in V.own_main_runs(bird):
        main_records[run.run_id] = {"pool_species": pool}
        models[run.run_id] = _Booster() if run.xgb else _Net()

    def fake_load(runs, *args):
        k = np.ones(len(V.HORIZONS))
        return ([(r, models[r.run_id], k) for r in runs if not r.xgb], [(r, models[r.run_id], k) for r in runs if r.xgb])

    monkeypatch.setattr(E, "load_models", lambda runs, recs, device, context: fake_load(runs))
    monkeypatch.setattr(V, "load_env_models", lambda runs, recs, device: fake_load(runs))
    out = V.predict_bird(bird, [], records, main_records, "cpu", overwrite=False)

    expected = {"P", *records, *main_records}
    assert len(expected) == 25 and {Path(f).stem.split("__", 1)[1] for f in out["files"]} == expected
    tables = V.load_tables(bird)
    for tag, table in tables.items():
        for horizon in V.HORIZONS if not tag.startswith("XGB") else (int(tag.rsplit("_h", 1)[1]),):
            E.assert_aligned(E.at(table, horizon), E.at(tables["P"], horizon), tag)

    plain = next(m for t, m in models.items() if t.startswith("Bb_")).seen[0]
    assert plain.shape[-1] == 12
    for run in V.own_env_runs(bird):
        x = np.concatenate(models[run.run_id].seen) if not run.xgb else models[run.run_id].seen[0]
        if not run.env:
            assert x.shape[-1] == 12 and np.array_equal(x, plain[: len(x)]) if not run.xgb else x.shape[-1] == 12
            continue
        assert x.shape[-1] == 27
        z = x[..., 12:26]
        if records[run.run_id]["env_stats"] is tight:
            assert np.abs(z).max() == era5.CLIP_SD  # standardised with this record's stats, then clipped
        else:
            assert np.abs(z).max() < 1.0
        assert (x[..., 26] == 1).all()
        if not run.xgb:
            np.testing.assert_array_equal(x[..., :12], plain)  # the movement channels are the plain windows'

    # a second call finds every file present and scores nothing
    for model in models.values():
        model.seen.clear()
    V.predict_bird(bird, [], records, main_records, "cpu", overwrite=False)
    assert not any(model.seen for model in models.values())


def test_predict_bird_refuses_a_model_whose_pool_held_the_bird():
    bird = V.BIRDS[0]
    run = V.own_env_runs(bird)[0]
    records = {r.run_id: {"pool_species": [bird] if r == run else []} for r in V.own_env_runs(bird)}
    with pytest.raises(SystemExit, match=f"{run.run_id}'s pool held {bird}"):
        V.predict_bird(bird, [], records, {}, "cpu", overwrite=False)


# --------------------------------------------------------------------------- E6.1: FP32 at inference and the 1 h tie-back
def test_main_switches_tf32_off_before_anything_runs(monkeypatch):
    import sys

    import torch

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    seen = {}
    monkeypatch.setattr(V, "diagnostic", lambda: seen.update(env.tf32_flags()))
    monkeypatch.setattr(sys, "argv", ["bird_1h_env_sealed_eval.py", "--diagnostic"])
    V.main()
    assert seen == {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}


def test_tf32_that_cannot_be_switched_off_is_refused(monkeypatch):
    monkeypatch.setattr(env, "disable_tf32", lambda: None)
    monkeypatch.setattr(env, "tf32_flags", lambda: {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False})
    with pytest.raises(SystemExit, match="TF32 is on"):
        V.fp32()


def _one_h_cell(skills, scored=True):
    if not scored:
        return {"scored": False}
    return {"scored": True, "arms": {arm: {"skill": {"travel": {"p90_m": v}}} for arm, v in skills.items()}}


def _env_cell(skills, scored=True):
    return {"scored": scored, "readings": {"against_p": skills}} if scored else {"scored": False}


def test_tie_back_reports_differences_per_arm_without_a_verdict():
    ours = {"species": {
        "a": {"horizons": {"1": _env_cell({"Bb": 0.10, "LSTMBb": 0.20, "XGBBb": 0.30}),
                           "6": _env_cell({"Bb": 0.05, "LSTMBb": 0.00, "XGBBb": -0.10}),
                           "24": _env_cell({}, scored=False)}},
        "b": {"horizons": {"1": _env_cell({"Bb": 0.0, "LSTMBb": 0.0, "XGBBb": 0.0}),
                           "6": _env_cell({"Bb": 0.0, "LSTMBb": 0.0, "XGBBb": 0.0}),
                           "24": _env_cell({"Bb": 0.0, "LSTMBb": 0.0, "XGBBb": 0.0})}},
    }}
    theirs = {"attestation": {"freeze_commit": "c"}, "species": {
        "a": {"horizons": {"1": _one_h_cell({"Bb": 0.10, "LSTMBb": 0.21, "XGBBb": 0.30, "LSTMS": 9.0}),
                           "6": _one_h_cell({"Bb": 0.04, "LSTMBb": 0.00, "XGBBb": -0.10}),
                           "24": _one_h_cell({}, scored=False)}},
        "b": {"horizons": {"1": _one_h_cell({"Bb": 0.0, "LSTMBb": 0.0, "XGBBb": 0.0}),
                           "6": _one_h_cell({"Bb": 0.0, "LSTMBb": 0.0, "XGBBb": 0.0}),
                           "24": _one_h_cell({}, scored=False)}},
    }}
    tie = V.tie_back(ours, theirs)
    assert tie["cells"]["a"]["1"] == pytest.approx({"Bb": 0.0, "LSTMBb": -0.01, "XGBBb": 0.0})
    assert tie["cells"]["a"]["6"]["Bb"] == pytest.approx(0.01)
    assert tie["arms"]["Bb"] == {"n": 4, "max_abs_diff": pytest.approx(0.01), "median_diff": pytest.approx(0.0)}
    assert tie["arms"]["LSTMBb"]["max_abs_diff"] == pytest.approx(0.01)
    assert tie["scored_differs"] == ["b 24 h"] and "24" not in tie["cells"]["b"]
    assert tie["attestation"] == {"freeze_commit": "c"} and tie["source"].endswith("birds_1h_results.json")
    assert not any(key in json.dumps(tie) for key in ("pass", "fail"))


def test_a_missing_or_unattested_1h_report_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "ONE_H_REPORT", tmp_path / "birds_1h_results.json")
    monkeypatch.setattr(V, "MANIFEST", tmp_path / "manifest.json")
    V.MANIFEST.write_text("{}")
    monkeypatch.setattr(V, "assert_sealed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(V, "read_env_records", _refuse)
    for call in (lambda: V.run_predictions("cpu"), V.build_report):
        with pytest.raises(SystemExit, match="does not exist; the tie-back"):
            call()
    V.ONE_H_REPORT.write_text(json.dumps({"study": "birds_1h", "attestation": None}))
    with pytest.raises(SystemExit, match="not an attested 1 h sealed report"):
        V.build_report()


def test_the_markdown_renders_every_section_from_a_synthetic_report(tmp_path):
    persistence, frames, records, expected = _cell(noise=0.3)
    arms, readings = V.cell_readings(persistence, frames, records, BOUNDARY, expected, "synthetic")
    cell = {"scored": True, "at_floor": 6, "individuals": 6, "share_in_travel": 1.0, "arms": arms, "readings": readings}
    birds = V.BIRDS[:2]
    report = {
        "registered_by": "preregistration/birds_1h_env_exploratory_freeze.md",
        "attestation": {"freeze_commit": "a" * 40, "freeze_blob": "b" * 40},
        "evaluation_floor": 150, "min_scored_individuals": 5, "env_clip_sd": 10.0,
        "tf32": {"predict": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}, "report": {}},
        "effects": {f: f"{a} - {b}" for f, (a, b) in V.EFFECTS.items()},
        "contrasts": {n: n for n in V.CONTRASTS}, "hardware": {f: f"{a} - {b}" for f, (a, b) in V.HARDWARE.items()},
        "completeness": {"env": {"n_records": 150, "n_planned": 150, "n_clean": 150},
                         "main": {"n_planned": 90, "n_clean": 90}},
        "species": {bird: {"fold": V.FOLD_OF[bird], "horizons": {
            "1": cell, "6": cell, "24": {"scored": False, "at_floor": 2, "individuals": 3}}} for bird in birds},
    }
    report["across_birds"] = V.across_birds(report)
    one_h = {"attestation": {"freeze_commit": "c"}, "species": {bird: {"horizons": {
        h: ({"scored": True, "arms": {a: arms[a] for a in V.MAIN_ARMS}} if h != "24" else {"scored": False})
        for h in ("1", "6", "24")}} for bird in birds}}
    report["tie_back"] = V.tie_back(report, one_h)
    assert report["tie_back"]["arms"]["Bb"] == {"n": 4, "max_abs_diff": 0.0, "median_diff": 0.0}
    path = tmp_path / "report.md"
    V.write_markdown(report, path)
    text = path.read_text()
    for heading in ("skill against P", "env effect and contrasts", "Across birds", "Hardware check",
                    "Tie-back to the 1 h sealed report"):
        assert heading in text
    assert "TF32 at prediction" in text and text.count("unscored") == 2
