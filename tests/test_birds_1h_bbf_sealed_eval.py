# GenAI reference: I outlined tests for our repaired transformer study's sealed evaluator (comparison and contrast arithmetic, intervals from 
# shared resamples, the fold 6 sets, etc; and GenAI wrote them
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_1h_bbf_sealed_eval as W
from animal_movement_fm.data import era5

E, V, F, env, TR, b1h, RS, S2, E3 = W.E, W.V, W.F, W.env, W.TR, W.b1h, W.RS, W.S2, W.E3
TEST_WINDOWS, ENV_WINDOWS, HORIZON_WINDOWS = E.test_windows, env.env_windows, b1h.horizon_windows  # before the fixture
T0 = pd.Timestamp("2021-01-01", tz="UTC")
BOUNDARY = 1.0  # every persistence error is above it, so every window is travel
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
DEFAULT_SWITCHES = {"input_scale": "sqrt_d", "norm_first": False, "step_transform": "none"}


def _refuse(*args, **kwargs):
    raise AssertionError("a test reached a test-partition reader")


@pytest.fixture(autouse=True)
def _no_test_partition(monkeypatch):
    """Every reader of test-bird fixes, ids or windows raises, so no test here can open a test partition."""
    for owner, attr in ((E, "test_windows"), (E, "scored_ids"), (E, "scoring_state"), (RS, "load_corpus"),
                        (RS, "test_frame"), (env, "env_windows"), (b1h, "horizon_windows")):
        monkeypatch.setattr(owner, attr, _refuse)


# --------------------------------------------------------------------------- layout
def test_arms_comparisons_and_the_fallback_layout():
    assert W.labels_of(False) == tuple(W.ARMS) and len(W.ARMS) == 13
    assert W.labels_of(True) == tuple(label for label in W.ARMS if label != "LSTMBbR_C96")
    assert list(W.comparisons_of(False)) == [
        "BbF - LSTMBbR", "BbF - LSTMBb", "BbF - XGBBb", "BbF - BbR", "BbF - Bb", "BbFE - BbF",
        "BbF_C96 - LSTMBbR_C96", "BbF_C96 - LSTMBb_C96", "BbF_C96 - XGBBb_C96", "BbF_C96 - Bb_C96"]
    assert "BbF_C96 - LSTMBbR_C96" not in W.comparisons_of(True) and "BbF_C96 - LSTMBb_C96" in W.comparisons_of(True)
    assert list(W.hardware_of(True)) == ["LSTMBbR - LSTMBb", "BbR - Bb"]
    assert len(W.planned_runs(False)) == 120 and len(W.planned_runs(True)) == 90
    assert W.SELECTION_FOLD == 6 and W.FALLBACK_FROM == dt.date(2026, 10, 25)
    assert W.DEFAULTS == DEFAULT_SWITCHES
    assert W.FREEZE.name == "birds_1h_bbf_exploratory_freeze.md" and W.SEALED.name == "birds_1h_bbf_sealed"


def test_tags_reach_each_studys_own_fold_runs():
    bird = next(b for b in W.BIRDS if W.FOLD_OF[b] == 3)
    assert W.tags_at(bird, "BbF", 6) == [("BbF_f3_s0", 0), ("BbF_f3_s1", 1), ("BbF_f3_s2", 2)]
    assert W.tags_at(bird, "BbF_C96", 1) == [(f"BbF_f3_s{s}_C96", s) for s in (0, 1, 2)]
    assert W.tags_at(bird, "LSTMBbR_C96", 1) == [(f"LSTMBbR_f3_s{s}_C96", s) for s in (0, 1, 2)]
    assert W.tags_at(bird, "LSTMBbR", 24) == [(f"LSTMBbR_f3_s{s}", s) for s in (0, 1, 2)]
    assert W.tags_at(bird, "XGBBb", 6) == [("XGBBb_f3_h6", 0)]
    assert W.tags_at(bird, "XGBBb_C96", 24) == [("XGBBb_f3_h24_C96", 0)]
    assert W.tags_at(bird, "Bb_C96", 6) == [(f"Bb_f3_s{s}_C96", s) for s in (0, 1, 2)]
    reused = W.reused_files(bird, False)
    assert reused[:3] == [("1h", "P"), ("1h", "P_C96"), ("env", "P")]
    assert ("1h", "XGBBb_f3_h1_C96") in reused and ("env", "LSTMBbE_f3_s2") in reused
    assert not any(study == "bbf" for study, _ in reused) and reused == W.reused_files(bird, True)


# --------------------------------------------------------------------------- statistic
SCALES = {"BbF": 0.80, "BbFE": 0.74, "BbR": 0.99, "LSTMBbR": 0.88, "LSTMBbE": 0.87, "Bb": 0.98, "LSTMBb": 0.89,
          "XGBBb": 0.83, "BbF_C96": 0.70, "LSTMBbR_C96": 0.84, "Bb_C96": 1.00, "LSTMBb_C96": 0.85, "XGBBb_C96": 0.72}


def _persistence(n_individuals=6, n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"individual_id": f"i{k}", "timestamp": T0 + pd.Timedelta(hours=i), "target_lon": 0.1 * i,
             "target_lat": 1.0} for k in range(n_individuals) for i in range(n)]
    frame = pd.DataFrame(rows)
    frame["error_m"] = rng.gamma(2.0, 500.0, size=len(frame)) + 200.0
    return frame


def _cell(scales=SCALES, noise=0.0):
    """P and every label's frames on the same windows; neural labels at three seeds, the median seed at the label's scale."""
    persistence = _persistence()
    rng = np.random.default_rng(7)
    frames, parsed, expected = {"P": persistence}, {}, {}
    for label, scale in scales.items():
        seeds = (0,) if label.startswith("XGB") else (0, 1, 2)
        expected[label] = set(seeds)
        for seed in seeds:
            tag = f"{label}_tag_s{seed}"
            factor = scale * (1 + 0.01 * (seed - 1) * (len(seeds) > 1))
            jitter = 1 + noise * rng.normal(size=len(persistence))
            frames[tag] = persistence.assign(error_m=persistence["error_m"] * factor * jitter)
            parsed[tag] = {"arm": label, "seed": seed}
    return persistence, frames, parsed, expected


def test_comparisons_contrast_and_hardware_have_the_registered_arithmetic_and_sign():
    persistence, frames, parsed, expected = _cell()
    _, readings = W.cell_readings(persistence, frames, parsed, BOUNDARY, expected, "synthetic", False)
    skill = {label: 1 - scale for label, scale in SCALES.items()}
    assert readings["against_p"] == pytest.approx(skill)
    for key, (a, b) in W.comparisons_of(False).items():
        reading = readings["comparisons"][key]
        assert (reading["left"], reading["right"]) == (a, b)
        assert reading["skill_difference"] == pytest.approx(skill[a] - skill[b])
        boot = reading["skill_bootstrap"]["skill_difference"]
        assert boot["lo"] == pytest.approx(skill[a] - skill[b]) == boot["hi"]
    c = readings["comparisons"]
    assert c["BbF - BbR"]["skill_difference"] == pytest.approx(0.19)  # the repair, C = 12
    assert c["BbF - LSTMBbR"]["skill_difference"] == pytest.approx(0.08)
    assert c["BbF_C96 - Bb_C96"]["skill_difference"] == pytest.approx(0.30)
    assert c["BbF_C96 - XGBBb_C96"]["skill_difference"] == pytest.approx(0.02)
    assert [RS.band(c[k]["skill_difference"]) for k in ("BbF - BbR", "BbF_C96 - XGBBb_C96", "BbF - XGBBb")] == [
        "above", "within", "within"]
    # (BbFE - BbF) - (LSTMBbE - LSTMBbR) = 0.06 - 0.01
    contrast = readings["contrasts"]["(BbFE - BbF) - (LSTMBbE - LSTMBbR)"]
    assert contrast["difference"] == pytest.approx(0.05)
    assert contrast["label"] == "(BbFE - BbF) - (LSTMBbE - LSTMBbR)"
    boot = contrast["skill_bootstrap"]["difference"]
    assert boot["lo"] == pytest.approx(0.05) and boot["hi"] == pytest.approx(0.05) and boot["excludes_zero"]
    hardware = readings["hardware"]
    assert hardware["LSTMBbR_C96 - LSTMBb_C96"]["skill_difference"] == pytest.approx(0.01)
    assert hardware["BbR - Bb"]["skill_difference"] == pytest.approx(-0.01)


def test_the_fallback_cell_has_no_lstmbbr96():
    scales = {k: v for k, v in SCALES.items() if k != "LSTMBbR_C96"}
    persistence, frames, parsed, expected = _cell(scales)
    _, readings = W.cell_readings(persistence, frames, parsed, BOUNDARY, expected, "synthetic", True)
    assert "LSTMBbR_C96" not in readings["against_p"] and "BbF_C96 - LSTMBbR_C96" not in readings["comparisons"]
    assert readings["comparisons"]["BbF_C96 - LSTMBb_C96"]["skill_difference"] == pytest.approx(0.15)
    with pytest.raises(SystemExit, match="LSTMBbR_C96"):  # without the fallback the label is required
        W.cell_readings(persistence, frames, parsed, BOUNDARY, expected | {"LSTMBbR_C96": {0, 1, 2}}, "synthetic",
                        False)


def test_contrast_intervals_come_from_the_same_resamples_as_the_comparisons():
    persistence, frames, parsed, expected = _cell(noise=0.3)
    with W.bound():
        arms = RS.arm_blocks(E3.summarise_all(frames, persistence, BOUNDARY, parsed, parse=E.parse_tag))
        resampler = RS.Resampler(persistence, BOUNDARY)
        for pair in (("BbFE", "BbF"), ("LSTMBbE", "LSTMBbR")):
            RS.compare(*pair, arms, frames, persistence, BOUNDARY, resampler)
        cached = dict(resampler._cache)
        result = W.contrast(("BbFE", "BbF"), ("LSTMBbE", "LSTMBbR"), arms, frames, resampler)
    assert resampler._cache.keys() == cached.keys() and all(resampler._cache[k] is cached[k] for k in cached)
    s = {arm: 1 - cached[arm] / resampler.reference for arm in ("BbFE", "BbF", "LSTMBbE", "LSTMBbR")}
    values = (s["BbFE"] - s["BbF"]) - (s["LSTMBbE"] - s["LSTMBbR"])
    boot = result["skill_bootstrap"]
    assert (boot["n_resamples"], boot["seed"]) == (2000, 42)
    assert boot["difference"]["lo"] == np.nanpercentile(values, 2.5)
    assert boot["difference"]["hi"] == np.nanpercentile(values, 97.5)


def test_a_label_missing_a_seed_is_refused():
    persistence, frames, parsed, expected = _cell()
    del frames["BbF_C96_tag_s2"], parsed["BbF_C96_tag_s2"]
    with pytest.raises(SystemExit, match=r"BbF_C96 has seeds \[0, 1\], expected \[0, 1, 2\]"):
        W.cell_readings(persistence, frames, parsed, BOUNDARY, expected, "synthetic", False)


def _reading_cell(diff, scored=True):
    if not scored:
        return {"scored": False}
    reading = {"skill_difference": diff, "skill_bootstrap": {"skill_difference": {"excludes_zero": diff > 0.1}}}
    return {"scored": True, "readings": {
        "against_p": {label: diff for label in W.ARMS},
        "comparisons": {key: reading for key in W.comparisons_of(False)},
        "contrasts": {key: {"difference": diff, "skill_bootstrap": {"difference": {"excludes_zero": False}}}
                      for key in W.CONTRASTS},
        "hardware": {key: reading for key in W.hardware_of(False)}}}


def test_across_birds_repeats_the_medians_without_the_selection_folds_birds():
    six = [b for b in W.BIRDS if W.FOLD_OF[b] == 6][:2]
    others = [b for b in W.BIRDS if W.FOLD_OF[b] != 6][:2]
    values = {six[0]: 0.4, six[1]: 0.3, others[0]: 0.0, others[1]: 0.1}
    report = {"species": {b: {"horizons": {h: _reading_cell(v, scored=not (b == others[1] and h == "24"))
                                           for h in ("1", "6", "24")}} for b, v in values.items()}}
    across = W.across_birds(report, False)
    assert across["common_set"] == [six[0], six[1], others[0]] and across["selection_fold"] == 6
    one = across["horizons"]["1"]["comparisons"]["BbF - BbR"]
    assert one["scored"]["difference"] == {"n": 4, "median": pytest.approx(0.2)}
    assert one["scored_without_fold6"]["difference"] == {"n": 2, "median": pytest.approx(0.05)}
    assert one["common_without_fold6"]["difference"] == {"n": 1, "median": pytest.approx(0.0)}
    assert one["scored"]["bands"] == {"above": 3, "within": 1, "below": 0, "not scored": 0}
    assert one["scored"]["intervals_excluding_zero"] == 2
    skill = across["horizons"]["6"]["skill"]["BbF_C96"]
    assert set(skill) == set(W.SETS) and skill["common"] == {"n": 3, "median": pytest.approx(0.3)}
    assert set(across["horizons"]["1"]["hardware"]["BbR - Bb"]["scored"]) == {"difference"}


def test_tie_back_reports_reused_arms_against_their_reports_without_a_verdict():
    bird = W.BIRDS[0]
    against = {label: 0.1 for label in W.ARMS} | {"LSTMBb_C96": 0.30, "LSTMBbE": 0.22}
    ours = {"species": {bird: {"horizons": {"6": {"scored": True, "readings": {"against_p": against}},
                                            "24": {"scored": False}}}}}
    one_h_arms = {arm: {"skill": {"travel": {"p90_m": 0.1}}} for arm in ("Bb", "LSTMBb", "XGBBb")}
    ablation = {"96": {"all": {arm: {"skill": 0.1} for arm in ("Bb", "XGBBb")} | {"LSTMBb": {"skill": 0.28}}}}
    sources = {
        "birds_1h": {"attestation": {"freeze_commit": "a"}, "species": {bird: {"horizons": {
            "6": {"scored": True, "arms": one_h_arms, "ablation": ablation}, "24": {"scored": True}}}}},
        "birds_1h_env": {"attestation": {"freeze_commit": "b"}, "species": {bird: {"horizons": {
            "6": {"scored": True, "readings": {"against_p": {"BbR": 0.1, "LSTMBbR": 0.1, "LSTMBbE": 0.2}}},
            "24": {"scored": False}}}}},
    }
    tie = W.tie_back(ours, sources, False)
    assert tie["cells"][bird]["6"] == pytest.approx({label: 0.0 for label in W.TIED_BACK} | {
        "LSTMBb_C96": 0.02, "LSTMBbE": 0.02})
    assert tie["arms"]["LSTMBb_C96"] == {"n": 1, "max_abs_diff": pytest.approx(0.02), "median_diff": pytest.approx(0.02)}
    assert tie["scored_differs"] == [f"{bird} 24 h (birds_1h)"]
    assert set(tie["arms"]) == set(W.TIED_BACK) and not any(k in json.dumps(tie) for k in ("pass", "fail"))


# --------------------------------------------------------------------------- the fallback
def test_the_fallback_is_refused_before_25_10_and_when_lstmbbr96_is_complete(monkeypatch):
    complete = {"complete": True, "n_clean": 30, "n_planned": 30}
    partial = {"complete": False, "n_clean": 12, "n_planned": 30}
    monkeypatch.setattr(W, "completeness", lambda records, runs: partial)
    assert W.fallback_guard(False, {}) == {"used": False, "lstmbbr96_clean": 12, "lstmbbr96_planned": 30}
    monkeypatch.setattr(W, "today", lambda: dt.date(2026, 10, 24))
    with pytest.raises(SystemExit, match="only from 25-10-2026"):
        W.fallback_guard(True, {})
    monkeypatch.setattr(W, "today", lambda: dt.date(2026, 10, 25))
    used = W.fallback_guard(True, {})
    assert used["used"] is True and "V100" in used["reason"] and used["lstmbbr96_clean"] == 12
    monkeypatch.setattr(W, "completeness", lambda records, runs: complete)
    with pytest.raises(SystemExit, match="fallback is not needed"):
        W.fallback_guard(True, {})


# --------------------------------------------------------------------------- records, twins and guards
def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def clean(tmp_path, monkeypatch):
    """All 120 records, clean, with checkpoints; the BbE twins' records; env files; the probe records."""
    payload = b1h.load_registered()
    for owner, attr in ((F, "RESULTS"), (env, "RESULTS"), (env, "ENV_DIR"), (TR, "RESULTS")):
        monkeypatch.setattr(owner, attr, tmp_path / f"{owner.__name__}_{attr}")
        getattr(owner, attr).mkdir()
    for bird in W.BIRDS:
        env.env_path(bird).write_bytes(f"era5 {bird}".encode())
    digests = {bird: _sha(env.env_path(bird)) for bird in W.BIRDS}
    history = [float(v) for v in np.linspace(20.0, 17.0, 40)]
    for index, run in enumerate(F.PLAN):
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        stats = ({"channels": list(era5.ENV_CHANNELS[:14]), "mean": [float(run.fold)] * 14, "sd": [1.0] * 14}
                 if run.env else None)
        files = {n: digests[n] for n in names} if run.env else None
        record = {
            "run_id": run.run_id, "plan_index": index, "arm": run.arm, "fold": run.fold, "seed": run.seed,
            "context": run.context, "switches": run.switches, "env": run.env,
            "env_channels": 15 if run.env else 0, "env_clip_sd": era5.CLIP_SD if run.env else None,
            "env_stats": stats, "env_files": files, "queue": F.queue_of(index), "tier": F.TAG,
            "opened_test_partition": False, "pool_species": names,
            "excluded_individuals": b1h.RT.excluded_in(names, payload),
            "pool_windows": sum(b1h.TRAIN_WINDOWS[n][0] for n in names),
            "config": {"steps": 20_000, "env_channels": 15 if run.env else 0, **DEFAULT_SWITCHES, **run.switches},
            "tf32": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}, "device": "NVIDIA H200",
            "pbs_queue": "gpuhopper-exec", "applied_updates": 20_000, "skipped_steps": 0, "withheld_fraction": 0,
            "nonfinite_steps": [], "reported_checkpoint": f"{run.run_id}_reported.pt",
            "loss_history": history, "final_loss": 17.0,
        }
        (F.RESULTS / f"{run.run_id}.json").write_text(json.dumps(record))
        (F.RESULTS / f"{run.run_id}_reported.pt").write_bytes(b"")
        if run.env:
            twin = {"run_id": run.env_reference, "env_clip_sd": era5.CLIP_SD, "env_channels": 15,
                    "env_stats": stats, "env_files": files}
            (env.RESULTS / f"{run.env_reference}.json").write_text(json.dumps(twin))
        if run.repeats:
            (TR.RESULTS / f"{run.repeats}.json").write_text(json.dumps({"loss_history": history, "final_loss": 17.0}))
    return payload


def _rewrite(run_id, **changes):
    path = F.RESULTS / f"{run_id}.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def test_clean_records_pass_every_guard(clean):
    records = W.read_records()
    assert len(records) == 120
    gap, exclusions, digests = W.checked_state(records, clean, False)
    assert gap["complete"] and gap["n_clean"] == 120 and exclusions["runs"] == 120 and len(digests) == 30
    assert set(gap["by_arm"]) == {"BbF", "BbF_C96", "BbFE", "LSTMBbR_C96"}
    repeats = W.repeat_checks(records)
    assert set(repeats) == {"BbF_f6_s0", "BbF_f6_s0_C96", "LSTMBbR_f6_s0_C96"}
    assert all(c["identical"] and c["max_abs_diff"] == 0 for c in repeats.values())


@pytest.mark.parametrize("run_id, changes, message", [
    ("BbF_f3_s1", {"switches": {}}, "switches {}, expected"),
    ("BbF_f3_s1_C96", {"config": {"steps": 20_000, "env_channels": 0, **DEFAULT_SWITCHES}}, "config switches"),
    ("LSTMBbR_f2_s0_C96", {"config": {"steps": 20_000, "env_channels": 0, **DEFAULT_SWITCHES, "norm_first": True}},
     "config switches"),
    ("BbFE_f1_s2", {"env_clip_sd": 5.0}, "env_clip_sd 5.0, expected 10.0"),
    ("BbFE_f1_s2", {"config": {"steps": 20_000, "env_channels": 0, **V4}}, "config env_channels 0"),
    ("BbF_f4_s0", {"tf32": {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False}}, "tf32"),
    ("BbF_f4_s0_C96", {"applied_updates": 19_999}, "applied 19999 of 20000"),
    ("BbFE_f5_s1", {"opened_test_partition": True}, "opened_test_partition"),
])
def test_an_unclean_run_is_refused(clean, run_id, changes, message):
    _rewrite(run_id, **changes)
    with pytest.raises(SystemExit, match=f"{run_id}.*{message}"):
        W.checked_state(W.read_records(), clean, False)


def test_bbfe_off_its_twins_env_inputs_is_refused(clean):
    path = env.RESULTS / "BbE_f2_s1.json"
    twin = json.loads(path.read_text())
    path.write_text(json.dumps(twin | {"env_stats": twin["env_stats"] | {"sd": [2.0] * 14}}))
    with pytest.raises(SystemExit, match="(?s)env inputs differ.*BbFE_f2_s1: env_stats differ from BbE_f2_s1"):
        W.checked_state(W.read_records(), clean, False)


def test_an_env_file_changed_on_disk_is_refused(clean):
    env.env_path(W.BIRDS[7]).write_bytes(b"changed")
    with pytest.raises(SystemExit, match=f"(?s)env inputs differ.*{W.BIRDS[7]}: sha256"):
        W.checked_state(W.read_records(), clean, False)


def test_missing_lstmbbr96_is_refused_unless_the_fallback_drops_it(clean):
    for run in F.LSTMBBR96[5:]:
        (F.RESULTS / f"{run.run_id}.json").unlink()
    with pytest.raises(SystemExit, match=r"LSTMBbR_f1_s2_C96 \(index 95\): no record"):
        W.checked_state(W.read_records(), clean, False)
    gap, _, _ = W.checked_state(W.read_records(), clean, True)
    assert gap["n_planned"] == 90 and gap["complete"]


def test_the_evaluator_refuses_before_its_freeze_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "FREEZE", tmp_path / "birds_1h_bbf_exploratory_freeze.md")
    monkeypatch.setattr(W, "MANIFEST", tmp_path / "manifest.json")
    for call in (lambda: W.run_predictions("cpu"), W.diagnostic):
        with pytest.raises(SystemExit, match="does not exist"):
            call()


def test_predictions_check_the_seal_before_anything_else(monkeypatch):
    order = []

    def seal():
        order.append("seal")
        raise SystemExit("sealed")

    monkeypatch.setattr(W, "assert_sealed", seal)
    for name in ("source_reports", "read_records"):
        monkeypatch.setattr(W, name, lambda name=name: order.append(name))
    with pytest.raises(SystemExit, match="sealed"):
        W.run_predictions("cpu")
    assert order == ["seal"]


def test_missing_or_unattested_source_reports_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "REPORT_JSON", tmp_path / "birds_1h_results.json")
    monkeypatch.setattr(V, "REPORT_JSON", tmp_path / "birds_1h_env_results.json")
    with pytest.raises(SystemExit, match="birds_1h_results.json does not exist"):
        W.source_reports()
    E.REPORT_JSON.write_text(json.dumps({"study": "birds_1h", "attestation": {"freeze_commit": "a"}}))
    with pytest.raises(SystemExit, match="birds_1h_env_results.json does not exist"):
        W.source_reports()
    V.REPORT_JSON.write_text(json.dumps({"study": "birds_1h_env", "attestation": None}))
    with pytest.raises(SystemExit, match="not an attested birds_1h_env sealed report"):
        W.source_reports()


def test_predictions_refuse_missing_reused_files_before_any_test_read(clean, monkeypatch, tmp_path):
    monkeypatch.setattr(W, "assert_sealed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(W, "source_reports", lambda: {})
    monkeypatch.setattr(E, "PREDICTIONS", tmp_path / "one_h")
    monkeypatch.setattr(V, "PREDICTIONS", tmp_path / "env_pred")
    with pytest.raises(SystemExit, match="reused prediction files are missing"):
        W.run_predictions("cpu")
    for bird in W.BIRDS:
        for study, tag in W.reused_files(bird, False):
            path = W.prediction_path(bird, tag, study)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
    with pytest.raises(AssertionError, match="test-partition reader"):  # every guard passed; scoring reads test next
        W.run_predictions("cpu")


def _committed(monkeypatch):
    monkeypatch.setattr(W, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(RS, "committed_freeze", lambda: "## 6. Expectations\n\nmine\n\n## 7. As trained\n\nTo fill\n")


def test_the_diagnostic_passes_on_clean_records_and_never_opens_test(clean, monkeypatch, capsys):
    _committed(monkeypatch)
    W.diagnostic()  # the autouse fixture raises if any test-partition reader is reached
    out = capsys.readouterr().out
    assert "sec 7            : " in out and "not enforced here" in out
    assert "120 of 120 recorded, 120 clean" in out and "TF32 off on 120" in out
    assert "30 runs at ±10 SD; env_stats and env sha256 against the BbE twins: equal" in out
    assert "BbF_f6_s0 vs BbV4_f6_s0: identical" in out and "LSTMBbR_f6_s0_C96 vs LSTMBbV0_f6_s0_C96: identical" in out
    assert "C = 96 fallback | not used; LSTMBbR96 30 of 30 clean" in out
    assert "test-partition reads | none" in out


def test_the_diagnostic_fails_on_a_wrong_switch_and_reports_a_differing_repeat(clean, monkeypatch, capsys):
    _committed(monkeypatch)
    _rewrite("BbF_f6_s0", loss_history=[float(v) for v in np.linspace(20.0, 17.5, 40)])
    _rewrite("BbF_f9_s2", switches={"norm_first": True})
    with pytest.raises(SystemExit, match="diagnostic found a problem"):
        W.diagnostic()
    out = capsys.readouterr().out
    assert "BbF_f9_s2" in out and "BbF_f6_s0 vs BbV4_f6_s0: max |Δ| 0.5 over 40 points" in out


# --------------------------------------------------------------------------- saved predictions
def _frame(n=4):
    return _persistence(1, n).assign(pred_lon=0.0, pred_lat=0.0, horizon=6)


def _manifest_for(bird, fallback=False):
    return {"reused": {bird: {f"{s}:{t}": W.sha256(W.prediction_path(bird, t, s))
                              for s, t in W.reused_files(bird, fallback)}}}


def _write_saved(tmp_path, monkeypatch, bird):
    for owner, attr in ((W, "PREDICTIONS"), (E, "PREDICTIONS"), (V, "PREDICTIONS")):
        monkeypatch.setattr(owner, attr, tmp_path / f"{owner.__name__}_pred")
        getattr(owner, attr).mkdir()
    for tag in ("P", "P_C96", *(r.run_id for r in W.own_runs(bird, F.PLAN))):
        _frame().to_parquet(W.prediction_path(bird, tag))
    for study, tag in W.reused_files(bird, False):
        _frame().to_parquet(W.prediction_path(bird, tag, study))


def test_saved_predictions_load_with_the_reused_files_and_their_sha256(tmp_path, monkeypatch):
    bird = W.BIRDS[0]
    _write_saved(tmp_path, monkeypatch, bird)
    tables = W.load_tables(bird, _manifest_for(bird), False)
    assert {"P", "P_C96", "BbF_f0_s0", "LSTMBbE_f0_s2", "XGBBb_f0_h24_C96"} <= set(tables)
    manifest = _manifest_for(bird)
    _frame(5).to_parquet(W.prediction_path(bird, "LSTMBbR_f0_s1", "env"))
    with pytest.raises(SystemExit, match="LSTMBbR_f0_s1.parquet is missing or differs from the sha256"):
        W.load_tables(bird, manifest, False)


def test_a_reused_p_that_differs_from_this_studys_is_refused(tmp_path, monkeypatch):
    bird = W.BIRDS[0]
    _write_saved(tmp_path, monkeypatch, bird)
    _frame().assign(error_m=1.0).to_parquet(W.prediction_path(bird, "P", "env"))
    with pytest.raises(SystemExit, match="persistence"):
        W.load_tables(bird, _manifest_for(bird), False)


def test_saved_predictions_from_outside_the_fold_are_refused(tmp_path, monkeypatch):
    bird = W.BIRDS[0]
    _write_saved(tmp_path, monkeypatch, bird)
    other = next(r for r in F.PLAN if r.fold != W.FOLD_OF[bird])
    _frame().to_parquet(W.prediction_path(bird, other.run_id))
    with pytest.raises(SystemExit, match=f"outside its own fold.*{other.run_id}"):
        W.load_tables(bird, _manifest_for(bird), False)


def test_the_report_refuses_a_fallback_that_differs_from_predict(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "source_reports", lambda: {})
    monkeypatch.setattr(W, "MANIFEST", tmp_path / "manifest.json")
    W.MANIFEST.write_text(json.dumps({"c96_fallback": {"used": False}}))
    with pytest.raises(SystemExit, match="--predict ran with the C = 96 fallback False"):
        W.build_report(fallback=True)


# --------------------------------------------------------------------------- prediction wiring (synthetic test bird)
class _Net:
    """A stand-in neural model: records its inputs and mask, and predicts zero displacement."""

    def __init__(self):
        self.seen, self.masks = [], []

    def __call__(self, x, observed=None):
        import torch

        self.seen.append(x.detach().cpu().numpy())
        self.masks.append(observed)
        return torch.zeros(x.shape[0], len(W.HORIZONS), 2)


def _synthetic_bird(root, stem, n_bins=140):
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


def test_predict_bird_scores_each_arm_on_its_windows_and_checks_p_against_the_reused(tmp_path, monkeypatch):
    bird = W.BIRDS[0]
    monkeypatch.setattr(b1h, "SEGMENTS", tmp_path)
    monkeypatch.setattr(env, "ENV_DIR", tmp_path / "env")
    env.ENV_DIR.mkdir()
    _synthetic_bird(tmp_path, bird)
    for owner, attr in ((W, "PREDICTIONS"), (E, "PREDICTIONS"), (V, "PREDICTIONS")):
        monkeypatch.setattr(owner, attr, tmp_path / f"{owner.__name__}_pred")
        getattr(owner, attr).mkdir()
    monkeypatch.setattr(E, "test_windows", TEST_WINDOWS)  # this test's "test bird" is synthetic
    monkeypatch.setattr(env, "env_windows", ENV_WINDOWS)
    monkeypatch.setattr(b1h, "horizon_windows", HORIZON_WINDOWS)
    monkeypatch.setattr(E, "scored_ids", lambda stem, dropped: [f"c_{stem}"])

    ids = [f"c_{bird}"]
    for context, tag, studies in ((12, "P", ("1h", "env")), (96, "P_C96", ("1h",))):
        windows = TEST_WINDOWS(bird, ids, context)
        p = E.frame_table(windows, None, context, E.history_share(windows, context), persistence=True)
        for study in studies:
            p.to_parquet(W.prediction_path(bird, tag, study), index=False)

    pool = [b for b in W.BIRDS if b != bird]
    stats = {"channels": list(era5.ENV_CHANNELS[:14]), "mean": [50.0] * 14, "sd": [0.5] * 14}
    own = W.own_runs(bird, F.PLAN)
    records = {run.run_id: {"pool_species": pool, "env_stats": stats if run.env else None} for run in own}
    models = {run.run_id: _Net() for run in own}
    monkeypatch.setattr(W, "load_models", lambda runs, recs, device: [(r, models[r.run_id], np.ones(3)) for r in runs])
    out = W.predict_bird(bird, [], records, "cpu", overwrite=False, runs=F.PLAN)

    assert {Path(f).stem.split("__", 1)[1] for f in out["files"]} == {"P", "P_C96", *records}
    for run in own:
        x = np.concatenate(models[run.run_id].seen)
        if run.context == 96:
            assert x.shape[1:] == (95, 12) and models[run.run_id].masks[0] is not None
        elif run.env:
            assert x.shape[1:] == (11, 27) and np.abs(x[..., 12:26]).max() == era5.CLIP_SD
        else:
            assert x.shape[1:] == (11, 12) and models[run.run_id].masks[0] is None
    for run in own:  # every saved frame describes P's windows
        frame = pd.read_parquet(W.prediction_path(bird, run.run_id))
        for h in W.HORIZONS:
            E.assert_aligned(E.at(frame, h), E.at(pd.read_parquet(W.prediction_path(bird, "P")), h), run.run_id)

    pd.read_parquet(W.prediction_path(bird, "P", "env")).assign(error_m=0.0).to_parquet(
        W.prediction_path(bird, "P", "env"), index=False)
    with pytest.raises(SystemExit, match="persistence"):
        W.predict_bird(bird, [], records, "cpu", overwrite=False, runs=F.PLAN)


def test_predict_bird_refuses_a_model_whose_pool_held_the_bird():
    bird = W.BIRDS[0]
    own = W.own_runs(bird, F.PLAN)
    records = {r.run_id: {"pool_species": [bird] if r == own[4] else []} for r in own}
    with pytest.raises(SystemExit, match=f"{own[4].run_id}'s pool held {bird}"):
        W.predict_bird(bird, [], records, "cpu", overwrite=False, runs=F.PLAN)


# --------------------------------------------------------------------------- FP32 and the markdown
def test_main_switches_tf32_off_before_anything_runs(monkeypatch):
    import sys

    import torch

    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    seen = {}
    monkeypatch.setattr(W, "diagnostic", lambda fallback: seen.update(env.tf32_flags() | {"fallback": fallback}))
    monkeypatch.setattr(sys, "argv", ["bird_1h_bbf_sealed_eval.py", "--diagnostic", "--c96-fallback"])
    W.main()
    assert seen == {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False, "fallback": True}


def test_the_markdown_renders_every_section_from_a_synthetic_report(tmp_path):
    persistence, frames, parsed, expected = _cell(noise=0.3)
    arms, readings = W.cell_readings(persistence, frames, parsed, BOUNDARY, expected, "synthetic", False)
    cell = {"scored": True, "at_floor": 6, "individuals": 6, "share_in_travel": 1.0, "arms": arms, "readings": readings}
    birds = [W.BIRDS[0], next(b for b in W.BIRDS if W.FOLD_OF[b] == 6)]
    report = {
        "registered_by": "preregistration/birds_1h_bbf_exploratory_freeze.md",
        "attestation": {"freeze_commit": "a" * 40, "freeze_blob": "b" * 40},
        "bbf": "Bb with the repaired encoder (the repair probe's V4 switches)", "switches": V4,
        "evaluation_floor": 150, "min_scored_individuals": 5,
        "tf32": {"predict": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}, "report": {}},
        "c96_fallback": {"used": False, "lstmbbr96_clean": 30, "lstmbbr96_planned": 30},
        "arms": {label: {} for label in W.labels_of(False)}, "comparisons": list(W.comparisons_of(False)),
        "contrasts": list(W.CONTRASTS), "hardware": list(W.hardware_of(False)),
        "completeness": {"n_planned": 120, "n_clean": 120},
        "species": {bird: {"fold": W.FOLD_OF[bird], "horizons": {
            "1": cell, "6": cell, "24": {"scored": False, "at_floor": 2, "individuals": 3}}} for bird in birds},
    }
    report["across_birds"] = W.across_birds(report, False)
    report["tie_back"] = {"arms": {label: {"n": 4, "max_abs_diff": 0.0, "median_diff": 0.0} for label in W.TIED_BACK},
                          "scored_differs": []}
    path = tmp_path / "report.md"
    W.write_markdown(report, path)
    text = path.read_text()
    for heading in ("skill against P", "comparisons and contrast", "Across birds: skill", "without fold 6",
                    "Hardware across birds", "Tie-back of the reused arms"):
        assert heading in text
    assert "C = 96 fallback not used: LSTMBbR96 30 of 30 clean" in text and text.count("unscored") == 2
