# GenAI reference: I asked for tests of the BbF extension's --extension mode tests. 
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_1h_bbf_sealed_eval as W

E, F, env, b1h, RS, E3 = W.E, W.F, W.env, W.b1h, W.RS, W.E3
TEST_WINDOWS, HORIZON_WINDOWS = E.test_windows, b1h.horizon_windows  # before the fixture
T0 = pd.Timestamp("2021-01-01", tz="UTC")
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
DEFAULT_SWITCHES = {"input_scale": "sqrt_d", "norm_first": False, "step_transform": "none"}
BIRD = next(b for b in W.BIRDS if W.FOLD_OF[b] == 3)


def _refuse(*args, **kwargs):
    raise AssertionError("a test reached a test-partition reader")


@pytest.fixture(autouse=True)
def _no_test_partition(monkeypatch):
    """Every reader of test-bird fixes, ids or windows raises, so no test here can open a test partition."""
    for owner, attr in ((E, "test_windows"), (E, "scored_ids"), (E, "scoring_state"), (RS, "load_corpus"),
                        (RS, "test_frame"), (env, "env_windows"), (b1h, "horizon_windows")):
        monkeypatch.setattr(owner, attr, _refuse)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Every prediction folder and the extension's own folder under tmp_path."""
    for owner, attr in ((W, "PREDICTIONS"), (E, "PREDICTIONS"), (W, "EXT_PREDICTIONS")):
        monkeypatch.setattr(owner, attr, tmp_path / f"{owner.__name__.rsplit('.', 1)[-1]}_{attr}")
        getattr(owner, attr).mkdir()
    monkeypatch.setattr(W, "EXT_SEALED", tmp_path / "ext")
    monkeypatch.setattr(W, "EXT_MANIFEST", tmp_path / "ext" / "manifest.json")
    return tmp_path


# --------------------------------------------------------------------------- layout
def test_the_extension_layout_and_its_own_folder():
    assert list(W.EXT_ARMS)[:5] == ["BbF", "BbF_C24", "BbF_C48", "BbF_C72", "BbF_C96"] and len(W.EXT_ARMS) == 20
    assert {label for label, (study, *_) in W.EXT_ARMS.items() if study == "ext"} == {"BbF_C24", "BbF_C48", "BbF_C72"}
    assert W.EXT_ARMS["BbF"] == W.ARMS["BbF"] and W.EXT_ARMS["BbF_C96"] == W.ARMS["BbF_C96"]
    assert W.EXT_ARMS["LSTMBb_C48"] == ("1h", "LSTMBb", 48) and W.EXT_ARMS["XGBBb"] == ("1h", "XGBBb", 12)
    assert [W.name(p) for p in W.EXT_COMPARISONS] == [
        f"BbF_C{c} - {other}" for c in (24, 48, 72) for other in ("P", f"LSTMBb_C{c}", f"XGBBb_C{c}")]
    assert W.LADDER == (24, 48, 72, 96) and W.EXT_ABLATION_ARMS == ("BbF", "Bb", "LSTMBb", "XGBBb")
    assert W.EXT_SEALED.name == "birds_1h_bbf_extension_sealed" and W.EXT_SEALED != W.SEALED
    for path in (W.EXT_PREDICTIONS, W.EXT_REPORT_JSON, W.EXT_REPORT_MD, W.EXT_MANIFEST):
        assert path.parent in (W.EXT_SEALED, W.EXT_PREDICTIONS) and W.SEALED not in path.parents
    assert "V100" in W.LSTM_COMPARATOR and "H200" in W.LSTM_COMPARATOR


def test_tags_and_read_files_reach_each_studys_own_fold_runs():
    assert W.ext_tags_at(BIRD, "BbF_C48", 6) == [(f"BbF_f3_s{s}_C48", s) for s in (0, 1, 2)]
    assert W.ext_tags_at(BIRD, "BbF", 6) == W.tags_at(BIRD, "BbF", 6)
    assert W.ext_tags_at(BIRD, "LSTMBb_C24", 1) == [(f"LSTMBb_f3_s{s}_C24", s) for s in (0, 1, 2)]
    assert W.ext_tags_at(BIRD, "XGBBb_C72", 24) == [("XGBBb_f3_h24_C72", 0)]
    assert W.ext_tags_at(BIRD, "XGBBb", 1) == [("XGBBb_f3_h1", 0)]
    reads = W.ext_read_files(BIRD)
    assert reads[:7] == [("1h", "P"), ("1h", "P_C24"), ("1h", "P_C48"), ("1h", "P_C72"), ("1h", "P_C96"),
                         ("bbf", "P"), ("bbf", "P_C96")]
    assert not any(study == "ext" for study, _ in reads) and len(reads) == len(set(reads))
    assert ("bbf", "BbF_f3_s2_C96") in reads and ("1h", "Bb_f3_s1_C72") in reads and ("1h", "XGBBb_f3_h6_C24") in reads
    assert len(reads) == 7 + 3 * 2 + 3 * 5 * 2 + 3 * 5  # P files, BbF 12 / 96, Bb and LSTM-Bb, XGB-Bb x 3 horizons


# --------------------------------------------------------------------------- seal
FILLED = ("## 6. Expectations\n\nmine\n\n## 7. As trained\n\nfilled\n\n## 10. Extension (26-10-2026)\n\n- bullets\n\n"
          "### Extension expectations\n\nAt 6 h, BbF - LSTM-Bb is within 0.05.\n\n### Extension as trained\n\n- **Runs.** 90.\n")


@pytest.mark.parametrize("text, problem", [
    (FILLED, None),
    (FILLED.replace("At 6 h, BbF - LSTM-Bb is within 0.05.", "Jack to write."), "'Extension expectations' still says"),
    (FILLED.replace("- **Runs.** 90.", "To fill after training."), "'Extension as trained' still says"),
    (FILLED.replace("- **Runs.** 90.", ""), "'Extension as trained' is empty"),
    (FILLED.replace("### Extension expectations", "### Expectations"), "has no '### Extension expectations'"),
])
def test_both_sec_10_subsections_must_be_filled(text, problem):
    problems = W.ext_section_problems(text)
    assert problems == [] if problem is None else (len(problems) == 1 and problem in problems[0])


def test_the_seal_uses_the_extensions_manifest_and_names_the_filled_sections(monkeypatch):
    seen = []
    monkeypatch.setattr(RS, "assert_sealed", lambda: seen.append((RS.FREEZE, RS.MANIFEST)) or {"freeze_blob": "b"})
    monkeypatch.setattr(RS, "committed_freeze", lambda: FILLED)
    attestation = W.ext_assert_sealed()
    assert seen == [(W.FREEZE, W.EXT_MANIFEST)] and RS.MANIFEST != W.EXT_MANIFEST  # restored afterwards
    assert attestation["sections_filled"] == [6, 7, "10: Extension expectations", "10: Extension as trained"]
    monkeypatch.setattr(RS, "committed_freeze", lambda: FILLED.replace("- **Runs.** 90.", "To fill"))
    with pytest.raises(SystemExit, match="Extension as trained' still says"):
        W.ext_assert_sealed()


def test_predictions_check_the_seal_before_anything_else(monkeypatch):
    order = []

    def seal():
        order.append("seal")
        raise SystemExit("sealed")

    monkeypatch.setattr(W, "ext_assert_sealed", seal)
    for name in ("ext_source_reports", "read_records", "ext_checked_state", "ext_missing_reads"):
        monkeypatch.setattr(W, name, lambda *a, name=name: order.append(name))
    with pytest.raises(SystemExit, match="sealed"):
        W.ext_run_predictions("cpu")
    assert order == ["seal"]


def test_the_extension_needs_the_1h_and_bbf_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "REPORT_JSON", tmp_path / "birds_1h_results.json")
    monkeypatch.setattr(W, "REPORT_JSON", tmp_path / "birds_1h_bbf_results.json")
    E.REPORT_JSON.write_text(json.dumps({"study": "birds_1h", "attestation": {"freeze_commit": "a"}}))
    with pytest.raises(SystemExit, match="birds_1h_bbf_results.json does not exist"):
        W.ext_source_reports()
    W.REPORT_JSON.write_text(json.dumps({"study": "birds_1h_bbf", "attestation": {"freeze_commit": "b"}}))
    assert set(W.ext_source_reports()) == {"birds_1h", "birds_1h_bbf"}


# --------------------------------------------------------------------------- records
@pytest.fixture
def clean(tmp_path, monkeypatch):
    """The 90 extension records, clean, with checkpoints."""
    payload = b1h.load_registered()
    monkeypatch.setattr(F, "RESULTS", tmp_path / "bbf_results")
    F.RESULTS.mkdir()
    for run in F.EXTENSION:
        index = F.PLAN.index(run)
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        record = {
            "run_id": run.run_id, "plan_index": index, "arm": run.arm, "fold": run.fold, "seed": run.seed,
            "context": run.context, "switches": V4, "env": False, "env_channels": 0, "env_clip_sd": None,
            "queue": F.queue_of(index), "tier": F.TAG, "opened_test_partition": False, "pool_species": names,
            "excluded_individuals": b1h.RT.excluded_in(names, payload),
            "pool_windows": sum(b1h.TRAIN_WINDOWS[n][0] for n in names),
            "config": {"steps": 20_000, "env_channels": 0, **V4},
            "tf32": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}, "device": "NVIDIA H200",
            "pbs_queue": "gpuhopper-exec", "applied_updates": 20_000, "skipped_steps": 0, "withheld_fraction": 0,
            "nonfinite_steps": [], "reported_checkpoint": f"{run.run_id}_reported.pt",
        }
        (F.RESULTS / f"{run.run_id}.json").write_text(json.dumps(record))
        (F.RESULTS / f"{run.run_id}_reported.pt").write_bytes(b"")
    return payload


def _rewrite(run_id, **changes):
    path = F.RESULTS / f"{run_id}.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def test_clean_extension_records_pass_and_the_original_120_are_not_needed(clean):
    gap, exclusions = W.ext_checked_state(W.read_records(), clean)
    assert gap["complete"] and gap["n_clean"] == 90 and exclusions["runs"] == 90
    assert set(gap["by_arm"]) == {"BbF_C24", "BbF_C48", "BbF_C72"}


@pytest.mark.parametrize("run_id, changes, message", [
    ("BbF_f3_s1_C24", {"switches": {}}, "switches {}, expected"),
    ("BbF_f3_s1_C48", {"config": {"steps": 20_000, "env_channels": 0, **DEFAULT_SWITCHES}}, "config switches"),
    ("BbF_f4_s0_C72", {"context": 96}, "context 96, expected 72"),
    ("BbF_f4_s0_C72", {"tf32": {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False}}, "tf32"),
    ("BbF_f9_s2_C24", {"plan_index": 3}, "plan_index 3, expected 149"),
])
def test_an_unclean_extension_run_is_refused(clean, run_id, changes, message):
    _rewrite(run_id, **changes)
    with pytest.raises(SystemExit, match=f"{run_id}.*{message}"):
        W.ext_checked_state(W.read_records(), clean)


def test_predictions_refuse_missing_sealed_files_before_any_test_read(clean, dirs, monkeypatch):
    monkeypatch.setattr(W, "ext_assert_sealed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(W, "ext_source_reports", lambda: {})
    with pytest.raises(SystemExit, match="sealed prediction files are missing"):
        W.ext_run_predictions("cpu")
    for bird in W.BIRDS:
        for study, tag in W.ext_read_files(bird):
            W.ext_path(bird, tag, study).write_bytes(b"")
    with pytest.raises(AssertionError, match="test-partition reader"):  # every guard passed; scoring reads test next
        W.ext_run_predictions("cpu")


def test_the_diagnostic_prints_bullets_the_seal_accepts_and_never_opens_test(clean, monkeypatch, capsys):
    monkeypatch.setattr(W, "freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(RS, "committed_freeze", lambda: FILLED.replace("- **Runs.** 90.", "To fill after training."))
    W.ext_diagnostic()
    out = capsys.readouterr().out
    assert "Extension expectations : filled" in out and "still says 'To fill' in the committed version (not enforced here)" in out
    bullets = out.split("\"Extension as trained\" ---\n\n", 1)[1]
    assert "90 of 90 recorded, 90 clean (BbF_C24 30 of 30; BbF_C48 30 of 30; BbF_C72 30 of 30)" in bullets
    assert "TF32 off on 90; devices NVIDIA H200 90; PBS queues gpuhopper-exec 90" in bullets
    assert f"{V4} recorded on 90 runs" in bullets and "PASS for all 90 runs" in bullets
    assert not RS.PLACEHOLDER.search(bullets)
    _rewrite("BbF_f2_s0_C48", applied_updates=19_000)
    with pytest.raises(SystemExit, match="diagnostic found a problem"):
        W.ext_diagnostic()


def test_main_refuses_the_c96_fallback_with_the_extension_and_dispatches_its_diagnostic(monkeypatch):
    seen = []
    monkeypatch.setattr(W, "ext_diagnostic", lambda: seen.append("extension"))
    monkeypatch.setattr(W, "diagnostic", lambda fallback: seen.append("original"))
    monkeypatch.setattr(sys, "argv", ["x", "--diagnostic", "--extension"])
    W.main()
    monkeypatch.setattr(sys, "argv", ["x", "--diagnostic", "--extension", "--c96-fallback"])
    with pytest.raises(SystemExit, match="belongs to the original 120 runs"):
        W.main()
    assert seen == ["extension"]


# --------------------------------------------------------------------------- frames, arithmetic, tie-back
SCALES = {"BbF": 0.80, "BbF_C24": 0.75, "BbF_C48": 0.72, "BbF_C72": 0.71, "BbF_C96": 0.70,
          "Bb": 0.98, "Bb_C24": 0.99, "Bb_C48": 1.00, "Bb_C72": 1.00, "Bb_C96": 1.00,
          "LSTMBb": 0.89, "LSTMBb_C24": 0.85, "LSTMBb_C48": 0.80, "LSTMBb_C72": 0.82, "LSTMBb_C96": 0.83,
          "XGBBb": 0.83, "XGBBb_C24": 0.80, "XGBBb_C48": 0.79, "XGBBb_C72": 0.79, "XGBBb_C96": 0.78}
GAPPY = {"BbF_C48": 0.50}  # on gappy windows this label's errors are scaled by this instead


def _persistence(n_individuals=6, n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"individual_id": f"i{k}", "timestamp": T0 + pd.Timedelta(hours=i), "target_lon": 0.1 * i,
             "target_lat": 1.0} for k in range(n_individuals) for i in range(n)]
    frame = pd.DataFrame(rows)
    frame["error_m"] = rng.gamma(2.0, 500.0, size=len(frame)) + 200.0
    return frame.assign(pred_lon=0.0, pred_lat=0.0, horizon=6, history_share=np.nan)


def _share(frame):
    """Every individual's first 40 windows gappy (0.5 observed), the rest full."""
    position = frame.groupby("individual_id").cumcount().to_numpy()
    return np.where(position < 40, 0.5, 1.0)


def _tables():
    p = _persistence()
    share = _share(p)
    tables = {"P": p, **{E.persistence_tag(c): p.assign(history_share=share) for c in W.LADDER}}
    for label, scale in SCALES.items():
        pairs = W.ext_tags_at(BIRD, label, 6)
        for tag, seed in pairs:
            factor = scale * (1 + 0.01 * (seed - 1) * (len(pairs) > 1))
            if label in GAPPY:
                factor = np.where(share < 0.9, GAPPY[label], scale) * (1 + 0.01 * (seed - 1))
            tables[tag] = p.assign(error_m=p["error_m"] * factor)
    return tables


ELIGIBLE = {"6": {"individuals": 6, "at_floor": 6, "windows": 960, "scored": True}}


def test_the_cell_has_the_sec_10_arithmetic_and_the_ablations_shape(monkeypatch):
    monkeypatch.setattr(b1h, "BOUNDARY_M", defaultdict(lambda: 1.0))  # every window is travel
    cell = W.ext_cell(BIRD, 6, _tables(), ELIGIBLE)
    skill = {label: 1 - scale for label, scale in SCALES.items() if label not in GAPPY}
    assert {k: v for k, v in cell["readings"]["against_p"].items() if k not in GAPPY} == pytest.approx(skill)
    c = cell["readings"]["comparisons"]
    assert c["BbF_C24 - P"]["skill_difference"] == pytest.approx(0.25)
    assert c["BbF_C24 - LSTMBb_C24"]["skill_difference"] == pytest.approx(0.10)
    assert c["BbF_C72 - XGBBb_C72"]["skill_difference"] == pytest.approx(0.08)
    boot = c["BbF_C72 - LSTMBb_C72"]["skill_bootstrap"]["skill_difference"]
    assert boot["lo"] == pytest.approx(0.11) == boot["hi"]
    ablation = cell["ablation"]
    assert set(ablation) == {"24", "48", "72", "96"} and set(ablation["24"]) == {"all", "gappy", "full"}
    a24 = ablation["24"]["all"]
    assert a24["BbF"] == pytest.approx({"skill": 0.25, "skill_C12": 0.20, "difference": 0.05})
    assert a24["LSTMBb"] == pytest.approx({"skill": 0.15, "skill_C12": 0.11, "difference": 0.04})
    assert ablation["96"]["full"]["Bb"] == pytest.approx({"skill": 0.0, "skill_C12": 0.02, "difference": -0.02})
    assert ablation["48"]["gappy"]["BbF"]["skill"] == pytest.approx(0.50)
    assert ablation["48"]["full"]["BbF"]["skill"] == pytest.approx(0.28)
    assert ablation["48"]["gappy"]["n_individuals"] == 6 and ablation["48"]["full"]["n_travel_windows"] == 720


def test_a_label_missing_a_seed_or_misaligned_is_refused(monkeypatch):
    monkeypatch.setattr(b1h, "BOUNDARY_M", defaultdict(lambda: 1.0))
    tables = _tables()
    del tables["BbF_f3_s2_C48"]
    with pytest.raises(SystemExit, match="no predictions for BbF_f3_s2_C48"):
        W.ext_cell(BIRD, 6, tables, ELIGIBLE)
    tables = _tables()
    tables["LSTMBb_f3_s0_C24"] = tables["LSTMBb_f3_s0_C24"].iloc[::-1].reset_index(drop=True).assign(
        timestamp=lambda f: f["timestamp"] + pd.Timedelta(minutes=1))
    with pytest.raises(SystemExit, match="LSTMBb_f3_s0_C24 does not describe the same windows"):
        W.ext_cell(BIRD, 6, tables, ELIGIBLE)


def _report_from(cell, birds, every_horizon=False):
    other = cell if every_horizon else {"scored": False, "at_floor": 2, "individuals": 3}
    return {"species": {b: {"fold": W.FOLD_OF[b], "horizons": {"1": other, "6": cell, "24": other}} for b in birds}}


def test_across_birds_and_the_ablation_summary_have_the_figures_shape(monkeypatch):
    monkeypatch.setattr(b1h, "BOUNDARY_M", defaultdict(lambda: 1.0))
    cell = W.ext_cell(BIRD, 6, _tables(), ELIGIBLE)
    report = _report_from(cell, [BIRD], every_horizon=True)
    across = W.ext_across_birds(report)
    assert across["common_set"] == [BIRD] and across["horizons"]["6"]["scored"] == [BIRD]
    assert across["horizons"]["6"]["skill"]["BbF_C72"]["scored"] == {"n": 1, "median": pytest.approx(0.29)}
    summary = across["horizons"]["6"]["comparisons"]["BbF_C48 - LSTMBb_C48"]["scored"]
    assert set(summary) == {"difference", "bands", "intervals_excluding_zero"}
    ablation = W.ext_ablation_summary(report, across["common_set"])
    node = ablation["BbF"]["6"]["24"]["all"]
    assert node["scored"]["difference"] == {"n": 1, "median": pytest.approx(0.05)} and node["common"] == node["scored"]
    assert set(ablation) == {"BbF", "Bb", "LSTMBb", "XGBBb"} and set(ablation["XGBBb"]["6"]) == {"24", "48", "72", "96"}


def test_tie_back_is_exact_on_matching_reports_and_shows_any_drift(monkeypatch):
    monkeypatch.setattr(b1h, "BOUNDARY_M", defaultdict(lambda: 1.0))
    cell = W.ext_cell(BIRD, 6, _tables(), ELIGIBLE)
    ours = _report_from(cell, [BIRD])
    one_h_arms = {arm: {"skill": {"travel": {"p90_m": cell["readings"]["against_p"][arm]}}}
                  for arm in ("Bb", "LSTMBb", "XGBBb")}
    sources = {
        "birds_1h": {"attestation": {"freeze_commit": "a"}, "species": {BIRD: {"horizons": {
            "1": {"scored": False}, "24": {"scored": False},
            "6": {"scored": True, "arms": one_h_arms, "ablation": json.loads(json.dumps(cell["ablation"]))}}}}},
        "birds_1h_bbf": {"attestation": {"freeze_commit": "b"}, "species": {BIRD: {"horizons": {
            "1": {"scored": False}, "24": {"scored": False},
            "6": {"scored": True, "readings": {"against_p": {"BbF": 0.20, "BbF_C96": 0.30}}}}}}},
    }
    tie = W.ext_tie_back(ours, sources)
    assert set(tie["arms"]) == set(W.EXT_TIED_BACK) and len(W.EXT_TIED_BACK) == 17
    assert all(entry["max_abs_diff"] == pytest.approx(0, abs=1e-12) for entry in tie["arms"].values())
    assert all(entry["max_abs_diff"] == pytest.approx(0, abs=1e-12) for entry in tie["ablation"].values())
    sources["birds_1h"]["species"][BIRD]["horizons"]["6"]["ablation"]["72"]["gappy"]["LSTMBb"]["skill"] += 0.01
    sources["birds_1h_bbf"]["species"][BIRD]["horizons"]["6"]["readings"]["against_p"]["BbF_C96"] = 0.29
    tie = W.ext_tie_back(ours, sources)
    assert tie["ablation"]["LSTMBb"]["max_abs_diff"] == pytest.approx(0.01)
    assert tie["arms"]["BbF_C96"]["max_abs_diff"] == pytest.approx(0.01)
    assert tie["ablation"]["BbF"]["max_abs_diff"] == pytest.approx(0.01)
    assert not any(k in json.dumps(tie) for k in ("pass", "fail"))


def test_the_markdown_renders_from_a_synthetic_report(tmp_path, monkeypatch):
    monkeypatch.setattr(b1h, "BOUNDARY_M", defaultdict(lambda: 1.0))
    cell = W.ext_cell(BIRD, 6, _tables(), ELIGIBLE) | {"at_floor": 6, "individuals": 6}
    report = _report_from(cell, [BIRD]) | {
        "registered_by": "preregistration/birds_1h_bbf_exploratory_freeze.md sec 10",
        "attestation": {"freeze_commit": "a" * 40, "freeze_blob": "b" * 40},
        "bbf": "Bb with the repaired encoder", "switches": V4, "evaluation_floor": 150, "min_scored_individuals": 5,
        "lstm_comparator": W.LSTM_COMPARATOR, "tf32": {"predict": {}, "report": {}},
        "comparisons": [W.name(p) for p in W.EXT_COMPARISONS], "completeness": {"n_planned": 90, "n_clean": 90},
        "tie_back": {"arms": {label: {"n": 1, "max_abs_diff": 0.0, "median_diff": 0.0} for label in W.EXT_TIED_BACK},
                     "ablation": {"LSTMBb": {"n": 36, "max_abs_diff": 0.0, "median_diff": 0.0}}, "scored_differs": []},
    }
    report["across_birds"] = W.ext_across_birds(report)
    report["ablation"] = W.ext_ablation_summary(report, [])
    path = tmp_path / "report.md"
    W.ext_write_markdown(report, path)
    text = path.read_text()
    for heading in ("skill against P", "comparisons (skill difference", "Across birds: skill", "History gain on",
                    "Tie-back of the read arms", "ablation block: LSTMBb", "LSTM comparator: the 1 h ablation's"):
        assert heading in text
    assert "90 of 90 clean" in text and text.count("unscored") == 2


# --------------------------------------------------------------------------- the sealed files and prediction wiring
def _write_reads(bird, tables):
    for study, tag in W.ext_read_files(bird):
        tables[tag].to_parquet(W.ext_path(bird, tag, study), index=False)


def _manifest(bird):
    return {"read": {bird: {f"{s}:{t}": W.sha256(W.ext_path(bird, t, s)) for s, t in W.ext_read_files(bird)}}}


def test_saved_files_load_with_their_sha256_and_ps_checked(dirs):
    tables = _tables()
    tables |= {tag: tables["P"] for _, tag in W.ext_read_files(BIRD) if tag not in tables}  # XGB at 1 h, 24 h
    _write_reads(BIRD, tables)
    for c in W.EXT_CONTEXTS:
        tables[E.persistence_tag(c)].to_parquet(W.ext_path(BIRD, E.persistence_tag(c)), index=False)
    for run in W.ext_own_runs(BIRD):
        tables[run.run_id].to_parquet(W.ext_path(BIRD, run.run_id), index=False)
    manifest = _manifest(BIRD)
    loaded = W.ext_load_tables(BIRD, manifest)
    assert set(loaded) == set(tables) and loaded["P_C48"]["history_share"].notna().all()

    E.at(tables["P"], 6).assign(history_share=0.95).to_parquet(W.ext_path(BIRD, "P_C24"), index=False)
    with pytest.raises(SystemExit, match="history_share differs"):
        W.ext_load_tables(BIRD, manifest)
    tables["P_C24"].to_parquet(W.ext_path(BIRD, "P_C24"), index=False)
    tables["P"].assign(error_m=1.0).to_parquet(W.ext_path(BIRD, "P_C96", "bbf"), index=False)
    with pytest.raises(SystemExit, match="P_C96.parquet is missing or differs from the sha256"):
        W.ext_load_tables(BIRD, manifest)
    with pytest.raises(SystemExit, match="persistence"):
        W.ext_load_tables(BIRD, _manifest(BIRD))
    tables["P"].to_parquet(W.ext_path(BIRD, "P_C96", "bbf"), index=False)
    stray = next(r for r in F.EXTENSION if r.fold != W.FOLD_OF[BIRD])
    tables["P"].to_parquet(W.ext_path(BIRD, stray.run_id), index=False)
    with pytest.raises(SystemExit, match=f"outside its own fold.*{stray.run_id}"):
        W.ext_load_tables(BIRD, _manifest(BIRD))


class _Net:
    """A stand-in neural model: records its inputs and mask, and predicts zero displacement."""

    def __init__(self):
        self.seen, self.masks = [], []

    def __call__(self, x, observed=None):
        import torch

        self.seen.append(x.detach().cpu().numpy())
        self.masks.append(observed)
        return torch.zeros(x.shape[0], len(W.HORIZONS), 2)


def _synthetic_bird(root, stem, n_bins=160):
    rng = np.random.default_rng(3)
    rows = [{"segment_id": f"c_{stem}_s0", "individual_id": f"c_{stem}", "bin_index": b,
             "timestamp": pd.Timestamp("2020-03-01", tz="UTC") + pd.Timedelta(seconds=3600 * b + int(rng.integers(0, 3000))),
             "lon": 10.0 + 0.004 * b + rng.normal(0, 3e-4), "lat": 50.0 + 0.002 * b + rng.normal(0, 3e-4)}
            for b in range(n_bins) if not 60 <= b < 70]  # a gap, so the extended history is gappy somewhere
    pd.DataFrame(rows).to_parquet(root / f"{stem}_segments.parquet")


def test_predict_bird_writes_only_the_extensions_folder_and_checks_p_against_the_1h(dirs, monkeypatch):
    monkeypatch.setattr(b1h, "SEGMENTS", dirs)
    _synthetic_bird(dirs, BIRD)
    monkeypatch.setattr(E, "test_windows", TEST_WINDOWS)  # this test's "test bird" is synthetic
    monkeypatch.setattr(b1h, "horizon_windows", HORIZON_WINDOWS)
    monkeypatch.setattr(E, "scored_ids", lambda stem, dropped: [f"c_{stem}"])
    ids = [f"c_{BIRD}"]
    for context in (12, *W.EXT_CONTEXTS):
        windows = TEST_WINDOWS(BIRD, ids, context)
        p = E.frame_table(windows, None, context, E.history_share(windows, context), persistence=True)
        p.to_parquet(W.ext_path(BIRD, E.persistence_tag(context), "1h"), index=False)
    before = sorted(p.name for p in W.PREDICTIONS.iterdir()) + sorted(p.name for p in E.PREDICTIONS.iterdir())

    own = W.ext_own_runs(BIRD)
    records = {run.run_id: {"pool_species": [b for b in W.BIRDS if b != BIRD]} for run in own}
    models = {run.run_id: _Net() for run in own}
    monkeypatch.setattr(W, "load_models", lambda runs, recs, device: [(r, models[r.run_id], np.ones(3)) for r in runs])
    out = W.ext_predict_bird(BIRD, [], records, "cpu", overwrite=False)

    assert {Path(f).stem.split("__", 1)[1] for f in out["files"]} == {"P_C24", "P_C48", "P_C72", *records}
    assert {p.stem.split("__", 1)[1] for p in W.EXT_PREDICTIONS.iterdir()} == {"P_C24", "P_C48", "P_C72", *records}
    assert sorted(p.name for p in W.PREDICTIONS.iterdir()) + sorted(p.name for p in E.PREDICTIONS.iterdir()) == before
    for run in own:
        x = np.concatenate(models[run.run_id].seen)
        assert x.shape[1:] == (run.context - 1, 12) and models[run.run_id].masks[0] is not None
        frame = pd.read_parquet(W.ext_path(BIRD, run.run_id))
        for h in W.HORIZONS:
            E.assert_aligned(E.at(frame, h), E.at(pd.read_parquet(W.ext_path(BIRD, "P", "1h")), h), run.run_id)

    sealed = pd.read_parquet(W.ext_path(BIRD, "P_C48", "1h"))
    sealed.assign(history_share=sealed["history_share"] * 0.5).to_parquet(W.ext_path(BIRD, "P_C48", "1h"), index=False)
    with pytest.raises(SystemExit, match="history_share differs"):
        W.ext_predict_bird(BIRD, [], records, "cpu", overwrite=False)


def test_predict_bird_refuses_a_model_whose_pool_held_the_bird():
    own = W.ext_own_runs(BIRD)
    records = {r.run_id: {"pool_species": [BIRD] if r == own[4] else []} for r in own}
    with pytest.raises(SystemExit, match=f"{own[4].run_id}'s pool held {BIRD}"):
        W.ext_predict_bird(BIRD, [], records, "cpu", overwrite=False)
