# GenAI reference: I listed what the pooling evaluator must guarantee, which is the seal before any test read, only the 15
# pooling runs predicted, clean run and routing checks, the interaction on the same resamples, the TV summary, dilution,
# etc. GenAI wrote these tests from that, on synthetic data only.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.pooling_sealed_eval as P
from tests.test_bird_rotation_sealed_eval import _prediction

PAYLOAD = P.PT.load_registered_pool()
COUNTS = PAYLOAD["train_window_counts"]
HELD_OUT = [entry[0] for entry in P.T.HELD_OUT]


# --------------------------------------------------------------------------- seal

FREEZE_TEXT = (
    "# Pooling\n\n## 5. Readings\n\ntext\n\n## 6. Pre run predictions\n\n| a | b |\n\n"
    "## 7. As trained\n\n15 of 15 runs, 257.18 SU.\n\n## 8. Limitations\n\n- one\n"
)


class _Shown:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


def test_sections_are_found_by_number():
    assert P.section_body(FREEZE_TEXT, 6) == "| a | b |"
    assert P.section_body(FREEZE_TEXT, 7) == "15 of 15 runs, 257.18 SU."
    with pytest.raises(SystemExit, match="no section 9"):
        P.section_body(FREEZE_TEXT, 9)


@pytest.mark.parametrize("body", ["", "WILL FILL", "to fill", "source: JACK TO WRITE", "DRAFT"])
def test_a_placeholder_or_empty_section_is_refused(body):
    with pytest.raises(SystemExit):
        P.check_filled(7, body)


def _seal(monkeypatch, tmp_path, text=FREEZE_TEXT):
    seen = {}

    def committed():
        seen["freeze"] = P.S2.FREEZE
        return {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40}

    monkeypatch.setattr(P.S2, "assert_freeze_committed", committed)
    monkeypatch.setattr(P.E4, "git", lambda *args: _Shown(text))
    monkeypatch.setattr(P, "MANIFEST", tmp_path / "manifest.json")
    return seen


def test_the_seal_checks_this_freeze_and_restores_the_tier2_one(monkeypatch, tmp_path):
    original = P.S2.FREEZE
    seen = _seal(monkeypatch, tmp_path)
    assert P.assert_sealed()["sections_filled"] == [6, 7]
    assert seen["freeze"] == P.FREEZE and P.FREEZE.name == "pooling_exploratory_freeze.md"
    assert P.S2.FREEZE == original


def test_predict_stops_at_an_unfilled_sec_7_before_any_test_partition_is_read(monkeypatch, tmp_path):
    _seal(monkeypatch, tmp_path, FREEZE_TEXT.replace("15 of 15 runs, 257.18 SU.", "WILL FILL."))

    def opened(species):
        raise AssertionError(f"{species}'s test partition was opened before the seal")

    monkeypatch.setattr(P.S2, "test_frame", opened)
    with pytest.raises(SystemExit, match="sec 7"):
        P.run_predictions(device="cpu")


def test_only_the_15_pooling_runs_are_predicted(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "assert_sealed", lambda: {"freeze_blob": "b"})
    monkeypatch.setattr(P, "completeness", lambda records, payload: {"complete": True, "problems": []})
    monkeypatch.setattr(P, "read_records", lambda: {})
    monkeypatch.setattr(P, "PREDICTIONS", tmp_path / "predictions")
    monkeypatch.setattr(P, "MANIFEST", tmp_path / "manifest.json")
    loaded = []
    monkeypatch.setattr(P.S2, "test_frame", lambda species: pd.DataFrame({"individual_id": [1]}))
    monkeypatch.setattr(P.E3, "load_reported", lambda directory, run_id: loaded.append((directory, run_id)) or (None, {}))
    monkeypatch.setattr(P, "MovementFMPredictor", lambda model, **kwargs: None)
    monkeypatch.setattr(P, "evaluate_batched", lambda predictor, frame, **kwargs: pd.DataFrame({"error_m": [1.0]}))
    manifest = P.run_predictions(device="cpu")
    ids = [run.run_id for run in P.PT.PLAN]
    assert len(ids) == 15
    for species in HELD_OUT:
        assert manifest["species"][species] == [f"{species}__{run_id}.parquet" for run_id in ids]
    assert {directory for directory, _ in loaded} == {P.PT.RESULTS}
    assert [run_id for _, run_id in loaded] == ids * 3


# --------------------------------------------------------------------------- runs and routing

def _record(run) -> dict:
    spec = P.PT.ARMS[run.arm]
    weights = P.PT.balanced_weights(PAYLOAD) if spec["balanced"] else PAYLOAD["source_pool"]
    record = {
        "run_id": run.run_id, "arm": run.arm, "seed": run.seed, "tier": "pooling",
        "config": {"steps": 20_000, "trunk": spec["trunk"], "num_layers": 4},
        "applied_updates": 20_000, "skipped_steps": 0, "withheld_fraction": 0.0, "nonfinite_steps": [],
        "final_loss": 0.2, "n_parameters": P.N_PARAMETERS[spec["trunk"]], "pool_windows": P.POOL_WINDOWS,
        "draw_probabilities": dict(weights),
        "loss_shares": [{name: 1.0 / len(weights) for name in weights}] * 4,
    }
    if spec["trunk"] == "moe":
        record["routing"] = {
            name: {"domain": PAYLOAD["domains"][name], "windows": min(2_048, COUNTS[name]),
                   "layers": [{"mean_prob": [0.125] * 8, "top_k_share": [0.125] * 8, "window_agreement": 0.9}] * 4}
            for name in COUNTS
        }
    return record


@pytest.fixture
def checkpoints(monkeypatch):
    monkeypatch.setattr(P.E3, "load_reported", lambda directory, run_id: (object(), {}))


def test_a_clean_plan_is_complete(checkpoints):
    gap = P.completeness({run.run_id: _record(run) for run in P.PT.PLAN}, PAYLOAD)
    assert gap["complete"] and gap["n_clean"] == 15


@pytest.mark.parametrize("change, expect", [
    ({"applied_updates": 19_999}, "applied"),
    ({"skipped_steps": 3, "withheld_fraction": 0.001}, "withheld"),
    ({"nonfinite_steps": [{"step": 1}]}, "non-finite"),
    ({"final_loss": float("nan")}, "final loss"),
    ({"n_parameters": 2_112_259}, "parameters"),
    ({"pool_windows": 2_570_301}, "pool"),
])
def test_each_unclean_run_is_named(change, expect, checkpoints):
    run = P.PT.PLAN[5]
    assert any(expect in problem for problem in P.run_problems(run, _record(run) | change, PAYLOAD))


def test_wrong_draw_probabilities_and_a_missing_checkpoint_are_named(monkeypatch):
    run = P.PT.PLAN[0]  # balanced
    record = _record(run) | {"draw_probabilities": dict(PAYLOAD["source_pool"])}
    monkeypatch.setattr(P.E3, "load_reported", lambda directory, run_id: None)
    problems = P.run_problems(run, record, PAYLOAD)
    assert "draw probabilities differ from sec 2" in problems and "no reported checkpoint" in problems


def test_routing_must_cover_every_source_layer_and_window_count():
    routing = _record(P.PT.PLAN[5])["routing"]
    assert P.routing_problems(routing, 4, COUNTS) == []
    assert P.routing_problems(None, 4, COUNTS) == ["no routing record"]
    short = {k: v for k, v in routing.items() if k != "lynx_rufus"}
    assert any("25 of 26 sources" in p for p in P.routing_problems(short, 4, COUNTS))
    small = dict(routing, aptenodytes_patagonicus=dict(routing["aptenodytes_patagonicus"], windows=2_048))
    assert any("expected 1170" in p for p in P.routing_problems(small, 4, COUNTS))
    assert any("routing layers" in p for p in P.routing_problems(routing, 3, COUNTS))


# --------------------------------------------------------------------------- statistic

BOUNDARY = 1.0  # every persistence error is above it, so every window is travel


def _arms_and_frames(frames: dict[str, pd.DataFrame], persistence: pd.DataFrame):
    """Minimal arm blocks: one seed per arm, its skill computed on the point estimate."""
    resampler = P.RE.Resampler(persistence, BOUNDARY)

    def p90(frame):
        return float(np.quantile(frame["error_m"].to_numpy()[resampler.selected], 0.9))

    arms = {arm: {"skill": {"travel": {"p90_m": 1 - p90(frame) / p90(persistence)}},
                  "headline": {"travel": {"headline_seed": 0}}, "tags": {"0": arm}}
            for arm, frame in frames.items()}
    return arms, resampler


def test_the_interaction_is_zero_when_the_two_differences_coincide():
    ids = list(range(1, 7))
    persistence = _prediction(ids, seed=0)
    a, b = _prediction(ids, seed=1), _prediction(ids, seed=2)
    frames = {"B26balmoe": a, "B26bal": b, "B26moe": a, "B26": b}
    arms, resampler = _arms_and_frames(frames, persistence)
    out = P.interaction(arms, frames, resampler)
    assert out["skill_difference"] == pytest.approx(0.0, abs=1e-12)
    for key in ("skill_difference", "difference_m"):
        assert out["skill_bootstrap"][key]["lo"] == pytest.approx(0.0, abs=1e-9)
        assert out["skill_bootstrap"][key]["hi"] == pytest.approx(0.0, abs=1e-9)


def test_the_interaction_uses_the_same_resamples_as_every_comparison():
    ids = list(range(1, 7))
    persistence = _prediction(ids, seed=0)
    a, m, r = (_prediction(ids, seed=s) for s in (1, 2, 3))
    # B26bal = B26, so the interaction collapses to B26balmoe - B26moe
    frames = {"B26balmoe": a, "B26bal": r, "B26moe": m, "B26": r}
    arms, resampler = _arms_and_frames(frames, persistence)
    out = P.interaction(arms, frames, resampler)["skill_bootstrap"]
    pair = P.RE.Resampler(persistence, BOUNDARY).skill_interval("B26balmoe", a, "B26moe", m)
    assert out["skill_difference"]["lo"] == pytest.approx(pair["skill_difference"]["lo"], abs=1e-12)
    assert out["skill_difference"]["hi"] == pytest.approx(pair["skill_difference"]["hi"], abs=1e-12)
    assert out["difference_m"]["lo"] == pytest.approx(pair["difference_m"]["lo"], abs=1e-9)
    assert out["difference_m"]["hi"] == pytest.approx(pair["difference_m"]["hi"], abs=1e-9)


@pytest.mark.parametrize("gap, fires", [(0.05, False), (0.0501, True), (-0.2, False), (float("nan"), False)])
def test_the_conditional_control_fires_only_strictly_above_0_05(gap, fires):
    report = {"species": {species: {"readings": {"against_b26": {"B26moe": {"skill_difference": 0.0}}}}
                          for species in HELD_OUT}}
    report["species"][P.NES]["readings"]["against_b26"]["B26moe"]["skill_difference"] = gap
    control = P.conditional_control(report)
    assert control["fires"] is fires and control["fires_on"] == ([P.NES] if fires else [])


# --------------------------------------------------------------------------- expert use and dilution

def test_tv_is_zero_for_identical_routing_and_one_for_disjoint():
    same = {d: np.full(8, 0.125) for d in ("flying", "marine", "terrestrial")}
    assert P.mean_pairwise_tv(same) == pytest.approx(0.0)
    disjoint = {d: np.eye(8)[i] for i, d in enumerate(("flying", "marine", "terrestrial"))}
    assert P.mean_pairwise_tv(disjoint) == pytest.approx(1.0)


def test_expert_use_weights_each_domain_s_sources_equally_and_reports_tv_per_seed():
    records = {run.run_id: _record(run) for run in P.PT.PLAN}
    marine = [name for name in COUNTS if PAYLOAD["domains"][name] == "marine"]
    for run in P.PT.PLAN:
        if run.arm == "B26moe":
            routing = records[run.run_id]["routing"]
            routing[marine[0]] = dict(routing[marine[0]], layers=[
                {"mean_prob": [0.125] * 8, "top_k_share": [1.0] + [0.0] * 7, "window_agreement": 1.0}] * 4)
    use = P.expert_use(records)
    last = use["B26moe"]["seeds"][0][-1]
    assert last["top_k_share"]["marine"][0] == pytest.approx((1.0 + 0.125 * (len(marine) - 1)) / len(marine))
    assert last["top_k_share"]["flying"] == pytest.approx([0.125] * 8)
    assert use["B26balmoe"]["median_tv_distance"] == pytest.approx([0.0] * 4)
    assert use["B26moe"]["median_tv_distance"][0] > 0


def test_dilution_reports_raw_and_rescaled_domain_shares():
    names = ["a", "b", "c"]
    domains = {"a": "flying", "b": "marine", "c": "terrestrial"}
    records = [{"loss_shares": [{"a": 0.6, "b": 0.2, "c": 0.1}] * 4}] * 3
    block = P.dilution_block(records, {"a": 0.5, "b": 0.3, "c": 0.2}, domains)
    assert block["n_seeds"] == 3 and [row["species"] for row in block["rows"]] == names
    assert block["domains"]["flying"]["loss_share_raw"] == pytest.approx(0.6)
    assert sum(v["loss_share_rescaled"] for v in block["domains"].values()) == pytest.approx(1.0)
    assert block["domains"]["flying"]["loss_share_rescaled"] == pytest.approx(0.6 / 0.9)


# --------------------------------------------------------------------------- carried and seeds

def _summaries(species: str, drop: str | None = None) -> dict:
    arms = [*P.NEW_ARMS, *P.CARRIED_TIER4, *P.CARRIED_TIER3]
    out = {"P": {"by_seed": {-1: {}}}}
    for arm in arms:
        if arm == "BdM" and species != P.NES:
            continue
        out[arm] = {"by_seed": {seed: {} for seed in P.PT.SEEDS if (arm, seed) != (drop, 4)}}
    return out


def test_every_arm_must_hold_seeds_0_to_4_and_bdm_only_on_nes():
    P.check_seeds("larus_fuscus", _summaries("larus_fuscus"))
    P.check_seeds(P.NES, _summaries(P.NES))
    with pytest.raises(SystemExit, match="XGBB26 has seeds"):
        P.check_seeds(P.NES, _summaries(P.NES, drop="XGBB26"))


def _skills(values: dict) -> dict:
    return {arm: {"skill": {"travel": {"p90_m": value}}} for arm, value in values.items()}


def test_carried_skills_must_reproduce_tier4_and_the_freeze():
    values = {"B26": 0.0016, "BdM": 0.14571, "B9": 0.12719, "LSTMB26": 0.12281, "XGBB26": 0.17281}
    report = {"species": {P.NES: {"arms": _skills(values)}}}
    P.carried_check(P.NES, _skills(values), report)
    with pytest.raises(SystemExit, match="B26"):
        P.carried_check(P.NES, _skills(values | {"B26": 0.0017}), report)
    off = values | {"BdM": 0.1459}
    with pytest.raises(SystemExit, match="freeze"):
        P.carried_check(P.NES, _skills(off), {"species": {P.NES: {"arms": _skills(off)}}})


def test_bdm_is_expected_only_on_nes():
    assert "BdM" in P.expected_carried(P.NES) and "BdM" not in P.expected_carried("canis_lupus")
    assert P.expected_carried("canis_lupus") == {"P": 1, "B26": 5, "LSTMB26": 5, "XGBB26": 5, "B9": 5}


def test_the_sec_7_block_carries_compute_and_the_persistence_source(checkpoints):
    gap = P.completeness({run.run_id: _record(run) for run in P.PT.PLAN}, PAYLOAD)
    block = P.section7_markdown(gap)
    assert "15 of 15" in block and "257.18 SU" in block and "00:13:59 (index 5)" in block
    assert "Tier 2's sealed prediction files" in block and "| deferred or missing | none |" in block
