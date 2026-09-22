# GenAI reference: I used GenAI to check the Tier 3 evaluator's plumbing on synthetic prediction frames. I have checked it all

import json

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.tier3_sealed_eval as E

N_INDIVIDUALS, N_WINDOWS = 3, 40  # above the A5 floor of 25 windows per individual
BOUNDARY = 100.0


def frame(error_m: float, *, offset: float = 0.0) -> pd.DataFrame:
    """One prediction file's worth of rows: constant error, so every statistic is that error."""
    rows = []
    for individual in range(N_INDIVIDUALS):
        for window in range(N_WINDOWS):
            rows.append(
                {
                    "individual_id": f"i{individual}",
                    "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=6 * window),
                    "target_lon": 10.0 + window,
                    "target_lat": -30.0 - window,
                    "pred_lon": 10.0 + window + offset,
                    "pred_lat": -30.0 - window - offset,
                    "error_m": error_m,
                }
            )
    return pd.DataFrame(rows)


PERSISTENCE = frame(1_000.0)  # every window is travel at BOUNDARY = 100 m


def make_record(tag: str) -> dict:
    """
    The result record a Tier 3 training run would have written for this run id.

    The evaluator reads the record rather than the filename, so a test that stands in for a trained run has to
    supply one too.
    """
    parts = tag.split("_")
    arm, seed, rest = parts[0], int(parts[-1].removeprefix("s")), parts[1:-1]
    k = draw = None
    if len(rest) >= 2 and rest[-1].startswith("d") and rest[-2].startswith("k"):
        k, draw = int(rest[-2].removeprefix("k")), int(rest[-1].removeprefix("d"))
        rest = rest[:-2]
    return {
        "run_id": tag, "arm": arm, "species": "_".join(rest) or None, "seed": seed,
        "k_percent": k, "draw": draw, "tier": "3",
    }


def arms_from(frames: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """Run the evaluator's own summary path over synthetic frames and return (arms, frames)."""
    carried = {name for name, _ in E.LADDER_FREE} | {f"{n}_dom" for n in E.DOMAIN_ROW_CARRIED} | set(E.TIER_E_TEST_ARMS)
    records = {
        tag: make_record(tag)
        for tag in frames
        if not tag.startswith("tier2:") and tag.partition("_s")[0] not in carried
    }
    summaries = E.summarise_all(frames, PERSISTENCE, BOUNDARY, records)
    persistence_headline = {
        stratum: {name: E.S2.headline(summaries["P"]["by_seed"], stratum, name)[0] for name in E.STATISTICS}
        for stratum in (*E.STRATA, "all")
    }
    arms = {}
    for arm, entry in summaries.items():
        block = (
            E.headline_block(entry, persistence_headline)
            if entry["by_seed"]
            else {"n_seeds": None, "headline": {}, "skill": {}, "per_seed": {}}
        )
        if entry["by_draw"]:
            block["by_draw"] = E.draw_block(entry, persistence_headline)
        block["tags"] = {E.tag_key(key): tag for key, tag in entry["tags"].items()}
        arms[arm] = block
    return arms, frames


def test_tag_keys_are_strings_for_seeds_and_subsets():
    assert E.tag_key(3) == "3"
    assert E.tag_key(-1) == "-1"
    assert E.tag_key((1, 2)) == "k001_d2"
    assert E.tag_key((100, 0)) == "k100_d0"


def test_tags_are_parsed_from_the_run_record_not_the_filename():
    records = {"F9_canis_lupus_k001_d2_s0": make_record("F9_canis_lupus_k001_d2_s0")}
    assert E.parse_tag("F9_canis_lupus_k001_d2_s0", records) == {"arm": "F9", "seed": 0, "k": 1, "draw": 2}
    assert E.parse_tag("tier2:Z_s3", {}) == {"arm": "Z", "seed": 3, "k": None, "draw": None}
    assert E.parse_tag("tier2:P", {}) == {"arm": "P", "seed": None, "k": None, "draw": None}
    assert E.parse_tag("ARIMA", {}) == {"arm": "ARIMA", "seed": None, "k": None, "draw": None}
    assert E.parse_tag("E1embed_dom_s1", {}) == {"arm": "E1embed_dom", "seed": 1, "k": None, "draw": None}
    assert E.parse_tag("SB_s1", {}) == {"arm": "SB", "seed": 1, "k": None, "draw": None}
    # a Tier 3 tag with no result record is reported and ignored, never guessed at from its name
    assert E.parse_tag("Z9_s0", {}) is None
    assert E.parse_tag("something_else_s0", {}) is None


def test_alignment_guard_refuses_a_file_describing_other_windows():
    E.assert_aligned(frame(500.0), PERSISTENCE, "ok")
    with pytest.raises(SystemExit):
        E.assert_aligned(frame(500.0).iloc[:-1], PERSISTENCE, "short")
    shifted = frame(500.0)
    shifted["target_lon"] = shifted["target_lon"] + 1.0
    with pytest.raises(SystemExit):
        E.assert_aligned(shifted, PERSISTENCE, "shifted")


def test_skill_and_noise_scale():
    assert E.skill(500.0, 1_000.0) == pytest.approx(0.5)
    assert np.isnan(E.skill(500.0, 0.0))
    assert np.isnan(E.skill(None, 1_000.0))
    assert E.resolved(0.06) is True
    assert E.resolved(-0.06) is True
    assert E.resolved(0.04) is False
    assert E.resolved(float("nan")) is None


def test_zero_shot_comparison_reports_the_pair_and_the_bootstrap(monkeypatch):
    monkeypatch.setattr(E.S2, "N_BOOTSTRAP", 50)  # the design is Tier 2's; 50 resamples keeps the test quick
    arms, frames = arms_from({"tier2:P": PERSISTENCE, "Z9_s0": frame(400.0), "tier2:Z_s0": frame(600.0)})
    out = E.compare_arms("Z9", "Z", arms, frames, PERSISTENCE, BOUNDARY)

    assert out["Z9_travel_p90_skill"] == pytest.approx(0.6)
    assert out["Z_travel_p90_skill"] == pytest.approx(0.4)
    assert out["left_better"] is True
    assert out["resolved_against_noise_scale"] is True
    difference = out["bootstrap"]["series"]["travel_p90_m"]["difference_m"]
    assert difference["label"] == "Z9 - Z"
    # constant errors make every resample identical: Z9 is 200 m better and the interval excludes zero
    assert difference["lo"] == pytest.approx(-200.0) and difference["hi"] == pytest.approx(-200.0)
    assert difference["excludes_zero"] is True


def test_a_missing_arm_is_reported_rather_than_scored():
    arms, frames = arms_from({"tier2:P": PERSISTENCE, "Z9_s0": frame(400.0)})
    out = E.compare_arms("Z9", "Zd3", arms, frames, PERSISTENCE, BOUNDARY)
    assert "error" in out and "Zd3" in out["error"]
    assert "left_better" not in out


def test_few_shot_is_reported_per_draw_and_never_pooled(monkeypatch):
    monkeypatch.setattr(E.S2, "N_BOOTSTRAP", 50)
    frames = {"tier2:P": PERSISTENCE}
    # F9 wins two draws and loses one, which is exactly the case a pooled interval would hide
    for arm, errors in (("F9", (300.0, 800.0, 400.0)), ("Sk", (600.0, 600.0, 600.0))):
        for draw, error in enumerate(errors):
            frames[f"{arm}_canis_lupus_k001_d{draw}_s0"] = frame(error)
    arms, frames = arms_from(frames)
    out = E.compare_draws("F9", "Sk", 1, arms, frames, PERSISTENCE, BOUNDARY)

    assert sorted(out["draws"]) == ["d0", "d1", "d2"]
    assert out["draws"]["d0"]["difference_m"] == pytest.approx(-300.0)
    assert out["draws"]["d1"]["left_better"] is False
    assert out["across_draws"]["n_draws"] == 3
    assert out["across_draws"]["mean_difference_m"] == pytest.approx(-100.0)
    assert out["across_draws"]["range_m"] == [pytest.approx(-300.0), pytest.approx(200.0)]
    assert out["across_draws"]["left_better_in"] == 2
    assert out["across_draws"]["descriptive"] is True
    # every draw carries its own interval, and there is no single pooled interval anywhere in the output
    assert all("bootstrap" in detail for detail in out["draws"].values())
    assert "bootstrap" not in out


def test_success_bands_read_tier3_arms_under_tier2_names():
    arms, _ = arms_from(
        {
            "tier2:P": PERSISTENCE,
            "Z9_s0": frame(400.0),
            "B9_s0": frame(700.0),
            "tier2:S_s0": frame(380.0),
            "Za9_canis_lupus_s0": frame(800.0),
        }
    )
    verdict = E.registered_bands({"species": {"canis_lupus": {"arms": arms}}})

    checks = verdict["bands"]["per_species"]["canis_lupus"]["checks"]
    assert checks["Z_beats_P_travel_p90"] is True
    assert checks["Z_beats_B_travel_p90"] is True
    # 400 m is within 1.25 x the specialist's 380 m, so the specialist check passes as well
    assert checks["Z_within_25pc_of_S_travel_p90"] is True
    assert verdict["bands"]["band"] == "STRONG"
    # Za is present as Za9, so it is not reported as a completeness gap
    assert "Za@canis_lupus" not in verdict["bands"]["deferred_comparators"]


def test_weak_band_when_the_nine_source_model_loses_to_its_own_baseline():
    arms, _ = arms_from({"tier2:P": PERSISTENCE, "Z9_s0": frame(900.0), "B9_s0": frame(400.0)})
    verdict = E.registered_bands({"species": {"canis_lupus": {"arms": arms}}})
    assert verdict["bands"]["per_species"]["canis_lupus"]["checks"]["Z_beats_B_travel_p90"] is False
    assert verdict["bands"]["band"] == "WEAK/NULL"


def test_arima_fallback_rate_is_counted_per_axis():
    fallback = frame(1_000.0)  # identical predictions to persistence on both axes
    moved = frame(1_000.0, offset=0.5)
    half = pd.concat([fallback.iloc[: len(fallback) // 2], moved.iloc[len(moved) // 2 :]]).reset_index(drop=True)

    assert E.arima_fallback_rate({"ARIMA": fallback, "tier2:P": PERSISTENCE}) == {
        "lon_axis": 1.0, "lat_axis": 1.0, "both_axes": 1.0, "either_axis": 1.0
    }
    assert E.arima_fallback_rate({"ARIMA": moved, "tier2:P": PERSISTENCE}) == {
        "lon_axis": 0.0, "lat_axis": 0.0, "both_axes": 0.0, "either_axis": 0.0
    }
    mixed = E.arima_fallback_rate({"ARIMA": half, "tier2:P": PERSISTENCE})
    assert mixed["both_axes"] == pytest.approx(0.5) and mixed["either_axis"] == pytest.approx(0.5)
    assert E.arima_fallback_rate({"tier2:P": PERSISTENCE}) is None


def test_tier_e_records_cannot_enter_a_tier3_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "TIER3_RESULTS", tmp_path)
    (tmp_path / "Z9_s0.json").write_text(json.dumps(make_record("Z9_s0")))
    (tmp_path / "Z9long_s0.json").write_text(
        json.dumps(make_record("Z9long_s0") | {"tier": "E", "exploratory": True})
    )
    assert [record["run_id"] for record in E.tier3_runs()] == ["Z9_s0"]

    (tmp_path / "X_s0.json").write_text(json.dumps(make_record("X_s0")))
    with pytest.raises(SystemExit):
        E.tier3_runs()

def test_xgboost_metadata_files_are_ignored(tmp_path, monkeypatch):
    """
    The XGBoost arms save a metadata file beside their result record. It has no run_id, and a record reader that does
    not skip it crashes the report; this is a regression test for that.
    """
    monkeypatch.setattr(E, "TIER3_RESULTS", tmp_path)
    (tmp_path / "XGBB9_s0.json").write_text(json.dumps(make_record("XGBB9_s0") | {"model": "WindowXGBoost"}))
    (tmp_path / "XGBB9_s0_xgb_meta.json").write_text(
        json.dumps({"params": {"n_estimators": 200}, "n_features": 132, "name": "window_xgboost"})
    )
    assert [r["run_id"] for r in E.tier3_runs()] == ["XGBB9_s0"]
    records = {
        record["run_id"]: record
        for record in (json.loads(open(path).read()) for path in sorted(tmp_path.glob("*.json")))
        if "run_id" in record and "arm" in record
    }
    assert list(records) == ["XGBB9_s0"]
