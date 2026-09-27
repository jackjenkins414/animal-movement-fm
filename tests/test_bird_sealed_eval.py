# GenAI reference: I outlined what our bird evaluator has to guarantee and GenAI wrote these tests from that specification, 
# will add the list later. 
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import pytest

import experiments.reporting.bird_sealed_eval as BE

GULL, GOOSE, DOVE = BE.GULL, BE.GOOSE, BE.DOVE
SIX = [490431939, 662388807, 1056306620, 662388728, 133992693, 133992680]
PAYLOAD = BE.BT.load_registered_pool()


# Seal.
FILLED = "# Birds\n\n## 7. As trained\n\n95 of 95 runs, 458.0 SU.\n\n## 8. Attestation\n\nText.\n"


@pytest.mark.parametrize("body", ["", "TO FILL after training.", "WILL FILL after training.",
                                  "**DRAFT (GenAI)** 95 of 95", "Will  fill later"])
def test_the_placeholder_check_refuses_an_unfilled_section(body):
    with pytest.raises(SystemExit):
        BE.check_as_trained(body)


def test_the_placeholder_check_accepts_a_filled_section():
    BE.check_as_trained(BE.as_trained_section(FILLED))


class _Shown:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


def test_the_seal_reads_section_7_from_head_and_restores_the_tier2_freeze(monkeypatch, tmp_path):
    seen = {}

    def committed():
        seen["freeze"] = BE.S2.FREEZE
        return {"freeze_commit": "c" * 40, "freeze_blob": "b" * 40}

    original = BE.S2.FREEZE
    monkeypatch.setattr(BE.S2, "assert_freeze_committed", committed)
    monkeypatch.setattr(BE.E4, "git", lambda *args: _Shown(FILLED))
    monkeypatch.setattr(BE, "MANIFEST", tmp_path / "manifest.json")
    attestation = BE.assert_sealed()
    assert seen["freeze"] == BE.PREREG
    assert BE.S2.FREEZE == original
    assert attestation["freeze_blob"] == "b" * 40 and attestation["as_trained_lines"] == 1


def test_the_seal_refuses_the_committed_placeholder(monkeypatch, tmp_path):
    monkeypatch.setattr(BE.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(BE.E4, "git", lambda *args: _Shown(FILLED.replace("95 of 95 runs, 458.0 SU.", "WILL FILL.")))
    monkeypatch.setattr(BE, "MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(SystemExit, match="still says 'WILL FILL'"):
        BE.assert_sealed()


def test_a_later_predict_run_under_another_blob_is_refused(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"attestation": {"freeze_blob": "a" * 40}}))
    monkeypatch.setattr(BE, "MANIFEST", manifest)
    monkeypatch.setattr(BE.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b" * 40})
    monkeypatch.setattr(BE.E4, "git", lambda *args: _Shown(FILLED))
    with pytest.raises(SystemExit, match="Rebuild the report with --report"):
        BE.assert_sealed()


def test_no_test_partition_is_read_before_the_seal(monkeypatch):
    opened = []
    monkeypatch.setattr(BE, "assert_sealed", lambda: (_ for _ in ()).throw(SystemExit("sealed shut")))
    monkeypatch.setattr(BE, "test_frame", lambda species: opened.append(species))
    with pytest.raises(SystemExit, match="sealed shut"):
        BE.run_predictions(device=None)
    assert opened == []


def test_report_mode_never_checks_or_rewrites_the_seal(monkeypatch, tmp_path):
    monkeypatch.setattr(BE, "assert_sealed", lambda: (_ for _ in ()).throw(AssertionError("report re-sealed")))
    monkeypatch.setattr(BE, "build_report", lambda: {"species": {}})
    monkeypatch.setattr(BE, "write_markdown", lambda report, path: None)
    monkeypatch.setattr(BE, "SEALED", tmp_path)
    monkeypatch.setattr(BE, "REPORT_JSON", tmp_path / "birds_results.json")
    monkeypatch.setattr(BE, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(sys, "argv", ["bird_sealed_eval.py", "--report"])
    BE.main()
    assert not (tmp_path / "manifest.json").exists()


# Records and boundaries.
def test_only_bird_records_are_read_and_xgb_metadata_is_skipped_silently(monkeypatch, tmp_path, capsys):
    (tmp_path / "Zb_s0.json").write_text(json.dumps({"run_id": "Zb_s0", "arm": "Zb", "tier": "birds", "seed": 0}))
    (tmp_path / "Z26_s0.json").write_text(json.dumps({"run_id": "Z26_s0", "arm": "Z26", "tier": "4"}))
    (tmp_path / "XGBS_anser_albifrons_s0_xgb_meta.json").write_text(json.dumps({"params": {}}))
    monkeypatch.setattr(BE, "BIRD_RESULTS", tmp_path)
    records = BE.read_records()
    printed = capsys.readouterr().out
    assert [r["run_id"] for r in records] == ["Zb_s0"]
    assert "Z26_s0.json" in printed and "xgb_meta" not in printed


def test_boundaries_come_from_bird_train_and_never_the_shared_profile_table(monkeypatch):
    assert BE.check_boundaries() == BE.BT.HELD_OUT_BOUNDARY_M
    monkeypatch.setitem(BE.BT.PROFILE_TABLES, GULL, BE.BT.IL.PROFILES_CSV)
    with pytest.raises(SystemExit, match="results/regime_profiles.csv"):
        BE.check_boundaries()


# Frames and decoding.
def _frame(path, n=4):
    pd.DataFrame({
        "individual_id": [1] * n,
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="6h"),
        "target_lon": np.arange(n, dtype=float),
        "target_lat": np.arange(n, dtype=float),
        "error_m": np.ones(n),
    }).to_parquet(path, index=False)


@pytest.fixture
def prediction_dirs(tmp_path, monkeypatch):
    dirs = {name: tmp_path / name for name in ("tier2", "tier3", "birds", "records")}
    for directory in dirs.values():
        directory.mkdir()
    monkeypatch.setattr(BE, "TIER2_PREDICTIONS", dirs["tier2"])
    monkeypatch.setattr(BE, "TIER3_PREDICTIONS", dirs["tier3"])
    monkeypatch.setattr(BE, "PREDICTIONS", dirs["birds"])
    monkeypatch.setattr(BE, "BIRD_RESULTS", dirs["records"])
    return dirs


def test_the_gull_loads_exactly_the_registered_carry_list(prediction_dirs):
    for tag in ("P", "V", "S_s0", "Z_s0", "Za_s0"):
        _frame(prediction_dirs["tier2"] / f"{GULL}__{tag}.parquet")
    for tag in ("ARIMA", "SB_s0", f"Sk_{GULL}_k001_d0_s0", f"LSTMS_{GULL}_s0", f"XGBS_{GULL}_s0", "Z9_s0",
                f"F9_{GULL}_k001_d0_s0", "E1embed_dom_s0"):
        _frame(prediction_dirs["tier3"] / f"{GULL}__{tag}.parquet")
    _frame(prediction_dirs["birds"] / f"{GULL}__Zb_s0.parquet")
    _, frames, _ = BE.species_frames(GULL)
    assert set(frames) == {
        "tier2:P", "tier2:V", "tier2:S_s0", "tier3:ARIMA", "tier3:SB_s0", f"tier3:Sk_{GULL}_k001_d0_s0",
        f"tier3:LSTMS_{GULL}_s0", f"tier3:XGBS_{GULL}_s0", "Zb_s0",
    }


def test_the_goose_uses_only_its_own_predictions(prediction_dirs):
    _frame(prediction_dirs["tier2"] / f"{GOOSE}__P.parquet")  # planted: must not be read
    for tag in ("P", "V", "ARIMA", "Zb_s0"):
        _frame(prediction_dirs["birds"] / f"{GOOSE}__{tag}.parquet")
    _, frames, _ = BE.species_frames(GOOSE)
    assert set(frames) == {"P", "V", "ARIMA", "Zb_s0"}


def test_the_goose_and_dove_rungs_are_the_tier2_and_tier3_models():
    tags = [tag for tag, _ in BE.training_free_models()]
    assert tags == ["P", "V", "ARIMA", "AVG3", "AVG10"] == list(BE.TRAINING_FREE)


RECORDS = {"Fb_anser_albifrons_k010_d2_s0": {"arm": "Fb", "seed": 0, "k_percent": 10, "draw": 2}}


@pytest.mark.parametrize("tag,expected", [
    ("Fb_anser_albifrons_k010_d2_s0", ("Fb", 0, 10, 2)),
    ("tier2:P", ("P", None, None, None)),
    ("tier2:S_s3", ("S", 3, None, None)),
    ("tier3:ARIMA", ("ARIMA", None, None, None)),
    ("tier3:SB_s1", ("SB", 1, None, None)),
    (f"tier3:Sk_{GULL}_k001_d2_s0", ("Sk", 0, 1, 2)),
    (f"tier3:LSTMS_{GULL}_s4", ("LSTMS", 4, None, None)),
    (f"tier3:XGBS_{GULL}_s0", ("XGBS", 0, None, None)),
    ("AVG10", ("AVG10", None, None, None)),
])
def test_tags_decode_to_their_arm_seed_and_draw(tag, expected):
    parsed = BE.parse_tag(tag, RECORDS)
    assert (parsed["arm"], parsed["seed"], parsed["k"], parsed["draw"]) == expected


@pytest.mark.parametrize("tag", ["tier2:Z_s0", "tier2:Za_s0", "tier3:Z9_s0", f"tier3:F9_{GULL}_k001_d0_s0",
                                 "tier3:E1embed_dom_s0", "Z26_s0"])
def test_arms_not_carried_for_the_gull_are_ignored(tag, capsys):
    assert BE.parse_tag(tag, RECORDS) is None
    assert "ignored" in capsys.readouterr().out


# Guards.
def test_two_tags_in_one_slot_are_a_collision():
    frames = {f"tier3:Sk_{GULL}_k001_d0_s0": None, f"tier3:Sk_{GULL}_k001_d0": None}
    with pytest.raises(SystemExit, match="same summary slot"):
        BE.assert_no_collisions(frames, {})


def test_a_missing_arm_is_named_with_its_bird():
    summaries = {arm: {} for arm in BE.REQUIRED_ARMS - {"Sk"}}
    assert BE.missing_arms(summaries, GULL) == [("Sk", GULL)]
    assert len(BE.REQUIRED_ARMS) == 17


def test_the_gulls_sk_draws_come_from_tier3s_plan_and_fb_from_the_bird_plan():
    paired = {(1, 0), (1, 1), (1, 2), (10, 0), (10, 1), (10, 2)}
    assert BE.registered_draws("Sk", GULL) == paired
    assert BE.registered_draws("Sk", GOOSE) == paired
    assert BE.registered_draws("Fb", DOVE) == paired | {(100, 0)}


def _summaries(species, drop=()):
    return {arm: {"by_seed": {}, "by_draw": {key: {} for key in BE.registered_draws(arm, species) if key not in drop}}
            for arm in BE.SUBSET_ARMS}


def test_draw_completeness_fails_on_a_missing_draw():
    BE.assert_draws_complete(_summaries(GOOSE), GOOSE)
    with pytest.raises(SystemExit, match=r"Sk / larus_fuscus: missing \[\(10, 2\)\]"):
        summaries = _summaries(GULL)
        summaries["Sk"]["by_draw"].pop((10, 2))
        BE.assert_draws_complete(summaries, GULL)


# Draw identity and goose exclusions, from real splits and the registered pool.
def _identity_records(tmp_path, monkeypatch):
    tier3 = tmp_path / "tier3"
    tier3.mkdir()
    monkeypatch.setattr(BE, "TIER3_RESULTS", tier3)
    records = {}
    for species in BE.BT.HELD_OUT:
        for k, draws in BE.T.FINETUNE_DRAWS_PER_FRACTION.items():
            for draw in range(draws):
                fb = BE.BT.Run("Fb", species, 0, k, draw)
                records[fb.run_id] = {"subset": BE.subset_metadata(BE.BT.finetune_subset(fb, PAYLOAD))}
                if k == 100:
                    continue
                if species == GULL:
                    sk = BE.T3.Run("Sk", species, 0, k, draw)
                    (tier3 / f"{sk.run_id}.json").write_text(
                        json.dumps({"subset": BE.subset_metadata(BE.T.finetune_subset(sk))}))
                else:
                    sk = BE.BT.Run("Sk", species, 0, k, draw)
                    records[sk.run_id] = {"subset": BE.subset_metadata(BE.BT.finetune_subset(sk, PAYLOAD))}
    return records


def test_draw_identity_passes_when_fb_and_sk_share_individuals(tmp_path, monkeypatch):
    result = BE.assert_draw_identity(PAYLOAD, _identity_records(tmp_path, monkeypatch))
    assert result == {"status": "PASS", "paired_draws": 18, "fb_draws": 21}


def test_draw_identity_fails_when_the_individuals_differ(tmp_path, monkeypatch):
    records = _identity_records(tmp_path, monkeypatch)
    original = BE.BT.finetune_subset

    def shifted(run, payload):
        subset = original(run, payload)
        if run.arm == "Fb" and run.species == DOVE and (run.k_percent, run.draw) == (10, 1):
            subset.individuals[0] = -1
        return subset

    monkeypatch.setattr(BE.BT, "finetune_subset", shifted)
    with pytest.raises(SystemExit, match="streptopelia_turtur k10 d1: Fb and Sk train on different individuals"):
        BE.assert_draw_identity(PAYLOAD, records)


def test_draw_identity_fails_when_a_record_disagrees_with_the_recomputation(tmp_path, monkeypatch):
    records = _identity_records(tmp_path, monkeypatch)
    records["Fb_anser_albifrons_k001_d0_s0"]["subset"]["n_individuals_available"] = 299
    with pytest.raises(SystemExit, match="anser_albifrons k1 d0: Fb record says"):
        BE.assert_draw_identity(PAYLOAD, records)


class _Windows:
    def __init__(self, n):
        self.n_windows = n


def _goose_records():
    kept = len(BE.BT.held_out_train_ids(GOOSE, PAYLOAD))
    records = {}
    for run in BE.BT.PLAN:
        if run.species != GOOSE:
            continue
        record = {"excluded_individuals": list(SIX)}
        spec = BE.BT.ARMS[run.arm]
        if spec["pool"] == "held_out_subset":
            record["subset"] = BE.subset_metadata(BE.BT.finetune_subset(run, PAYLOAD))
        else:
            record["training_windows" if spec.get("xgb") else "pool_windows"] = 10 * kept
        records[run.run_id] = record
    return records


@pytest.fixture
def ten_windows_per_bird(monkeypatch):
    monkeypatch.setattr(BE.T, "_load_species_windows", lambda stem, split, part, individuals=None: _Windows(
        10 * len(individuals)))


def test_goose_exclusions_pass_on_consistent_records(ten_windows_per_bird):
    result = BE.assert_goose_exclusions(PAYLOAD, _goose_records())
    assert result["status"] == "PASS" and result["runs"] == 29 and result["training_birds"] == 293
    assert (result["windows_without_the_six"], result["windows_with_the_six"]) == (2930, 2990)


def test_goose_exclusions_fail_on_a_record_that_does_not_name_the_six(ten_windows_per_bird):
    records = _goose_records()
    records["S_anser_albifrons_s0"]["excluded_individuals"] = SIX[:5]
    with pytest.raises(SystemExit, match="S_anser_albifrons_s0: record names"):
        BE.assert_goose_exclusions(PAYLOAD, records)


def test_goose_exclusions_fail_when_a_run_trained_on_the_full_split(ten_windows_per_bird):
    records = _goose_records()
    records["LSTMS_anser_albifrons_s2"]["pool_windows"] = 2990
    with pytest.raises(SystemExit, match="LSTMS_anser_albifrons_s2: pool_windows 2990"):
        BE.assert_goose_exclusions(PAYLOAD, records)


# Readings, at the +/- 0.05 edges.
def _arm(skill):
    return {"skill": {"travel": {"p90_m": skill}}, "headline": {}, "tags": {}}


def _read(reader, **skills):
    arms = {name: _arm(value) for name, value in skills.items()}
    return reader(arms, {}, None, 1.0)


@pytest.mark.parametrize("skill,reading", [(0.0501, "beats persistence zero-shot"),
                                           (0.05, "matches persistence zero-shot"),
                                           (-0.05, "matches persistence zero-shot"),
                                           (-0.0501, "loses to persistence zero-shot")])
def test_vs_persistence_bands(skill, reading):
    out = _read(BE.read_vs_persistence, P=0.0, Zb=skill, Bb=0.0)
    assert out["Zb"]["reading"] == reading


@pytest.mark.parametrize("zb,reading", [(0.30, "zero-shot matches the specialist"),
                                        (0.10, "zero-shot matches the specialist"),
                                        (0.05, "zero-shot matches the specialist"),
                                        (0.0499, "the specialist wins")])
def test_vs_specialist_has_two_outcomes_and_prints_the_gap(zb, reading):
    out = _read(BE.read_vs_specialist, S=0.10, Zb=zb, Bb=0.10)
    assert out["Zb"]["reading"].startswith(reading)
    assert f"gap {zb - 0.10:+.4f}" in out["Zb"]["reading"]


@pytest.mark.parametrize("zb,reading", [(0.06, "the package helps"), (0.05, "no difference"),
                                        (-0.06, "the package hurts")])
def test_package_bands(zb, reading):
    assert _read(BE.read_package, Zb=zb, Bb=0.0)["reading"] == reading


@pytest.mark.parametrize("dz,db,reading", [
    (0.06, 0.06, "congeners carry the transfer"),
    (0.01, -0.02, "congeners add nothing"),
    (0.06, 0.01, "the package and the congeners interact"),
    (0.01, 0.06, "the package and the congeners interact"),
    (-0.06, -0.06, "unresolved"),
    (0.06, -0.06, "unresolved"),
])
def test_the_four_congener_rows(dz, db, reading):
    assert BE.classify_congeners(dz, db, GOOSE) == reading
    assert BE.classify_congeners(dz, db, GULL) == reading


def test_the_dove_rows_are_worded_as_five_unrelated_birds():
    assert BE.classify_congeners(0.06, 0.06, DOVE) == "adding five unrelated birds helps"
    assert BE.classify_congeners(0.0, 0.0, DOVE) == "adding five unrelated birds changes nothing"
    assert BE.classify_congeners(0.06, 0.0, DOVE) == "adding five unrelated birds interacts with the package"
    assert BE.classify_congeners(-0.06, -0.06, DOVE) == "unresolved"


def _draws(skills):
    return {f"k{k:03d}_d{d}": {"k_percent": k, "draw": d, "skill": {"travel": {"p90_m": s}},
                               "headline": {"travel": {"p90_m": 100.0}}}
            for (k, d), s in skills.items()}


def test_fb_vs_sk_is_read_per_draw_and_counted_descriptively():
    arms = {
        "Fb": {"by_draw": _draws({(1, 0): 0.10, (1, 1): 0.02, (1, 2): -0.10, (100, 0): 0.2}), "tags": {}},
        "Sk": {"by_draw": _draws({(1, 0): 0.0, (1, 1): 0.0, (1, 2): 0.0}), "tags": {}},
        "S": _arm(0.15),
    }
    out = BE.read_few_shot(arms, {}, None, 1.0)
    readings = {key: entry["reading"] for key, entry in out["k001"]["per_draw"].items()}
    assert readings == {"k001_d0": "pretraining helps", "k001_d1": "no difference", "k001_d2": "pretraining loses"}
    assert out["k001"]["across_draws"]["descriptive"] is True
    assert out["k100"]["descriptive"] is True and out["k100"]["S_travel_p90_skill"] == 0.15
    assert "no Sk at 100%" in out["k100"]["reading"]


def test_the_ladder_has_no_pass_fail():
    report = {"species": {GULL: {"arms": {"Zb": _arm(0.01)}}}}
    ladder = BE.ladder_table(report)
    assert ladder["reading"] == "descriptive; no pass/fail"
    assert ladder["blocks"]["foundation_model"]["Zb"][GULL] == 0.01


# Completeness and the diagnostic.
def _clean_records(directory):
    records = {}
    for run in BE.BT.PLAN:
        if BE.BT.ARMS[run.arm].get("xgb"):
            for axis in ("east", "north"):
                (directory / f"{run.run_id}_xgb_{axis}.ubj").write_text("")
            records[run.run_id] = {"run_id": run.run_id, "arm": run.arm}
            continue
        (directory / f"{run.run_id}_reported.pt").write_text("")
        records[run.run_id] = {
            "run_id": run.run_id, "arm": run.arm, "config": {"steps": 20_000}, "applied_updates": 20_000,
            "skipped_steps": 0, "withheld_fraction": 0.0, "nonfinite_steps": [],
            "reported_checkpoint": f"{run.run_id}_reported.pt",
        }
    return records


def test_completeness_counts_95_runs_88_neural_7_xgb(tmp_path, monkeypatch):
    monkeypatch.setattr(BE, "BIRD_RESULTS", tmp_path)
    records = _clean_records(tmp_path)
    gap = BE.completeness(records)
    assert (gap["n_planned"], gap["n_records"], gap["n_clean"], gap["n_neural"], gap["n_xgb"]) == (95, 95, 95, 88, 7)
    records["Zb_s3"]["skipped_steps"] = 2
    gap = BE.completeness(records)
    assert not gap["complete"] and gap["problems"] == ["Zb_s3: withheld 2"]


def test_the_dilution_table_sums_the_congeners_per_seed_before_the_median():
    main, congener = PAYLOAD["probabilities"]["main"], PAYLOAD["probabilities"]["congener"]
    even = {name: 1.0 / len(congener) for name in congener}
    records = [{"arm": "Zb", "loss_shares": [{name: 1.0 / len(main) for name in main}] * 4},
               {"arm": "Zbc", "loss_shares": [even] * 4}]
    table = BE.dilution_table(records, PAYLOAD)
    assert set(table["Zb"]["rows"]) == set(main) and "error" in table["Bb"]
    assert table["Zbc"]["congeners_combined"]["loss_share"] == pytest.approx(5 / 27)
    assert table["Zbc"]["congeners_combined"]["draw_share"] == pytest.approx(
        sum(congener[name] for name in BE.BT.SPW.BIRDS_CONGENERS))


def test_the_section7_block_carries_the_facts_checks_and_clarifications(tmp_path, monkeypatch):
    monkeypatch.setattr(BE, "BIRD_RESULTS", tmp_path)
    gap = BE.completeness(_clean_records(tmp_path))
    block = BE.section7_markdown(gap, "PASS (paired_draws 18)", "PASS (runs 29)", {})
    for text in ("95 of 95", "gpuvolta, fp32, torch 2.5.1+cu121", "88 clean", "20,000 of 20,000", "7 fit once",
                 "458.0 SU; longest run 11m11s", "PASS (paired_draws 18)", "| Zb, Bb vs S |", "| dove Δ rows |"):
        assert text in block, text


def test_the_diagnostic_exits_non_zero_when_a_check_fails(monkeypatch, capsys):
    complete = {"n_records": 95, "n_planned": 95, "n_clean": 95, "by_arm": {}, "problems": [], "complete": True,
                "n_neural": 88, "n_xgb": 7}
    monkeypatch.setattr(BE, "read_records", lambda: [])
    monkeypatch.setattr(BE, "completeness", lambda records=None: complete)
    monkeypatch.setattr(BE, "assert_draw_identity", lambda p, r: (_ for _ in ()).throw(SystemExit("FAIL: draw x")))
    monkeypatch.setattr(BE, "assert_goose_exclusions", lambda p, r: {"status": "PASS", "runs": 29})
    monkeypatch.setattr(BE, "dilution_table", lambda records, payload: {})
    with pytest.raises(SystemExit, match="must not be filled"):
        BE.diagnostic()
    printed = capsys.readouterr().out
    assert "draw identity    : FAIL — draw x" in printed and "goose exclusions : PASS (runs 29)" in printed


def _realistic_loss_records():
    """Five seeds per dilution arm, shares proportional to draw share with a seeded wobble, as the records hold them."""
    rng = np.random.default_rng(0)
    records = []
    for arm in BE.DILUTION_ARMS:
        weights = PAYLOAD["probabilities"][BE.BT.ARMS[arm]["pool"]]
        for seed in range(5):
            steps = []
            for _ in range(40):
                raw = {name: p * rng.uniform(0.5, 2.0) for name, p in weights.items()}
                total = sum(raw.values())
                steps.append({name: value / total for name, value in raw.items()})
            records.append({"run_id": f"{arm}_s{seed}", "arm": arm, "loss_shares": steps})
    return records


def test_the_pasted_section7_block_passes_the_seals_placeholder_check(tmp_path, monkeypatch):
    monkeypatch.setattr(BE, "BIRD_RESULTS", tmp_path)
    gap = BE.completeness(_clean_records(tmp_path))
    identity, _ = BE.run_check(lambda: {"status": "PASS", "paired_draws": 18, "fb_draws": 21})
    exclusions, _ = BE.run_check(lambda: {"status": "PASS", "runs": 29, "training_birds": 293,
                                          "windows_without_the_six": 75_000, "windows_with_the_six": 77_550})
    block = BE.section7_markdown(gap, identity, exclusions, BE.dilution_table(_realistic_loss_records(), PAYLOAD))
    assert block.count("\n| ") > 30  # the facts, 27 sources, the congener total and the clarifications

    # pasted into §7 of a preregistration, the seal's own parse and check accept it
    prereg = f"# Birds\n\n## 7. As trained\n\n{block}\n\n## 8. Attestation\n\nText.\n"
    section = BE.as_trained_section(prereg)
    assert section == block.strip()
    BE.check_as_trained(section)

    # nothing the placeholder pattern catches, and no looser 'fill' or 'draft' that a later edit could turn into one
    assert BE.PLACEHOLDER.findall(block) == []
    assert re.findall(r"\bfill\w*|\bdraft\w*", block, re.IGNORECASE) == []
    for item, resolution in BE.CLARIFICATIONS:
        assert not BE.PLACEHOLDER.search(f"{item} {resolution}"), item
    assert {item for item, _ in BE.CLARIFICATIONS}.isdisjoint({"seal", "§8"})
