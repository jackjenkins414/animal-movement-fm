# GenAI reference: I asked GenAI to write a test file that checks the two seal gates, the tier == "4" record filter on both
# reading paths, the 2 x 2 delta arithmetic and sign convention, the freeze's classifications at the +/- 0.05 edges, the 
# dilution table from records alone, and the completeness gap. GenAI wrote those tests from that list and Ichecked them. 

import json

import numpy as np
import pytest

import experiments.reporting.tier4_sealed_eval as E4
import experiments.training.tier4_train as T4

NES = E4.NES
NOISE = E4.NOISE_SCALE


def record(run_id: str, arm: str, *, tier="4", species=None, seed=0, k=None, draw=None, shares=None) -> dict:
    return {
        "run_id": run_id, "arm": arm, "tier": tier, "exploratory": True, "seed": seed,
        "species": species, "k_percent": k, "draw": draw,
        "loss_shares": shares if shares is not None else [],
    }


def write_records(directory, records) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for payload in records:
        (directory / f"{payload['run_id']}.json").write_text(json.dumps(payload))


# Seal gates.
FREEZE_FILLED = """# Tier 4

## 6. Pre-run prediction

Something.

## 7. As trained

91 of 91 runs reported, 1,068 SU. Nothing deferred.

## 8. Note

Text.
"""
FREEZE_UNFILLED = FREEZE_FILLED.replace(
    "91 of 91 runs reported, 1,068 SU. Nothing deferred.", "TO FILL after training: runs, SU, anything deferred."
)
FREEZE_NO_HEADING = FREEZE_FILLED.replace("## 7. As trained", "## 7. Something else")


def fake_git(monkeypatch, blob: str | None, *, returncode: int = 0):
    """Stand in for `git show HEAD:<freeze>`."""
    class Result:
        def __init__(self):
            self.returncode = returncode if blob is None else 0
            self.stdout = blob or ""
            self.stderr = "" if blob is not None else "fatal: path does not exist in HEAD"

    monkeypatch.setattr(E4, "git", lambda *args: Result())


def test_as_trained_accepts_a_filled_section(monkeypatch, capsys):
    fake_git(monkeypatch, FREEZE_FILLED)
    body = E4.assert_as_trained_filled()
    assert "91 of 91" in body
    assert "filled" in capsys.readouterr().out


def test_as_trained_refuses_while_it_still_says_to_fill(monkeypatch):
    fake_git(monkeypatch, FREEZE_UNFILLED)
    with pytest.raises(SystemExit, match="still says TO FILL"):
        E4.assert_as_trained_filled()


def test_as_trained_refuses_an_empty_section(monkeypatch):
    fake_git(monkeypatch, FREEZE_FILLED.replace("91 of 91 runs reported, 1,068 SU. Nothing deferred.", ""))
    with pytest.raises(SystemExit, match="is empty"):
        E4.assert_as_trained_filled()


def test_as_trained_refuses_when_the_heading_is_missing(monkeypatch):
    fake_git(monkeypatch, FREEZE_NO_HEADING)
    with pytest.raises(SystemExit, match="no 'As trained' heading"):
        E4.assert_as_trained_filled()


def test_as_trained_refuses_when_the_freeze_is_not_committed(monkeypatch):
    fake_git(monkeypatch, None, returncode=128)
    with pytest.raises(SystemExit, match="not committed"):
        E4.assert_as_trained_filled()


def test_as_trained_is_read_from_the_committed_blob_not_the_working_tree(monkeypatch, tmp_path):
    # the working tree says TO FILL, the commit does not: the gate follows the commit
    working = tmp_path / "tier4_exploratory_freeze.md"
    working.write_text(FREEZE_UNFILLED)
    monkeypatch.setattr(E4, "FREEZE", working)
    monkeypatch.setattr(E4, "REPO", tmp_path)
    fake_git(monkeypatch, FREEZE_FILLED)
    assert "91 of 91" in E4.assert_as_trained_filled()


def test_the_seal_runs_the_committed_check_before_the_as_trained_check(monkeypatch):
    calls = []
    monkeypatch.setattr(E4.S2, "assert_freeze_committed", lambda: calls.append("committed") or {})
    monkeypatch.setattr(E4, "assert_as_trained_filled", lambda: calls.append("as_trained") or "one line")
    E4.assert_sealed()
    assert calls == ["committed", "as_trained"]


# Record filter, on both paths.
def test_only_tier_4_records_are_read(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    write_records(tmp_path, [
        record("Z26_s0", "Z26"),
        record("Z9_s0", "Z26", tier="3"),
        record("Z9long_s0", "Z26", tier="E"),
    ])
    found = E4.read_records()
    assert [r["run_id"] for r in found] == ["Z26_s0"]
    printed = capsys.readouterr().out
    assert "Z9_s0.json" in printed and "Z9long_s0.json" in printed


def test_the_run_id_map_applies_the_same_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    write_records(tmp_path, [
        record("Z26_s0", "Z26"),
        record("Z9_s0", "Z26", tier="3"),
    ])
    assert set(E4.record_map()) == {"Z26_s0"}


def test_exploratory_is_not_a_filter(tmp_path, monkeypatch):
    # every Tier 4 record is exploratory; filtering on it, as Tier 3 does, would reject all of them
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    payloads = [record(f"Z26_s{seed}", "Z26", seed=seed) for seed in range(5)]
    assert all(p["exploratory"] for p in payloads)
    write_records(tmp_path, payloads)
    assert len(E4.read_records()) == 5


def test_a_record_naming_an_unknown_arm_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    write_records(tmp_path, [record("Z9_s0", "Z9")])
    with pytest.raises(SystemExit, match="not a Tier 4 arm"):
        E4.read_records()


# Completeness.
def test_completeness_names_missing_records_and_checkpoints(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    kept = T4.PLAN[0]
    write_records(tmp_path, [record(kept.run_id, kept.arm)])
    (tmp_path / f"{kept.run_id}_reported.pt").write_text("")
    second = T4.PLAN[1]
    write_records(tmp_path, [record(second.run_id, second.arm, seed=second.seed)])  # record but no checkpoint

    gap = E4.completeness()
    assert gap["n_planned"] == 91 and gap["n_records"] == 2
    assert gap["complete"] is False
    assert second.run_id in gap["missing_checkpoints"]
    assert kept.run_id not in gap["missing_checkpoints"]
    assert len(gap["missing_records"]) == 89
    assert "completeness gap" in capsys.readouterr().out


def test_completeness_accepts_an_xgb_run_with_only_its_booster_files(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    xgb = next(run for run in T4.PLAN if T4.ARMS[run.arm].get("xgb"))
    write_records(tmp_path, [record(xgb.run_id, xgb.arm, seed=xgb.seed)])
    (tmp_path / f"{xgb.run_id}_xgb_model.ubj").write_text("")
    assert xgb.run_id not in E4.completeness()["missing_checkpoints"]


def test_completeness_is_clean_when_every_planned_run_has_both(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    write_records(tmp_path, [record(run.run_id, run.arm, seed=run.seed) for run in T4.PLAN])
    for run in T4.PLAN:
        if T4.ARMS[run.arm].get("xgb"):
            (tmp_path / f"{run.run_id}_xgb_model.ubj").write_text("")
        else:
            (tmp_path / f"{run.run_id}_reported.pt").write_text("")
    gap = E4.completeness()
    assert gap["complete"] is True and gap["missing_records"] == []


# The 2 x 2 arithmetic and sign convention.
def arms_with(skills: dict) -> dict:
    return {arm: {"skill": {"travel": {"p90_m": value}}} for arm, value in skills.items()}


def test_delta_is_left_minus_right_on_travel_p90_skill():
    arms = arms_with({"ZdMnoMir": 0.10, "ZdM": 0.30})
    assert E4.delta(arms, "ZdMnoMir", "ZdM") == pytest.approx(-0.20)


def test_removing_mirounga_costing_skill_is_a_negative_delta():
    # the freeze's convention: delta = without Mirounga - with Mirounga
    arms = arms_with({"ZdM": 0.1140, "ZdMnoMir": 0.0100, "BdM": 0.1272, "BdMnoMir": 0.0200})
    assert E4.delta(arms, "ZdMnoMir", "ZdM") < 0
    assert E4.delta(arms, "BdMnoMir", "BdM") < 0
    assert E4.classify_2x2(E4.delta(arms, "ZdMnoMir", "ZdM"), E4.delta(arms, "BdMnoMir", "BdM")) == "genus"


@pytest.mark.parametrize(
    "delta_z,delta_b,expected",
    [
        (-0.20, -0.20, "genus"),
        (-0.01, 0.02, "domain"),
        (-0.20, -0.01, "the package and the congener interact"),
        (-0.01, -0.20, "the package and the congener interact"),
        (0.20, 0.20, "removing M. leonina helps under both trunks; unexpected, needs its own explanation"),
        (0.20, -0.20, "unresolved"),
        (float("nan"), -0.20, "not scored"),
    ],
)
def test_the_2x2_classification_follows_the_freeze(delta_z, delta_b, expected):
    assert E4.classify_2x2(delta_z, delta_b) == expected


# The noise-scale edges.
@pytest.mark.parametrize("magnitude,is_resolved", [(NOISE - 1e-4, False), (NOISE, False), (NOISE + 1e-4, True)])
@pytest.mark.parametrize("sign", [1, -1])
def test_resolution_at_the_noise_scale_edges(magnitude, is_resolved, sign):
    assert bool(E4.E3.resolved(sign * magnitude)) is is_resolved


def test_a_gap_of_exactly_the_noise_scale_reads_as_domain_not_genus():
    # abs(delta) > 0.05 resolves, so exactly -0.05 on both trunks is unresolved and reads as domain
    assert E4.classify_2x2(-NOISE, -NOISE) == "domain"
    assert E4.classify_2x2(-(NOISE + 1e-4), -(NOISE + 1e-4)) == "genus"


@pytest.mark.parametrize(
    "deltas,expected",
    [
        ({"a": -0.20, "b": -0.01, "c": 0.01}, "the package does not pay at 26 sources"),
        ({"a": 0.20, "b": 0.01, "c": -0.01}, "the package needs pool diversity, not pool size"),
        ({"a": 0.20, "b": -0.20, "c": 0.0}, "mixed and species-specific; the package verdict is unresolved, not split"),
        ({"a": 0.01, "b": -0.02, "c": 0.0}, "unresolved: the package is neutral at this scale"),
        ({"a": NOISE, "b": -NOISE, "c": 0.0}, "unresolved: the package is neutral at this scale"),
    ],
)
def test_the_package_classification_follows_the_freeze(deltas, expected):
    assert E4.classify_package(deltas) == expected


@pytest.mark.parametrize(
    "deltas,expected",
    [
        ({"a": 0.20, "b": 0.20, "c": 0.0}, "scaling works"),
        ({"a": 0.01, "b": -0.02, "c": 0.0}, "going from 3 to 9 to 26 sources changes nothing"),
        ({"a": -0.20, "b": -0.01, "c": 0.0}, "more sources hurt"),
        ({"a": 0.20, "b": -0.20, "c": 0.0}, "unresolved"),
        ({"a": NOISE, "b": NOISE, "c": NOISE}, "going from 3 to 9 to 26 sources changes nothing"),
    ],
)
def test_the_scaling_classification_follows_the_freeze(deltas, expected):
    assert E4.classify_scaling(deltas) == expected


# The dilution diagnostic.
def test_dilution_reads_records_only_and_pairs_loss_against_draw_share(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    payload = json.loads(T4.REGISTERED_POOL.read_text())
    marine = sorted(payload["sub_pools"]["domain_marine"])
    # first half of the history is discarded, so only the second half sets the answer
    history = [dict.fromkeys(marine, 0.0)] * 4 + [{name: 1.0 / len(marine) for name in marine}] * 4
    write_records(tmp_path, [record(f"ZdM_{NES}_s{seed}", "ZdM", species=NES, seed=seed, shares=history)
                             for seed in range(5)])

    table = E4.dilution_table()
    block = table["arms"]["ZdM"]
    assert block["pool"] == "sub_pools.domain_marine" and block["n_seeds"] == 5
    assert {row["species"] for row in block["rows"]} == set(marine)
    for row in block["rows"]:
        assert row["loss_share"] == pytest.approx(1.0 / len(marine))
        assert row["draw_share"] == pytest.approx(payload["sub_pools"]["domain_marine"][row["species"]])
        assert row["loss_minus_draw"] == pytest.approx(row["loss_share"] - row["draw_share"])


def test_dilution_covers_the_six_arms_the_freeze_names_and_reports_missing_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    write_records(tmp_path, [])
    table = E4.dilution_table()
    assert set(table["arms"]) == set(E4.DILUTION_ARMS)
    assert E4.DILUTION_ARMS == ("Z26", "B26", "ZdM", "BdM", "ZdMnoMir", "BdMnoMir")
    for block in table["arms"].values():
        assert block["error"] == "no records on disk"


def test_loss_shares_averages_the_last_half_only():
    history = [{"a": 1.0}] * 6 + [{"a": 0.0}] * 6
    assert E4.loss_shares({"loss_shares": history})["a"] == pytest.approx(0.0)
    assert E4.loss_shares({"loss_shares": []}) == {}


# Carried comparators.
def test_every_carried_arm_the_freeze_lists_is_named():
    for arm in ("P", "V", "S", "Zd1", "F9", "F3c", "Sk", "SB", "LSTMS", "XGBS", "ARIMA", "AVG3", "AVG10"):
        assert arm in E4.CARRIED_TIER2 + E4.CARRIED_TIER3, arm
    for arm in ("Z9", "B9", "Za9", "Zd3", "B9dom", "LSTMB3", "LSTMB9", "XGBB3", "XGBB9", "E1embed_dom"):
        assert arm in E4.CARRIED_TIER3, arm


def test_no_tier4_arm_is_also_a_carried_arm():
    assert not set(T4.ARMS) & (set(E4.CARRIED_TIER2) | set(E4.CARRIED_TIER3))


@pytest.mark.parametrize(
    "tag,arm,seed,k,draw",
    [
        ("tier2:P", "P", None, None, None),
        ("tier2:Z_s3", "Z", 3, None, None),
        ("tier3:Z9_s0", "Z9", 0, None, None),
        ("tier3:E1embed_dom_s4", "E1embed_dom", 4, None, None),
        ("tier3:Za9_canis_lupus_s2", "Za9", 2, None, None),
        (f"tier3:F9_{NES}_k010_d1", "F9", None, 10, 1),
        (f"tier3:Sk_{NES}_k001_d0", "Sk", None, 1, 0),
        ("tier3:ARIMA", "ARIMA", None, None, None),
        # as Tier 3's subset files are actually named, with the seed on the end 
        (f"tier3:F9_{NES}_k010_d1_s0", "F9", 0, 10, 1),
        (f"tier3:Sk_{NES}_k001_d0_s0", "Sk", 0, 1, 0),
        ("tier3:F3c_canis_lupus_k100_d0_s0", "F3c", 0, 100, 0),
    ],
)
def test_carried_tags_parse_to_their_arm_seed_and_draw(tag, arm, seed, k, draw):
    parsed = E4.parse_tag(tag, {})
    assert parsed == {"arm": arm, "seed": seed, "k": k, "draw": draw}


def test_tier2s_own_fine_tuned_arm_is_not_carried():
    # the freeze carries Tier 3's F9 and F3c, not Tier 2's F, which ran at the old 2,000-step budget
    assert "F" not in E4.CARRIED_TIER2 and "F" not in E4.CARRIED_TIER3
    assert E4.parse_tag("tier2:F_k001_d2", {}) is None


def test_a_tier4_tag_parses_from_its_record():
    records = {"F26_canis_lupus_k001_d2_s0": record(
        "F26_canis_lupus_k001_d2_s0", "F26", species="canis_lupus", k=1, draw=2
    )}
    assert E4.parse_tag("F26_canis_lupus_k001_d2_s0", records) == {
        "arm": "F26", "seed": 0, "k": 1, "draw": 2
    }


def test_an_unknown_tag_is_ignored_rather_than_guessed(capsys):
    assert E4.parse_tag("tier3:SomethingElse_s0", {}) is None
    assert "ignored" in capsys.readouterr().out


# Report shape.
def test_the_report_declares_itself_exploratory_with_no_success_bands(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    monkeypatch.setattr(E4, "species_frames", lambda species: (_ for _ in ()).throw(SystemExit("stop")))
    with pytest.raises(SystemExit):
        E4.build_report()
    # the banner and the no-bands statement are constants, checked directly
    assert "Exploratory" in E4.BANNER and "not a tiered result" in E4.BANNER


def test_the_markdown_opens_with_the_exploratory_banner(tmp_path):
    report = {
        "evaluation_floor": 25, "noise_scale": NOISE, "banner": E4.BANNER,
        "comparators_read_from": ["results/tier2_sealed/predictions"],
        "completeness": {"complete": True, "n_planned": 91, "n_records": 91,
                         "missing_records": [], "missing_checkpoints": []},
        "species": {}, "across_species": {
            "2_package": {"reading": "unresolved", "per_species": {}},
            "3_scaling": {"reading": "unresolved", "per_species": {}},
        },
        "ladder": {"blocks": {}}, "dilution": {"arms": {}},
    }
    path = tmp_path / "report.md"
    E4.write_markdown(report, path)
    text = path.read_text()
    assert text.startswith("**Exploratory — not a tiered result.**")
    assert "No success bands are applied" in text
    assert "All 91 planned runs" in text


def test_the_markdown_states_a_completeness_gap(tmp_path):
    report = {
        "evaluation_floor": 25, "noise_scale": NOISE, "banner": E4.BANNER,
        "comparators_read_from": ["results/tier3_sealed/predictions"],
        "completeness": {"complete": False, "n_planned": 91, "n_records": 88,
                         "missing_records": ["F26_canis_lupus_k100_d0_s0"], "missing_checkpoints": []},
        "species": {}, "across_species": {
            "2_package": {"reading": "unresolved", "per_species": {}},
            "3_scaling": {"reading": "unresolved", "per_species": {}},
        },
        "ladder": {"blocks": {}}, "dilution": {"arms": {}},
    }
    path = tmp_path / "report.md"
    E4.write_markdown(report, path)
    text = path.read_text()
    assert "Reporting-completeness gap" in text
    assert "F26_canis_lupus_k100_d0_s0" in text


def test_the_four_way_table_uses_each_species_own_domain_arm():
    arms = arms_with({"Z26": 0.1, "Za26": 0.2, "ZdF": 0.3, "ZdT": 0.4, "ZdM": 0.5, "Zd1": 0.6})
    assert E4.four_way(arms, "larus_fuscus")["same_domain_arm"] == "ZdF"
    assert E4.four_way(arms, "canis_lupus")["same_domain_arm"] == "ZdT"
    assert E4.four_way(arms, NES)["same_domain_arm"] == "ZdM"
    table = E4.four_way(arms, NES)
    assert "ZdM" in table["travel_p90_skill"] and "ZdF" not in table["travel_p90_skill"]


# Carried tags decoded through summarise_all. 
# GenAI reference: GenAI wrote these tests after job 179924605's report dropped every carried Tier 3 arm; they drive the real E3.summarise_all with S2.summarise stubbed.
CARRIED_TAGS = {
    "tier2:P": ("P", "by_seed", -1),
    "tier2:Z_s0": ("Z", "by_seed", 0),
    "tier3:Z9_s0": ("Z9", "by_seed", 0),
    "tier3:ARIMA": ("ARIMA", "by_seed", -1),
    f"tier3:Zd1_{NES}_s0": ("Zd1", "by_seed", 0),
    f"tier3:Sk_{NES}_k001_d0_s0": ("Sk", "by_draw", (1, 0)),
    f"tier3:F9_{NES}_k010_d1_s0": ("F9", "by_draw", (10, 1)),
    "tier3:SB_s0": ("SB", "by_seed", 0),
    "tier3:E1embed_dom_s0": ("E1embed_dom", "by_seed", 0),
    "Z26_s0": ("Z26", "by_seed", 0),
}


@pytest.fixture
def echo_summarise(monkeypatch):
    """Stub the metric so each summary is the frame it was given, which here is the tag itself."""
    monkeypatch.setattr(E4.E3.S2, "summarise", lambda frame, persistence, boundary, floor: frame)


def test_summarise_all_places_every_carried_tag_with_tier4s_decoder(echo_summarise, capsys):
    frames = {tag: tag for tag in CARRIED_TAGS}
    records = {"Z26_s0": record("Z26_s0", "Z26")}
    out = E4.E3.summarise_all(frames, None, 1.0, records, parse=E4.parse_tag)
    assert "ignored" not in capsys.readouterr().out
    assert set(out) == {arm for arm, _, _ in CARRIED_TAGS.values()}
    for tag, (arm, table, key) in CARRIED_TAGS.items():
        assert out[arm][table][key] == tag, tag
        assert out[arm]["tags"][key] == tag, tag


def test_summarise_all_without_parse_still_uses_tier3s_decoder(echo_summarise, monkeypatch, capsys):
    seen = []
    original = E4.E3.parse_tag

    def spy(tag, records):
        seen.append(tag)
        return original(tag, records)

    monkeypatch.setattr(E4.E3, "parse_tag", spy)
    out = E4.E3.summarise_all({"tier2:P": "tier2:P", "tier3:Z9_s0": "tier3:Z9_s0"}, None, 1.0, {})
    assert seen == ["tier2:P", "tier3:Z9_s0"]
    assert set(out) == {"P"}  # Tier 3's own decoder, unchanged: it does not know the tier3: prefix
    assert "ignored tier3:Z9_s0" in capsys.readouterr().out


def test_missing_carried_names_each_absent_arm_for_the_species():
    carried = set(E4.CARRIED_TIER2) | set(E4.CARRIED_TIER3)
    assert E4.missing_carried({arm: {} for arm in carried}, NES) == []
    assert E4.missing_carried({arm: {} for arm in carried - {"Z9", "Sk"}}, NES) == [("Sk", NES), ("Z9", NES)]


def test_build_report_stops_when_a_carried_arm_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(E4, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(E4, "TIER4_RESULTS", tmp_path)
    monkeypatch.setattr(E4, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(E4, "species_frames", lambda species: (None, {}, {}))
    monkeypatch.setattr(E4.T, "boundary_m", lambda key: 1.0)
    calls = []

    def summaries(frames, persistence, boundary, records, parse=None):
        calls.append(parse)
        return {arm: {} for arm in (set(E4.CARRIED_TIER2) | set(E4.CARRIED_TIER3)) - {"Z9"}}

    monkeypatch.setattr(E4.E3, "summarise_all", summaries)
    with pytest.raises(SystemExit, match="Z9 / larus_fuscus"):
        E4.build_report()
    assert calls == [E4.parse_tag]


# Collisions, draw completeness and the placement log (27-09-2026 correction).
def _draw_summaries(species, drop=(), arms=E4.SUBSET_ARMS):
    return {
        arm: {"by_seed": {}, "by_draw": {key: {} for key in E4.registered_draws(arm, species) if key not in drop}}
        for arm in arms
    }


def test_registered_draws_are_read_off_the_plans():
    tier3_style = {(1, 0), (1, 1), (1, 2), (10, 0), (10, 1), (10, 2)}
    assert E4.registered_draws("Sk", "larus_fuscus") == tier3_style
    assert E4.registered_draws("F9", "larus_fuscus") == tier3_style | {(100, 0)}
    assert E4.registered_draws("F3c", NES) == tier3_style | {(100, 0)}
    assert E4.registered_draws("F26", "canis_lupus") == tier3_style | {(100, 0)}


def test_draw_completeness_passes_on_the_registered_sets():
    for species in ("larus_fuscus", "canis_lupus", NES):
        E4.assert_draws_complete(_draw_summaries(species), species)


def test_draw_completeness_fails_when_sk_misses_a_draw():
    with pytest.raises(SystemExit, match=r"Sk / larus_fuscus: missing \[\(10, 2\)\]"):
        E4.assert_draws_complete(_draw_summaries("larus_fuscus", drop={(10, 2)}), "larus_fuscus")


def test_draw_completeness_fails_when_sk_sits_under_a_seed_instead():
    summaries = _draw_summaries(NES)
    summaries["Sk"] = {"by_seed": {0: {}}, "by_draw": {}}  # what the pre-correction decoder produced
    with pytest.raises(SystemExit, match=f"Sk / {NES}: missing"):
        E4.assert_draws_complete(summaries, NES)


def test_two_tags_in_one_slot_are_a_collision():
    frames = {f"tier3:Sk_{NES}_k001_d0_s0": None, f"tier3:Sk_{NES}_k001_d0": None, "tier2:P": None}
    with pytest.raises(SystemExit, match=r"same summary slot"):
        E4.assert_no_collisions(frames, {})


def test_distinct_slots_are_not_a_collision():
    frames = {tag: None for tag in CARRIED_TAGS}
    E4.assert_no_collisions(frames, {"Z26_s0": record("Z26_s0", "Z26")})


def test_the_placement_log_is_one_line_per_arm():
    summaries = {
        "P": {"by_seed": {-1: {}}, "by_draw": {}},
        "Z9": {"by_seed": {s: {} for s in range(5)}, "by_draw": {}},
        "Sk": {"by_seed": {}, "by_draw": {(1, 0): {}, (1, 1): {}}},
    }
    lines = E4.placement_lines(summaries, NES)
    assert len(lines) == 3
    assert "P" in lines[0] and "seeds  0" in lines[0] and "one unseeded run" in lines[0]
    assert "Z9" in lines[1] and "seeds  5" in lines[1] and "draws  0" in lines[1]
    assert "Sk" in lines[2] and "draws  2" in lines[2]
