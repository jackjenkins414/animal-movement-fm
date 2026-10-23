# GenAI reference: I asked GenAI to draft a test file for the file it just completed; that being and didn't give it an specification; I wanted 
# to see how well it could implement one without my input. It was correct and tested things like subset arms, suffix split, interval harvesting etc. 
"""Tests for experiments/reporting/metric_tables.py, on synthetic report JSONs only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "reporting"))

import metric_tables as MT  # noqa: E402


def stats(p90, mean=None, p50=None, n_windows=None):
    out = {
        "p90_m": p90,
        "mean_m_individual_weighted": p90 / 2 if mean is None else mean,
        "p50_m": p90 / 4 if p50 is None else p50,
        "p99_m_individual_weighted": p90 * 2,
        "mean_m_pooled": p90 / 2,
        "p99_m_pooled": p90 * 2,
    }
    if n_windows is not None:
        out["n_windows"] = n_windows
    return out


def arm(p90_travel, p90_all, reference=None, seeds=3, travel_mean=None, travel_p50=None):
    headline = {"travel": stats(p90_travel, travel_mean, travel_p50), "low_displacement": stats(10.0),
                "all": stats(p90_all)}
    skill = {stratum: {name: (1 - value / reference[stratum][name]) if reference else 0.0
                       for name, value in block.items()}
             for stratum, block in headline.items()}
    per_seed = {"0": {"n_individuals": 7, "travel": {"n_windows": 100}, "all": {"n_windows": 400}}} if seeds else {}
    return {"n_seeds": seeds, "headline": headline, "skill": skill, "per_seed": per_seed, "tags": {}}


def comparison(left, right, bands: dict, skill_bands: dict | None = None):
    out = {"left": left, "right": right, "bootstrap": {"series": {
        "travel_p90_m": {"interval": bands},
        "all_mean_m": {"interval": bands},
    }}}
    if skill_bands:
        out["skill_bootstrap"] = {"skill": skill_bands}
    return out


def p_arm():
    return arm(1000.0, 500.0, seeds=None)


def reference():
    return p_arm()["headline"]


# Tier 2
def test_tier2_skill_is_read_flattened_and_p_shows_as_ref():
    report = {"species": {"larus_fuscus": {
        "regime_boundary_m": 50.0,
        "comparators": {
            "P": {"n_seeds": None, "headline": reference(), "per_seed": {}, "untrained": True},
            "Z": {"n_seeds": 5, "headline": arm(800.0, 450.0)["headline"], "per_seed": {},
                  "persistence_skill": {"travel_p90_m": 0.2, "all_p50_m": 0.1}},
        },
        "bootstrap": {"series": {"travel_p90_m": {"interval": {"P": {"lo": 900, "hi": 1100},
                                                               "Z": {"lo": 700, "hi": 900}}}}},
    }}}
    unit = MT.study_units("tier2", report)[0]
    assert unit["arms"]["Z"]["metrics"]["travel_p90"] == {"m": 800.0, "skill": 0.2}
    assert unit["arms"]["Z"]["metrics"]["travel_mean"]["skill"] is None  # not stored, not recomputed
    assert unit["arms"]["P"]["metrics"]["travel_p90"]["skill"] is None
    assert MT._cell("P", unit["arms"]["P"]["metrics"]["travel_p90"]) == "1,000 (ref)"
    assert unit["arms"]["Z"]["intervals"]["travel_p90_m"]["lo"] == 700
    assert unit["arms"]["Z"]["intervals"]["all_mean_m"] is None


# Arm-block family
def test_subset_arms_are_listed_per_draw_and_kept_out_of_headline_and_intervals():
    draw = {"k_percent": 1, "draw": 0, "headline": arm(900.0, 480.0)["headline"],
            "skill": arm(900.0, 480.0, reference())["skill"]}
    report = {"species": {"canis_lupus": {
        "arms": {"P": p_arm(), "S": arm(700.0, 400.0, reference()),
                 "Sk": {"n_seeds": None, "headline": {}, "skill": {}, "per_seed": {}, "by_draw": {"k001_d0": draw}}},
        "readings": {"x": comparison("S", "P", {"S": {"lo": 1, "hi": 2}, "P": {"lo": 3, "hi": 4}}),
                     "few_shot": {"draws": {"d0": comparison("Sk", "S", {"Sk": {"lo": 9, "hi": 9},
                                                                         "S": {"lo": 99, "hi": 99}})}}},
    }}}
    unit = MT.study_units("tier3", report)[0]
    assert list(unit["arms"]) == ["P", "S"]
    assert [(row["arm"], row["k_percent"], row["draw"]) for row in unit["subset"]] == [("Sk", 1, 0)]
    assert unit["arms"]["S"]["intervals"]["travel_p90_m"] == {"lo": 1, "hi": 2, "at": "readings/x/bootstrap"}
    assert unit["interval_conflicts"] == []  # the per-draw copy of S is skipped, not compared


def test_disagreeing_interval_copies_are_reported_not_resolved():
    tree = {"a": comparison("B", "P", {"B": {"lo": 1, "hi": 2}}),
            "b": comparison("B", "Z", {"B": {"lo": 1, "hi": 3}, "Z": {"lo": 0, "hi": 1}})}
    found, conflicts, ignored = MT.harvest(tree, {"B", "P"})
    assert found["B"]["travel_p90_m"]["hi"] == 2
    assert {c["series"] for c in conflicts} == {"travel_p90_m", "all_mean_m"}
    assert ignored == ["Z"]


# 1 h family
def hourly_report():
    birds = {}
    for name, fold, skill_shift in (("anser_anser", 1, 0.0), ("anser_fabalis", 1, 100.0), ("milvus_milvus", 6, 200.0)):
        arms = {"P": p_arm(), "BbF": arm(800.0 + skill_shift, 450.0, reference()),
                "BbF_C96": arm(700.0 + skill_shift, 420.0, reference())}
        readings = {"against_p": {"BbF_C96": comparison("BbF_C96", "P", {"BbF_C96": {"lo": 1, "hi": 2}},
                                                        {"BbF_C96": {"lo": 0.1, "hi": 0.2}})}}
        birds[name] = {"fold": fold, "horizons": {
            "6": {"boundary_m": 40.0, "scored": True, "arms": arms, "readings": readings},
            "24": {"boundary_m": 40.0, "scored": name != "milvus_milvus",
                   "arms": arms if name != "milvus_milvus" else {}, "readings": {}},
        }}
    return {"species": birds, "across_birds": {
        "common_set": ["anser_anser", "anser_fabalis"], "selection_fold": 6,
        "horizons": {"6": {"scored": list(birds), "skill": {"BbF": {"scored": {"n": 3, "median": 0.1}}}},
                     "24": {"scored": ["anser_anser", "anser_fabalis"], "skill": {}}},
    }}


def test_c_suffix_splits_units_and_p_serves_both():
    units = MT.study_units("birds_1h_bbf", hourly_report())
    six = [u for u in units if u["species"] == "anser_anser" and u["horizon"] == "6"]
    assert [(u["C"], list(u["arms"])) for u in six] == [(12, ["P", "BbF"]), (96, ["P", "BbF"])]
    assert six[1]["arms"]["BbF"]["report_arm"] == "BbF_C96"
    assert six[1]["arms"]["BbF"]["intervals"]["travel_p90_skill"]["hi"] == 0.2
    assert six[0]["arms"]["BbF"]["intervals"]["travel_p90_skill"] is None
    assert all(u["ignored_interval_labels"] == [] for u in six)
    unscored = [u for u in units if u["species"] == "milvus_milvus" and u["horizon"] == "24"]
    assert len(unscored) == 1 and unscored[0]["not_scored"]


def test_bird_medians_use_the_report_sets_and_check_its_median():
    report = hourly_report()
    table = MT.build("birds_1h_bbf", report)
    six = table["across_species"]["horizons"]["6"]
    assert six["sets"] == {"scored": 3, "common": 2, "scored_without_fold6": 2, "common_without_fold6": 2}
    entry = six["arms"]["BbF"]["scored"]
    assert entry["travel_p90"]["n"] == 3
    assert entry["travel_p90"]["median"] == pytest.approx(0.1)  # skills 0.2, 0.1, 0.0
    assert entry["matches_report"] is True
    assert six["arms"]["BbF"]["scored_without_fold6"]["travel_p90"]["median"] == pytest.approx(0.15)


def test_ablation_is_skill_only_with_report_check():
    report = hourly_report()
    for bird in report["species"].values():
        bird["horizons"]["6"]["ablation"] = {"24": {"all": {"n_individuals": 6, "p_p90_m": 1000.0,
                                                            "LSTMBb": {"skill": 0.05, "skill_C12": 0.01}}}}
    report["ablation"] = {"LSTMBb": {"6": {"24": {"all": {"scored": {"skill": {"n": 3, "median": 0.05}}}}}}}
    block = MT.ablation(report)
    assert {row["arm"] for row in block["rows"]} == {"LSTMBb"}
    assert set(block["rows"][0]) == {"bird", "horizon", "C", "subset", "arm", "skill", "skill_C12"}
    cell = block["median_over_scored_birds"]["6"]["LSTMBb"]
    assert cell["24"]["matches_report"] is True
    assert cell["12"]["median"] == pytest.approx(0.01)


# Ranking check
def test_ranking_reports_order_agreement_without_a_verdict():
    unit = {"arms": {
        "P": {"metrics": MT.metrics_of(p_arm(), "arms")},
        "A": {"metrics": MT.metrics_of(arm(800.0, 450.0, reference(), travel_mean=300.0), "arms")},
        "B": {"metrics": MT.metrics_of(arm(900.0, 480.0, reference(), travel_mean=200.0), "arms")},
    }}
    result = MT.ranking(unit)
    assert result["order"]["travel_p90"] == ["A", "B", "P"]
    assert result["order"]["travel_mean"] == ["B", "A", "P"]
    mean = result["against_travel_p90"]["travel_mean"]
    assert (mean["same_order"], mean["same_top"], mean["top"]) == (False, False, "B")
    assert mean["spearman"] == pytest.approx(0.5)
    assert result["against_travel_p90"]["travel_p50"]["same_order"] is True
    assert "pass" not in str(result).lower()


def test_ranking_needs_two_complete_arms():
    unit = {"arms": {"P": {"metrics": MT.metrics_of(p_arm(), "arms")}}}
    assert "error" in MT.ranking(unit)


def test_studies_read_report_json_only():
    for path, _ in MT.STUDIES.values():
        assert path.endswith(".json") and "_sealed/" in path
