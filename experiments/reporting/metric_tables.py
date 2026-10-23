# GenAI reference: I began implementing this file myself; imports, studies, metrics, etc, did some functions myself like metrics, load_report
# etc, but got GenAI to help me complete some more complex ones like harvet, which used recursive tree traversal, nested helpers, and others. 
# I checked its output and behaviour and it was correct. 
"""
Metric tables from the sealed report JSONs (descriptive).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = RESULTS / "metric_tables"

# study -> (sealed report JSON under results/, layout family)
STUDIES: dict[str, tuple[str, str]] = {
    "tier2": ("tier2_sealed/tier2_verdict.json", "tier2"),
    "tier3": ("tier3_sealed/tier3_results.json", "arms"),
    "tier4": ("tier4_sealed/tier4_results.json", "arms"),
    "birds": ("birds_sealed/birds_results.json", "arms"),
    "birds_rotation": ("birds_rotation_sealed/birds_rotation_results.json", "arms"),
    "pooling": ("pooling_sealed/pooling_results.json", "arms"),
    "tier4_repaired": ("tier4_repaired_sealed/tier4_repaired_results.json", "arms"),
    "birds_1h": ("birds_1h_sealed/birds_1h_results.json", "hourly"),
    "birds_1h_env": ("birds_1h_env_sealed/birds_1h_env_results.json", "hourly"),
    "birds_1h_bbf": ("birds_1h_bbf_sealed/birds_1h_bbf_results.json", "hourly"),
}

# (label, stratum, statistic name as in tier2_sealed_eval.STATISTICS); the first is the registered headline
METRICS = (
    ("travel_p90", "travel", "p90_m"),
    ("travel_mean", "travel", "mean_m_individual_weighted"),
    ("travel_p50", "travel", "p50_m"),
    ("all_mean", "all", "mean_m_individual_weighted"),
    ("all_p50", "all", "p50_m"),
)
# carried in the JSON output only
POOLED = (
    ("travel_mean_pooled", "travel", "mean_m_pooled"),
    ("all_mean_pooled", "all", "mean_m_pooled"),
)
TITLES = {
    "travel_p90": "travel p90",
    "travel_mean": "travel mean",
    "travel_p50": "travel p50",
    "all_mean": "all mean",
    "all_p50": "all p50",
}

# the bootstrap series the evaluators store per arm: two in metres (BOOTSTRAP_SERIES) and Resampler's skill one
INTERVALS = ("travel_p90_m", "all_mean_m", "travel_p90_skill")
# subtrees that hold per-draw comparisons or per-arm statistics, never a headline arm's interval
SKIP = {"draws", "by_draw", "per_draw", "per_seed", "arms", "comparators", "ablation", "floor_sensitivity", "seeds_0_2"}

C_SUFFIX = re.compile(r"^(?P<arm>.+)_C(?P<c>\d+)$")
MAIN_C = 12

NOTES = {
    "tier2": [
        "P's skill is not stored in tier2_verdict.json (P is the reference); shown as `ref`.",
        "F is not in tier2_verdict.json; tier2_finetune_curve.py reports it.",
    ],
    "birds_rotation": ["The descriptive `seeds_0_2` block (fold 0 at seeds 0-2) is not tabulated."],
    "birds_1h": [
        "The context ablation stores travel p90 skill (and P's travel p90) only; no other statistic, no arm metres "
        "and no interval. Metres are not backed out of skill.",
    ],
    "birds_1h_bbf": ["C is read from the `_C96` arm suffix; arms without one are at C = 12. P serves both."],
}
SUBSET_NOTE = "Subset arms are listed per (k, draw) as stored; their per-draw intervals sit in the report's readings."


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def short(name: str) -> str:
    genus, _, epithet = name.partition("_")
    return f"*{genus[0].upper()}. {epithet}*" if epithet else name


def load_report(study: str, results: Path = RESULTS) -> dict:
    path = results / STUDIES[study][0]
    if path.suffix != ".json":
        raise SystemExit(f"FAIL: {path} is not a report JSON")
    if not path.exists():
        raise SystemExit(f"FAIL: {path} does not exist")
    return json.loads(path.read_text())


# Arm values
def metrics_of(block: dict, family: str) -> dict:
    """
    The stored metres and skill of one arm (or one subset draw) for every tabulated statistic.
    """
    headline = block.get("headline") or {}
    out = {}
    for label, stratum, name in (*METRICS, *POOLED):
        value = (headline.get(stratum) or {}).get(name)
        if family == "tier2":
            skill = (block.get("persistence_skill") or {}).get(f"{stratum}_{name}")
        else:
            skill = ((block.get("skill") or {}).get(stratum) or {}).get(name)
        out[label] = {"m": value if finite(value) else None, "skill": skill if finite(skill) else None}
    return out


def has_headline(block: dict) -> bool:
    return any(finite(v) for v in ((block.get("headline") or {}).get("travel") or {}).values())


def window_counts(arms: dict) -> dict | None:
    """Individuals at the floor and windows per stratum, from any per-seed block (they belong to the stratum)."""
    for block in arms.values():
        for payload in (block.get("per_seed") or {}).values():
            return {
                "n_individuals": payload.get("n_individuals"),
                "travel_windows": (payload.get("travel") or {}).get("n_windows"),
                "all_windows": (payload.get("all") or {}).get("n_windows"),
            }
    return None


# Intervals
def harvest(tree: dict, names: set[str]) -> tuple[dict, list[dict], list[str]]:
    """
    Every interval the report stored for a headline arm, found in its comparison blocks.

    Returns {arm: {series: {lo, hi, at}}}, the copies that disagree with the first one found, and labels that are not
    headline arms here. Each arm's interval comes from the same draws on the same frame, so copies should agree.
    """
    found: dict[str, dict] = {}
    conflicts: list[dict] = []
    ignored: set[str] = set()

    def add(label: str, series: str, band: dict, where: str) -> None:
        if label not in names:
            ignored.add(label)
            return
        lo, hi = band.get("lo"), band.get("hi")
        if not (finite(lo) and finite(hi)):
            return
        slot = found.setdefault(label, {})
        if series not in slot:
            slot[series] = {"lo": lo, "hi": hi, "at": where}
            return
        kept = slot[series]
        if not (math.isclose(kept["lo"], lo, rel_tol=1e-9, abs_tol=1e-9)
                and math.isclose(kept["hi"], hi, rel_tol=1e-9, abs_tol=1e-9)):
            conflicts.append({"arm": label, "series": series, "kept": kept, "other": {"lo": lo, "hi": hi, "at": where}})

    def walk(node, path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        series = node.get("series")
        if isinstance(series, dict):
            for key in ("travel_p90_m", "all_mean_m"):
                for label, band in ((series.get(key) or {}).get("interval") or {}).items():
                    add(label, key, band, "/".join(path))
        boot = node.get("skill_bootstrap")
        if isinstance(boot, dict):
            for label, band in (boot.get("skill") or {}).items():
                add(label, "travel_p90_skill", band, "/".join([*path, "skill_bootstrap"]))
        for key, child in node.items():
            if key not in SKIP and key != "skill_bootstrap":
                walk(child, [*path, str(key)])

    walk(tree, [])
    return found, conflicts, sorted(ignored)


# Units: one species (or bird, horizon and C) with its arms
def make_unit(family: str, species: str, horizon, c, meta: dict, arms: dict, tree: dict, rename=None,
              harvested: tuple | None = None) -> dict:
    """
    Tabulate one unit's headline arms, its subset draws and the intervals stored for its arms.

    `harvested` passes in a harvest already taken over a wider block (all C at one bird and horizon).
    """
    rename = rename or (lambda arm: arm)
    headline = {arm: block for arm, block in arms.items() if has_headline(block)}
    intervals, conflicts, ignored = harvested or harvest(tree, set(headline))
    unit = {
        "species": species,
        "horizon": horizon,
        "C": c,
        "meta": {**meta, **(window_counts(arms) or {})},
        "arms": {},
        "subset": [],
        "interval_conflicts": conflicts,
        "ignored_interval_labels": ignored,
        "not_scored": False,
    }
    for arm, block in headline.items():
        unit["arms"][rename(arm)] = {
            "report_arm": arm,
            "n_seeds": block.get("n_seeds"),
            "metrics": metrics_of(block, family),
            "intervals": {series: intervals.get(arm, {}).get(series) for series in INTERVALS},
        }
    for arm, block in arms.items():
        for key, draw in sorted((block.get("by_draw") or {}).items()):
            unit["subset"].append({
                "arm": arm, "k_percent": draw.get("k_percent"), "draw": draw.get("draw"), "key": key,
                "metrics": metrics_of(draw, family),
            })
    return unit


def hourly_units(species: str, entry: dict, horizon: str, block: dict) -> list[dict]:
    """One unit per C at this bird and horizon; P sits in each."""
    meta = {"fold": entry.get("fold"), "boundary_m": block.get("boundary_m")}
    arms = block.get("arms") or {}
    if not block.get("scored") or not arms:
        return [{"species": species, "horizon": horizon, "C": MAIN_C, "meta": meta, "arms": {}, "subset": [],
                 "interval_conflicts": [], "ignored_interval_labels": [], "not_scored": True}]
    groups: dict[int, dict] = {}
    for arm, arm_block in arms.items():
        match = C_SUFFIX.match(arm)
        groups.setdefault(int(match["c"]) if match else MAIN_C, {})[arm] = arm_block
    # one harvest over every C; conflicts and ignored labels are carried by the first unit only
    intervals, conflicts, ignored = harvest(block, {arm for arm, b in arms.items() if has_headline(b)})
    units = []
    for c in sorted(groups):
        members = groups[c] if c == MAIN_C else {"P": arms["P"], **groups[c]}
        first = not units
        units.append(make_unit("hourly", species, horizon, c, meta, members, block,
                               rename=lambda arm: C_SUFFIX.sub(r"\g<arm>", arm),
                               harvested=(intervals, conflicts if first else [], ignored if first else [])))
    return units


def study_units(study: str, report: dict) -> list[dict]:
    family = STUDIES[study][1]
    units = []
    for species, entry in report["species"].items():
        if family == "hourly":
            for horizon, block in entry["horizons"].items():
                units += hourly_units(species, entry, horizon, block)
            continue
        arms = entry.get("comparators" if family == "tier2" else "arms") or {}
        meta = {"fold": entry.get("fold"), "domain": entry.get("domain"), "boundary_m": entry.get("regime_boundary_m")}
        units.append(make_unit(family, species, None, None, meta, arms, entry))
    return units


def unit_label(unit: dict) -> str:
    parts = [short(unit["species"])]
    if unit["horizon"] is not None:
        parts.append(f"{unit['horizon']} h")
    if unit["C"] is not None and unit["C"] != MAIN_C:
        parts.append(f"C = {unit['C']}")
    return ", ".join(parts)


# Ranking check
def order(arms: list[str], values: list[float]) -> list[str]:
    """Lowest error first; ties by arm name."""
    return [arm for _, arm in sorted(zip(values, arms))]


def ranking(unit: dict) -> dict:
    """Does the arm order under travel p90 match the order under each other statistic? Descriptive only."""
    rows = {arm: block["metrics"] for arm, block in unit["arms"].items()}
    arms = [arm for arm, m in rows.items() if all(m[label]["m"] is not None for label, _, _ in METRICS)]
    out: dict = {"n_arms": len(arms), "excluded": sorted(set(rows) - set(arms))}
    if len(arms) < 2:
        out["error"] = "fewer than two arms with all five statistics"
        return out
    base = [rows[arm]["travel_p90"]["m"] for arm in arms]
    reference = order(arms, base)
    out["order"] = {"travel_p90": reference}
    out["against_travel_p90"] = {}
    for label, _, _ in METRICS[1:]:
        values = [rows[arm][label]["m"] for arm in arms]
        other = order(arms, values)
        entry = {"same_order": other == reference, "same_top": other[0] == reference[0], "top": other[0]}
        if len(arms) >= 3:
            entry["spearman"] = float(spearmanr(base, values).statistic)
            entry["kendall"] = float(kendalltau(base, values).statistic)
        out["order"][label] = other
        out["against_travel_p90"][label] = entry
    return out


def ranking_summary(units: list[dict]) -> dict:
    checked = [unit["ranking"] for unit in units if "against_travel_p90" in unit.get("ranking", {})]
    out = {"n_units": len(checked)}
    for label, _, _ in METRICS[1:]:
        entries = [r["against_travel_p90"][label] for r in checked]
        rho = [e["spearman"] for e in entries if finite(e.get("spearman"))]
        tau = [e["kendall"] for e in entries if finite(e.get("kendall"))]
        out[label] = {
            "same_order": sum(e["same_order"] for e in entries),
            "same_top": sum(e["same_top"] for e in entries),
            "median_spearman": float(np.median(rho)) if rho else None,
            "min_spearman": float(np.min(rho)) if rho else None,
            "median_kendall": float(np.median(tau)) if tau else None,
        }
    return out


# Across-species summary
def side_by_side(units: list[dict]) -> dict:
    """Three-species studies: every arm's stored skills, species beside each other."""
    species = [unit["species"] for unit in units]
    arms: list[str] = []
    for unit in units:
        arms += [arm for arm in unit["arms"] if arm not in arms]
    return {
        "species": species,
        "arms": {
            arm: {unit["species"]: {label: (unit["arms"].get(arm) or {}).get("metrics", {}).get(label, {}).get("skill")
                                    for label, _, _ in METRICS}
                  for unit in units}
            for arm in arms
        },
    }


def median_block(values: list) -> dict:
    kept = [v for v in values if finite(v)]
    return {"n": len(kept), "median": float(np.median(kept)) if kept else None}


def bird_sets(study: str, report: dict, horizon: str | None) -> tuple[dict[str, list[str]], dict]:
    """The bird sets a summary is taken over, and the report's own travel p90 medians for them (1 h studies)."""
    species = report["species"]
    if study == "birds_rotation":
        return {"30 birds": list(species), "27 outside fold 0": [b for b, e in species.items() if e.get("fold") != 0]}, {}
    across = report.get("across_birds") or {}
    block = (across.get("horizons") or {}).get(horizon) or {}
    scored = list(block.get("scored") or [])
    common = list(across.get("common_set") or [])
    sets = {"scored": scored, "common": common}
    selection = across.get("selection_fold")
    if selection is not None:
        sets[f"scored_without_fold{selection}"] = [b for b in scored if species[b].get("fold") != selection]
        sets[f"common_without_fold{selection}"] = [b for b in common if species[b].get("fold") != selection]
    return sets, block.get("skill") or {}


def bird_medians(study: str, report: dict, units: list[dict]) -> dict:
    """Bird studies: the median over birds of each arm's stored skills, per horizon where the study has them."""
    out: dict = {}
    horizons = sorted({u["horizon"] for u in units}, key=lambda h: (h is None, int(h) if h else 0))
    for horizon in horizons:
        sets, stored = bird_sets(study, report, horizon)
        by_bird: dict[str, dict] = {}
        for unit in units:
            if unit["horizon"] != horizon:
                continue
            for block in unit["arms"].values():
                by_bird.setdefault(block["report_arm"], {})[unit["species"]] = block["metrics"]
        block_out: dict = {}
        for arm, birds in by_bird.items():
            arm_out = {}
            for name, members in sets.items():
                entry = {label: median_block([(birds.get(b) or {}).get(label, {}).get("skill") for b in members])
                         for label, _, _ in METRICS}
                reported = ((stored.get(arm) or {}).get(name) or {}).get("median")
                if reported is not None:
                    mine = entry["travel_p90"]["median"]
                    entry["report_travel_p90_median"] = reported
                    entry["matches_report"] = mine is not None and math.isclose(mine, reported, abs_tol=1e-12)
                arm_out[name] = entry
            block_out[arm] = arm_out
        out[str(horizon) if horizon is not None else "all"] = {"sets": {k: len(v) for k, v in sets.items()},
                                                               "arms": block_out}
    return out


# Context ablation (birds_1h)
def ablation(report: dict) -> dict:
    """Travel p90 skill per bird, horizon, arm and C, as stored; with medians over each horizon's scored birds."""
    rows: list[dict] = []
    for bird, entry in report["species"].items():
        for horizon, block in entry["horizons"].items():
            for c, subsets in (block.get("ablation") or {}).items():
                for subset, cell in subsets.items():
                    for arm, values in cell.items():
                        if isinstance(values, dict) and "skill" in values:
                            rows.append({"bird": bird, "horizon": horizon, "C": int(c), "subset": subset, "arm": arm,
                                         "skill": values.get("skill"), "skill_C12": values.get("skill_C12")})
    summary: dict = {}
    stored = report.get("ablation") or {}
    for horizon, block in ((report.get("across_birds") or {}).get("horizons") or {}).items():
        scored = set(block.get("scored") or [])
        cells = [r for r in rows if r["horizon"] == horizon and r["bird"] in scored and r["subset"] == "all"]
        if not cells:
            continue
        first_c = min(r["C"] for r in cells)  # skill_C12 is repeated under every C; read it once
        for arm in sorted({r["arm"] for r in cells}):
            for c in sorted({r["C"] for r in cells}):
                mine = median_block([r["skill"] for r in cells if r["arm"] == arm and r["C"] == c])
                reported = (((((stored.get(arm) or {}).get(horizon) or {}).get(str(c)) or {}).get("all") or {})
                            .get("scored") or {}).get("skill", {}).get("median")
                if reported is not None:
                    mine["report_median"] = reported
                    mine["matches_report"] = mine["median"] is not None and math.isclose(mine["median"], reported,
                                                                                         abs_tol=1e-12)
                summary.setdefault(horizon, {}).setdefault(arm, {})[str(c)] = mine
            summary[horizon][arm][str(MAIN_C)] = median_block(
                [r["skill_C12"] for r in cells if r["arm"] == arm and r["C"] == first_c])
    return {"statistic": "travel p90 skill", "rows": rows, "median_over_scored_birds": summary}


# Not stored
def not_stored(study: str, units: list[dict]) -> dict:
    scored = [unit for unit in units if not unit["not_scored"]]
    arm_rows = [block for unit in scored for block in unit["arms"].values()]
    missing_m = {label: sum(block["metrics"][label]["m"] is None for block in arm_rows) for label, _, _ in METRICS}
    missing_skill = {label: sum(block["metrics"][label]["skill"] is None for block in arm_rows) for label, _, _ in METRICS}
    present = {series: sum(block["intervals"][series] is not None for block in arm_rows) for series in INTERVALS}
    never = ["intervals on travel mean, travel p50 and all-window p50 (no evaluator bootstraps them)"]
    never += [f"{series} intervals" for series, n in present.items() if n == 0]
    return {
        "arm_rows": len(arm_rows),
        "units_not_scored": [unit_label(unit) for unit in units if unit["not_scored"]],
        "metres_missing": missing_m,
        "skill_missing": missing_skill,
        "intervals_present": present,
        "never_stored": never,
        "notes": NOTES.get(study, []) + ([SUBSET_NOTE] if any(unit["subset"] for unit in units) else []),
    }


def arm_order(units: list[dict]) -> list[str]:
    """One order for the whole study: P first, then each arm where it first appears."""
    names = ["P"]
    for unit in units:
        names += [arm for arm in unit["arms"] if arm not in names]
    return names


def build(study: str, report: dict) -> dict:
    units = study_units(study, report)
    names = arm_order(units)
    for unit in units:
        unit["arms"] = {arm: unit["arms"][arm] for arm in names if arm in unit["arms"]}
        unit["ranking"] = ranking(unit) if not unit["not_scored"] else {"error": "not scored"}
    attestation = report.get("attestation") or {}
    out = {
        "study": study,
        "source": f"results/{STUDIES[study][0]}",
        "registered_by": report.get("registered_by"),
        "freeze_commit": attestation.get("freeze_commit"),
        "evaluation_floor": report.get("evaluation_floor"),
        "seed_aggregation": report.get("seed_aggregation"),
        "units": units,
        "ranking_summary": ranking_summary(units),
        "interval_conflicts": [{"unit": unit_label(u), **c} for u in units for c in u["interval_conflicts"]],
        "ignored_interval_labels": sorted({label for u in units for label in u["ignored_interval_labels"]}),
        "not_stored": not_stored(study, units),
    }
    if len(report["species"]) <= 3 and STUDIES[study][1] != "hourly":
        out["across_species"] = {"form": "side_by_side", **side_by_side(units)}
    else:
        out["across_species"] = {"form": "median_over_birds", "horizons": bird_medians(study, report, units)}
    if study == "birds_1h":
        out["ablation"] = ablation(report)
    return out


# Markdown
def _m(value) -> str:
    if not finite(value):
        return "—"
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.1f}"


def _s(value) -> str:
    return f"{value:+.4f}" if finite(value) else "—"


def _band(band: dict | None, unit: str) -> str:
    if not band:
        return "—"
    return f"[{_m(band['lo'])}, {_m(band['hi'])}]" if unit == "m" else f"[{band['lo']:+.4f}, {band['hi']:+.4f}]"


def _cell(arm: str, entry: dict) -> str:
    if entry["m"] is None:
        return "not stored"
    skill = _s(entry["skill"]) if entry["skill"] is not None else ("ref" if arm == "P" else "not stored")
    return f"{_m(entry['m'])} ({skill})"


def unit_markdown(unit: dict) -> list[str]:
    meta = unit["meta"]
    facts = []
    if meta.get("fold") is not None:
        facts.append(f"fold {meta['fold']}")
    if finite(meta.get("boundary_m")):
        facts.append(f"boundary {meta['boundary_m']:,.1f} m")
    if meta.get("n_individuals") is not None:
        facts.append(f"{meta['n_individuals']} individuals at the floor")
    if meta.get("travel_windows") is not None:
        facts.append(f"{meta['travel_windows']:,} travel / {meta['all_windows']:,} windows")
    lines = [f"### {unit_label(unit)}", ""]
    if unit["not_scored"]:
        return lines + ["Not scored in the report.", ""]
    lines += ["; ".join(facts) + ".", ""] if facts else []
    lines += [
        "| arm | seeds | travel p90 | travel mean | travel p50 | all mean | all p50 | travel p90 95% (m) "
        "| travel p90 95% (skill) | all mean 95% (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for arm, block in unit["arms"].items():
        cells = [_cell(arm, block["metrics"][label]) for label, _, _ in METRICS]
        bands = block["intervals"]
        lines.append(
            f"| {arm} | {block['n_seeds'] or '—'} | " + " | ".join(cells)
            + f" | {_band(bands['travel_p90_m'], 'm')} | {_band(bands['travel_p90_skill'], 'skill')} "
            f"| {_band(bands['all_mean_m'], 'm')} |"
        )
    if unit["subset"]:
        lines += ["", "| subset arm | k | draw | travel p90 | travel mean | travel p50 | all mean | all p50 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in unit["subset"]:
            cells = [_cell(row["arm"], row["metrics"][label]) for label, _, _ in METRICS]
            lines.append(f"| {row['arm']} | {row['k_percent']}% | {row['draw']} | " + " | ".join(cells) + " |")
    return lines + [""]


def ranking_cell(entry: dict) -> str:
    agreement = "same order" if entry["same_order"] else ("same top" if entry["same_top"] else f"top {entry['top']}")
    if "spearman" in entry:
        return f"{entry['spearman']:+.3f} / {entry['kendall']:+.3f}, {agreement}"
    return agreement


def ranking_markdown(table: dict) -> list[str]:
    summary = table["ranking_summary"]
    lines = [
        "## Ranking check (descriptive, no pass/fail)",
        "",
        "Arms ranked by travel p90 (lowest first), against the ranking under each other statistic. Within a unit, "
        "ranking by metres and by skill is the same, since P is the common denominator.",
        "",
        f"Over {summary['n_units']} units:",
        "",
        "| against travel p90 | same order | same top arm | median ρ | min ρ | median τ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in METRICS[1:]:
        row = summary[label]
        lines.append(f"| {TITLES[label]} | {row['same_order']} of {summary['n_units']} | {row['same_top']} of "
                     f"{summary['n_units']} | {_s(row['median_spearman'])} | {_s(row['min_spearman'])} | "
                     f"{_s(row['median_kendall'])} |")
    lines += ["", "Per unit (Spearman ρ / Kendall τ against travel p90, then agreement):", "",
              "| unit | arms | top under travel p90 | " + " | ".join(TITLES[l] for l, _, _ in METRICS[1:]) + " |",
              "|---|---:|---|" + "---|" * (len(METRICS) - 1)]
    for unit in table["units"]:
        result = unit["ranking"]
        if "against_travel_p90" not in result:
            lines.append(f"| {unit_label(unit)} | {result.get('n_arms', 0)} | {result['error']} |"
                         + " |" * (len(METRICS) - 1))
            continue
        lines.append(f"| {unit_label(unit)} | {result['n_arms']} | {result['order']['travel_p90'][0]} | "
                     + " | ".join(ranking_cell(result["against_travel_p90"][l]) for l, _, _ in METRICS[1:]) + " |")
    return lines + [""]


def across_markdown(table: dict) -> list[str]:
    across = table["across_species"]
    lines = ["## Across species" if across["form"] == "side_by_side" else "## Across birds", ""]
    if across["form"] == "side_by_side":
        species = across["species"]
        lines += ["Stored skill against P, species side by side (no median over three).", ""]
        header = "| arm | " + " | ".join(f"{TITLES[l]}: {short(s)}" if i == 0 else short(s)
                                         for l, _, _ in METRICS for i, s in enumerate(species)) + " |"
        lines += [header, "|---|" + "---:|" * (len(species) * len(METRICS))]
        for arm, values in across["arms"].items():
            lines.append(f"| {arm} | " + " | ".join(_s(values[s][l]) for l, _, _ in METRICS for s in species) + " |")
        return lines + [""]
    lines += ["Median over birds of each arm's stored skill against P; n is the number of birds with a value.", ""]
    for horizon, block in across["horizons"].items():
        title = "" if horizon == "all" else f"{horizon} h, "
        for name in block["sets"]:
            count = "" if name[0].isdigit() else f" ({block['sets'][name]} birds)"  # rotation names carry the count
            lines += [f"### {title}{name}{count}", "",
                      "| arm | n | " + " | ".join(TITLES[l] for l, _, _ in METRICS) + " | report's travel p90 median |",
                      "|---|---:|" + "---:|" * len(METRICS) + "---|"]
            for arm, sets in block["arms"].items():
                entry = sets.get(name) or {}
                if not entry:
                    continue
                n = entry["travel_p90"]["n"]
                check = "—"
                if "report_travel_p90_median" in entry:
                    check = f"{_s(entry['report_travel_p90_median'])} ({'same' if entry['matches_report'] else 'DIFFERS'})"
                lines.append(f"| {arm} | {n} | " + " | ".join(_s(entry[l]["median"]) for l, _, _ in METRICS)
                             + f" | {check} |")
            lines.append("")
    return lines


def ablation_markdown(block: dict) -> list[str]:
    lines = ["## Context ablation (travel p90 skill only, as stored)", "",
             "Median over each horizon's scored birds, all windows; C = 12 is the stored `skill_C12`.", ""]
    for horizon, arms in block["median_over_scored_birds"].items():
        cs = sorted({int(c) for values in arms.values() for c in values})
        lines += [f"### {horizon} h", "", "| arm | " + " | ".join(f"C = {c}" for c in cs) + " |",
                  "|---|" + "---:|" * len(cs)]
        for arm, values in arms.items():
            cells = []
            for c in cs:
                entry = values.get(str(c)) or {}
                flag = " (DIFFERS from report)" if entry.get("matches_report") is False else ""
                cells.append(f"{_s(entry.get('median'))}{flag}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Per-bird values are in the JSON (`ablation.rows`), with the gappy and full subsets.", ""]
    return lines


def write_markdown(table: dict, path: Path) -> None:
    lines = [f"# Metric tables: {table['study']}", "",
             f"Read from `{table['source']}`" + (f" (`{table['registered_by']}`" if table["registered_by"] else "")
             + (f", freeze `{table['freeze_commit'][:12]}`)" if table["freeze_commit"] else (")" if table["registered_by"] else ""))
             + ". Descriptive: no prediction file and no test partition was read.", "",
             f"Floor F = {table['evaluation_floor']}; trained arms are the median across seeds (XGBoost single seed); "
             "means are individual-weighted. Each cell is metres (skill against P). Intervals are the report's paired "
             "bootstrap, 2,000 resamples at seed 42; — means the report stored none.", ""]
    lines += across_markdown(table)
    lines += ranking_markdown(table)
    lines += ["## Per unit", ""]
    for unit in table["units"]:
        lines += unit_markdown(unit)
    if "ablation" in table:
        lines += ablation_markdown(table["ablation"])
    gaps = table["not_stored"]
    lines += ["## Not stored", ""]
    lines += [f"- Never stored here: {item}." for item in gaps["never_stored"]]
    lines += [f"- {note}" for note in gaps["notes"]]
    present = ", ".join(f"{series} {n} of {gaps['arm_rows']}" for series, n in gaps["intervals_present"].items())
    lines.append(f"- Intervals present per arm row: {present}.")
    missing = {label: n for label, n in gaps["metres_missing"].items() if n}
    if missing:
        lines.append(f"- Metres missing: {missing}.")
    if gaps["units_not_scored"]:
        lines.append(f"- Not scored: {', '.join(gaps['units_not_scored'])}.")
    lines.append(f"- Interval copies that disagree: {len(table['interval_conflicts'])}"
                 + (" (listed in the JSON)." if table["interval_conflicts"] else "."))
    if table["ignored_interval_labels"]:
        lines.append(f"- Interval labels that are not headline arms, ignored: {', '.join(table['ignored_interval_labels'])}.")
    path.write_text("\n".join(lines) + "\n")


def write_ranking_overview(tables: dict[str, dict], directory: Path) -> list[str]:
    rows = {study: table["ranking_summary"] for study, table in tables.items()}
    lines = ["# Ranking check across studies (descriptive)", "",
             "Per study: units where the arm order under travel p90 is identical to the order under each other "
             "statistic, units with the same top arm, and the median Spearman ρ.", "",
             "| study | units | " + " | ".join(f"{TITLES[l]}: same order / same top / median ρ" for l, _, _ in METRICS[1:])
             + " |", "|---|---:|" + "---|" * (len(METRICS) - 1)]
    for study, summary in rows.items():
        n = summary["n_units"]
        cells = [f"{summary[l]['same_order']}/{n} · {summary[l]['same_top']}/{n} · {_s(summary[l]['median_spearman'])}"
                 for l, _, _ in METRICS[1:]]
        lines.append(f"| {study} | {n} | " + " | ".join(cells) + " |")
    (directory / "ranking_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (directory / "ranking_summary.md").write_text("\n".join(lines) + "\n")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", action="append", choices=sorted(STUDIES), help="one study; repeatable (default all)")
    parser.add_argument("--out", type=Path, default=OUT, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tables = {}
    for study in args.study or list(STUDIES):
        table = build(study, load_report(study))
        (args.out / f"{study}.json").write_text(json.dumps(table, indent=2) + "\n")
        write_markdown(table, args.out / f"{study}.md")
        tables[study] = table
        print(f"{study:<16} {len(table['units']):>4} units, interval conflicts {len(table['interval_conflicts'])}, "
              f"wrote {study}.md / .json")
    print("\n" + "\n".join(write_ranking_overview(tables, args.out)))


if __name__ == "__main__":
    main()
