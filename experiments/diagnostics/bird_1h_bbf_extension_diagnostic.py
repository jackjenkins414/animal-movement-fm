# GenAI reference: I outlined this validation only diagnostic for the BbF extension; compare BbF at C = 24/48/72 with its Bb and 
# LSTM-Bb twins on the same train/val samples using the collapse checks, with no test data, and draft the freeze’s “Extension as 
# trained” bullets; GenAI wrote it and I checked it
"""
Validation diagnostic for the BbF extension (`preregistration/birds_1h_bbf_exploratory_freeze.md` sec 10; exploratory;
no training; training and validation windows only).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO))

import bird_1h_bbf_train as F  # noqa: E402
import transformer_collapse as TC  # noqa: E402

b1h = TC.b1h
HORIZONS = TC.HORIZONS

OUT = REPO / "results" / "birds_1h_bbf_extension_diagnostic"
REPORT_JSON = OUT / "birds_1h_bbf_extension_diagnostic.json"
REPORT_MD = OUT / "birds_1h_bbf_extension_diagnostic.md"
MODE = "bbf_extension"
FOLDS = tuple(sorted(F.FOLDS))
CONTEXTS = F.EXTENSION_CONTEXTS
ARMS = ("BbF", "Bb", "LSTMBb")  # the repaired run, then its two 1 h ablation twins
GROUPS = ("pool", "held_out")
DEFAULTS = {name: F.TrainConfig.__dataclass_fields__[name].default for name in F.SWITCHES}
NEAR_ZERO = 0.99  # training loss / zero at or above: within 1% of the zero predictor's (Tier 4 repaired freeze sec 5)
_LABEL = re.compile(r"^(BbF|Bb|LSTMBb)_s(\d+)_C(\d+)$")

if F.b1h is not b1h:
    raise RuntimeError("the BbF driver and transformer_collapse must share one bird_1h_train module")


def twin_runs(run) -> dict:
    """An extension run and, at its fold, seed and C, the 1 h ablation's Bb (its base) and LSTM-Bb: {arm: run}."""
    return {"BbF": run, "Bb": run.base, "LSTMBb": b1h.Run("LSTMBb", run.fold, "", run.seed, 0, run.context)}


if any(twin not in b1h.PLAN for run in F.EXTENSION for arm, twin in twin_runs(run).items() if arm != "BbF"):
    raise RuntimeError("every extension run's Bb and LSTM-Bb twins must be 1 h ablation runs")


def label(arm: str, seed: int, context: int) -> str:
    return f"{arm}_s{seed}_C{context}"


def parse(key: str) -> tuple[str, int, int]:
    match = _LABEL.match(key)
    if not match:
        raise SystemExit(f"FAIL: {key!r} is not an extension diagnostic label")
    return match.group(1), int(match.group(2)), int(match.group(3))


def checked_load(arm: str, run):
    """transformer_collapse.load_run's checks, plus the switches on the record and on the built model."""
    model, k, record = TC.load_run(run, F.RESULTS, F.TAG) if arm == "BbF" else TC.load_run(run)
    want = F.SWITCHES if arm == "BbF" else DEFAULTS
    recorded = {name: (record.get("config") or {}).get(name, DEFAULTS[name]) for name in F.SWITCHES}
    built = {name: getattr(model, name, None) for name in F.SWITCHES}
    if recorded != want or built != want:
        raise SystemExit(f"FAIL: {run.run_id}: switches {recorded} recorded and {built} built; expected {want}")
    return model, k, record


def entries(fold: int) -> list[dict]:
    """transformer_collapse entries for one fold: per extension run (C, then seed), BbF and then its two twins."""
    out = []
    for run in (r for r in F.EXTENSION if r.fold == fold):
        for arm, twin in twin_runs(run).items():
            out.append({"label": label(arm, run.seed, run.context), "run_id": twin.run_id, "context": run.context,
                        "pair": f"{arm}_s{run.seed}", "transformer": arm != "LSTMBb",
                        "load": lambda arm=arm, twin=twin: checked_load(arm, twin)})
    if not out:
        raise SystemExit(f"FAIL: the extension has no run on fold {fold}")
    return out


def record_status(fold: int) -> dict[str, dict]:
    """Per arm, how many of the fold's records are on disk (the dry run reports this; nothing is loaded)."""
    out = {}
    for run in (r for r in F.EXTENSION if r.fold == fold):
        for arm, twin in twin_runs(run).items():
            results = F.RESULTS if arm == "BbF" else b1h.RESULTS
            row = out.setdefault(arm, {"planned": 0, "present": 0})
            row["planned"] += 1
            row["present"] += (results / f"{twin.run_id}.json").exists()
    return out


def run_fold(fold: int, **kwargs) -> dict:
    if fold not in FOLDS:
        raise SystemExit(f"FAIL: fold {fold} is not one of {FOLDS}")
    return TC.run_fold(fold, entries=entries(fold), mode=MODE, **kwargs)


# Summary
def run_rows(f: dict) -> list[dict]:
    """One row per model in a fold file: its losses, 6 h input sensitivity, behaviour and skill, and the bug checks."""
    rows = []
    for key in f["labels"]:
        arm, seed, context = parse(key)
        sens = TC._get(f, "checks", key, "token_sensitivity") or {}
        out6 = TC._get(sens, "median_abs_output", "h6")
        row = {
            "fold": f["fold"], "arm": arm, "seed": seed, "context": context, "label": key,
            "train_loss_over_zero": TC._ratio(TC._get(f, "losses", "train", key, "model", "mean"),
                                              TC._get(f, "losses", "train", key, "zero", "mean")),
            "swap_over_output_h6": TC._ratio(TC._get(sens, "last_swapped", "h6"), out6),
            "plus_1_over_output_h6": TC._ratio(TC._get(sens, "last_plus_1", "h6"), out6),
            "masked_noise_max": TC._get(f, "checks", key, "masked_noise", "max_abs_diff"),
            "fast_path_max": TC._get(f, "checks", key, "fast_path", "max_abs_diff"),
        }
        for group in GROUPS:
            row[f"val_loss_over_zero_{group}"] = TC._ratio(TC._get(f, "losses", "val", key, group, "model", "mean"),
                                                           TC._get(f, "losses", "val", key, group, "zero", "mean"))
            for h in map(str, HORIZONS):
                for metric in ("ratio", "cos_actual"):
                    row[f"{metric}_{group}_h{h}"] = [
                        TC._get(f, "behaviour", key, b, "all", h, metric) for b in f["group_of"]
                        if f["group_of"][b] == group and b in f["behaviour"].get(key, {})]
                row[f"skill_{group}_h{h}"] = [TC._get(f, "skill", b, h, key) for b in f["skill"]
                                              if f["group_of"][b] == group]
        rows.append(row)
    return rows


def summarise(folds: list[dict]) -> dict:
    """
    Per arm and C: medians over the 30 runs (losses, sensitivity), over bird x run cells (behaviour, skill); runs at or
    above NEAR_ZERO; the largest bug-check difference; and BbF's per-fold medians over seeds.
    """
    rows = [row for f in folds for row in run_rows(f)]
    out: dict = {"by_arm": {}, "bbf_by_fold": {}}
    for arm in ARMS:
        for context in CONTEXTS:
            group = [r for r in rows if r["arm"] == arm and r["context"] == context]
            losses = [r["train_loss_over_zero"] for r in group if r["train_loss_over_zero"] is not None]
            swaps = [r["swap_over_output_h6"] for r in group if r["swap_over_output_h6"] is not None]
            entry = {
                "runs": len(group),
                "train_loss_over_zero": TC._median(losses),
                "train_loss_over_zero_range": [min(losses), max(losses)] if losses else None,
                "runs_near_zero": sum(v >= NEAR_ZERO for v in losses),
                "swap_over_output_h6": TC._median(swaps),
                "swap_over_output_h6_min": min(swaps) if swaps else None,
                "plus_1_over_output_h6": TC._median([r["plus_1_over_output_h6"] for r in group]),
                "masked_noise_max": max((r["masked_noise_max"] for r in group if r["masked_noise_max"] is not None),
                                        default=None),
                "fast_path_max": max((r["fast_path_max"] for r in group if r["fast_path_max"] is not None),
                                     default=None),
            }
            for g in GROUPS:
                entry[f"val_loss_over_zero_{g}"] = TC._median([r[f"val_loss_over_zero_{g}"] for r in group])
                for h in map(str, HORIZONS):
                    for metric in ("ratio", "cos_actual", "skill"):
                        name = f"{metric}_{g}_h{h}"
                        entry[name] = TC._median([v for r in group for v in r[name] if v is not None])
            out["by_arm"][f"{arm}_C{context}"] = entry
    for f in folds:
        out["bbf_by_fold"][str(f["fold"])] = {
            str(c): TC._median([r["train_loss_over_zero"] for r in rows
                                if r["fold"] == f["fold"] and r["arm"] == "BbF" and r["context"] == c])
            for c in CONTEXTS}
    return out


def _v(value, digits: str = ".3g") -> str:
    return "–" if value is None else f"{value:{digits}}"


def _three(summary: dict, arm: str, name: str, digits: str = ".3g") -> str:
    return " / ".join(_v(summary["by_arm"][f"{arm}_C{c}"][name], digits) for c in CONTEXTS)


def as_trained_bullets(summary: dict) -> list[str]:
    """The diagnostic's bullets for the freeze's "Extension as trained", C = 24 / 48 / 72 in each list."""
    names = {"BbF": "BbF", "Bb": "Bb twins (1 h ablation, collapsed encoder)", "LSTMBb": "LSTM-Bb twins (1 h ablation)"}
    lines = [
        f"- **Validation diagnostic** (`bird_1h_bbf_extension_diagnostic.py`, `results/{OUT.name}/`): training and "
        f"validation windows only, no test partition read; {TC.N_TRAIN:,} A8 training draws and up to {TC.MAX_VAL:,} "
        "validation windows per bird, shared by every model of a fold. Medians over 10 folds x 3 seeds; each list is "
        "C = 24 / 48 / 72.",
    ]
    for arm in ARMS:
        near = " / ".join(str(summary["by_arm"][f"{arm}_C{c}"]["runs_near_zero"]) for c in CONTEXTS)
        lines.append(
            f"- **{names[arm]}.** Training loss / zero {_three(summary, arm, 'train_loss_over_zero')} "
            f"(runs at or above {NEAR_ZERO}: {near} of 30 each); last-token swap / |out| at 6 h "
            f"{_three(summary, arm, 'swap_over_output_h6')}; |pred| / |actual| at 6 h on held-out birds "
            f"{_three(summary, arm, 'ratio_held_out_h6')}.")
    transformers = [summary["by_arm"][f"{arm}_C{c}"] for arm in ("BbF", "Bb") for c in CONTEXTS]
    noise = max((e["masked_noise_max"] for e in transformers if e["masked_noise_max"] is not None), default=None)
    fast = max((e["fast_path_max"] for e in transformers if e["fast_path_max"] is not None), default=None)
    lines.append(f"- **Bug checks (BbF and Bb).** Noise in masked tokens moves the output by at most {_v(noise)}; eval "
                 f"against train mode without dropout differs by at most {_v(fast)} (model units).")
    return lines


def write_markdown(folds: list[dict], summary: dict, path: Path) -> None:
    keys = [f"{arm}_C{c}" for arm in ARMS for c in CONTEXTS]
    by = summary["by_arm"]
    lines = [
        "# BbF extension: validation diagnostic (exploratory, validation only)", "",
        f"Folds {', '.join(str(f['fold']) for f in folds)}; the 90 extension runs (BbF at C = "
        f"{', '.join(map(str, CONTEXTS))}, seeds 0-2) and their 1 h ablation twins Bb and LSTM-Bb at the same fold, "
        f"seed and C, from their reported checkpoints. Samples as the collapse diagnosis: {TC.N_TRAIN:,} A8 training "
        f"draws, up to {TC.MAX_VAL:,} validation windows per bird, {TC.N_CHECK:,} for the checks, shared by every "
        "model of a fold. No training and no test partition read. Validation skill is on a capped sample without the "
        "floor and is not comparable to any sealed number. Model units are metres / (window scale x k_h). No "
        "pass/fail.", "",
        "## 1. Loss against the zero predictor (model / zero; medians over runs)", "",
        f"| run | runs | training sample | range | at or above {NEAR_ZERO} | val pool | val held-out |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for key in keys:
        e = by[key]
        span = " – ".join(_v(v) for v in e["train_loss_over_zero_range"]) if e["train_loss_over_zero_range"] else "–"
        lines.append(f"| {key} | {e['runs']} | {_v(e['train_loss_over_zero'])} | {span} | {e['runs_near_zero']} | "
                     f"{_v(e['val_loss_over_zero_pool'])} | {_v(e['val_loss_over_zero_held_out'])} |")
    lines += ["", "## 2. Input sensitivity on validation, 6 h head (share of median |out|)", "",
              "| run | last swapped / \\|out\\| | smallest over runs | last +1 / \\|out\\| |", "|---|---:|---:|---:|"]
    for key in keys:
        e = by[key]
        lines.append(f"| {key} | {_v(e['swap_over_output_h6'])} | {_v(e['swap_over_output_h6_min'])} | "
                     f"{_v(e['plus_1_over_output_h6'])} |")
    lines += ["", "## 3. Validation behaviour and travel p90 skill against P (travel windows; bird x run cells)", "",
              "| run | h | \\|pred\\| / \\|actual\\| pool | held-out | cos(pred, actual) pool | held-out | skill pool | "
              "skill held-out |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in keys:
        e = by[key]
        for h in HORIZONS:
            lines.append(f"| {key} | {h} | {_v(e[f'ratio_pool_h{h}'])} | {_v(e[f'ratio_held_out_h{h}'])} | "
                         f"{_v(e[f'cos_actual_pool_h{h}'])} | {_v(e[f'cos_actual_held_out_h{h}'])} | "
                         f"{_v(e[f'skill_pool_h{h}'])} | {_v(e[f'skill_held_out_h{h}'])} |")
    lines += ["", "## 4. Bug checks (largest over runs; model units)", "",
              "| run | masked-token noise | eval against train mode |", "|---|---:|---:|"]
    for key in keys:
        lines.append(f"| {key} | {_v(by[key]['masked_noise_max'])} | {_v(by[key]['fast_path_max'])} |")
    lines += ["", "LSTM-Bb reads an observed flag rather than a mask, so noise in its unobserved tokens is information, "
              "not a bug.", "", "## 5. BbF training loss / zero by fold (median over seeds)", "",
              "| fold | " + " | ".join(f"C = {c}" for c in CONTEXTS) + " |", "|---:|" + "---:|" * len(CONTEXTS)]
    for fold, row in summary["bbf_by_fold"].items():
        lines.append(f"| {fold} | " + " | ".join(_v(row[str(c)]) for c in CONTEXTS) + " |")
    lines += ["", "## 6. For the freeze's \"Extension as trained\"", "", *as_trained_bullets(summary)]
    path.write_text("\n".join(lines) + "\n")


def report() -> None:
    files = [OUT / f"fold{k}.json" for k in FOLDS]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing {', '.join(missing)}; run every fold first")
    folds = [json.loads(p.read_text()) for p in files]
    for f in folds:
        expected = [e["label"] for e in entries(f["fold"])]
        if f.get("mode") != MODE or f.get("labels") != expected or f.get("opened_test_partition") is not False:
            raise SystemExit(f"FAIL: fold{f['fold']}.json was not written by this diagnostic for the planned runs")
        if f.get("dry_run"):
            raise SystemExit(f"FAIL: fold{f['fold']}.json is a dry run")
    summary = summarise(folds)
    REPORT_JSON.write_text(json.dumps({"exploratory": True, "opened_test_partition": False, "mode": MODE,
                                       "summary": summary, "folds": folds}, indent=1, default=str) + "\n")
    write_markdown(folds, summary, REPORT_MD)
    print(f"wrote {REPORT_JSON}\nwrote {REPORT_MD}\n")
    print("\n".join(as_trained_bullets(summary)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--dry-run", action="store_true", help="samples and guards; no model loaded, nothing written")
    parser.add_argument("--report", action="store_true", help="summarise the ten fold files")
    args = parser.parse_args()
    if args.report:
        return report()
    if args.fold is None:
        parser.error("--fold or --report")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    result = run_fold(args.fold, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"records": record_status(args.fold), "labels": len(result["labels"]),
                          "contexts": result["contexts"], "magnitudes": result["magnitudes"]}, indent=1))
        for group in GROUPS:
            cells = [result["skill"][b]["6"]["last_step"] for b in result["skill"] if result["group_of"][b] == group]
            print(f"last-step validation skill, {group}, 6 h: median {TC._m(cells)} over {len(cells)} birds")
        print(f"dry run done in {result['seconds']:.0f}s; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fold{args.fold}.json"
    path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {path} ({result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
