# GenAI reference: used GenAI to write this file based off my specification and and era5 file, specifically I asked for this script
# to write a probe for the era5 extraction and I haven't changed anything in this but have reviewed all of the code.
"""
ERA5 covariates at every 1 h segment row, from NCI's rt52 replica (env study, step E2).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import resource
import sys
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from animal_movement_fm.data import era5

REPO = Path(__file__).resolve().parents[2]
RT52 = Path("/g/data/rt52/era5/single-levels/reanalysis")
SEGMENTS = REPO / "data" / "processed_1h"
OUT = REPO / "data" / "processed_1h_env"
BUILD = REPO / "results" / "birds_1h_env_build"
PROBE_VARIABLE, PROBE_MONTH = "t2m", "2020-06"
STATIC_TASK = "static"

# directory names to look for under the root, per covariate; elevation comes from surface geopotential
SOURCES = {
    "u10": ("10u", "u10"),
    "v10": ("10v", "v10"),
    "u100": ("100u", "u100"),
    "v100": ("100v", "v100"),
    "t2m": ("2t", "t2m"),
    "blh": ("blh",),
    "tp": ("tp",),
    "msl": ("msl",),
    "lsm": ("lsm",),
    "elevation": ("z",),
}
_DATES = re.compile(r"(\d{8})-(\d{8})\.nc$")
_COORDS = {"time", "longitude", "latitude", "lon", "lat"}


@dataclass(frozen=True)
class Paths:
    root: Path = RT52
    segments: Path = SEGMENTS
    out: Path = OUT
    build: Path = BUILD

    @property
    def parts(self) -> Path:
        return self.out / "parts"

    @property
    def index(self) -> Path:
        return self.parts / "index.parquet"

    @property
    def catalogue(self) -> Path:
        return self.build / "catalogue.json"

    def part(self, variable: str, month: str) -> Path:
        return self.parts / variable / f"{month}.npz"


def _peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # bytes on macOS, KiB on Linux


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def _month(hour_s) -> str:
    return str(np.datetime64(int(hour_s), "s").astype("datetime64[M]"))


def _write_atomic(path: Path, write) -> None:
    """Write to path.tmp, then rename: a part exists under its own name only once it is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        write(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# Index: every segment row with its ERA5 hour.
def birds(paths: Paths) -> list[str]:
    return sorted(p.name.removesuffix("_segments.parquet") for p in paths.segments.glob("*_segments.parquet"))


def read_rows(paths: Paths, bird: str, *, keys: bool = False) -> pd.DataFrame:
    """One bird's segment rows in segment then bin order, the order of the index and of the env file."""
    columns = "segment_id, bin_index, " if keys else ""
    return duckdb.connect().execute(
        f"SELECT {columns}epoch(timestamp) AS epoch_s, lon, lat "
        f"FROM '{paths.segments / f'{bird}_segments.parquet'}' ORDER BY segment_id, bin_index"
    ).df()


def build_index(paths: Paths) -> pd.DataFrame:
    frames = []
    for bird in birds(paths):
        rows = read_rows(paths, bird)
        frames.append(pd.DataFrame({
            "bird": bird,
            "row": np.arange(len(rows), dtype=np.int64),
            "hour_s": era5.era5_hour(rows["epoch_s"].to_numpy(float)),
            "lon": rows["lon"].to_numpy(float),
            "lat": rows["lat"].to_numpy(float),
        }))
    if not frames:
        raise SystemExit(f"FAIL: no *_segments.parquet under {paths.segments}")
    index = pd.concat(frames, ignore_index=True)
    index["bird"] = index["bird"].astype("category")
    return index


def load_index(paths: Paths) -> pd.DataFrame:
    """The cached index, built once so part positions stay valid across resumed runs."""
    if paths.index.exists():
        return pd.read_parquet(paths.index)
    index = build_index(paths)
    _write_atomic(paths.index, lambda handle: index.to_parquet(handle, index=False))
    return index


# Catalogue: what the root holds, read from the files themselves.
def find_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    wanted = {n.lower() for n in names}
    found = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.lower() in wanted]
    return found[0] if found else None


def data_variable(dataset) -> str:
    names = [n for n, v in dataset.variables.items() if n not in _COORDS and len(v.dimensions) >= 2]
    if len(names) != 1:
        raise ValueError(f"expected one data variable, found {names}")
    return names[0]


def time_range(path: Path) -> tuple[int | None, int | None, int]:
    """(first, last, steps) of a file's time coordinate as epoch seconds; (None, None, 0) with no time dimension."""
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        dims = dataset.variables[data_variable(dataset)].dimensions
        if len(dims) == 2:
            return None, None, 0
        times = era5._epoch_hours(dataset.variables[dims[0]])
        return int(times[0]), int(times[-1]), len(times)


def name_range(path: Path) -> tuple[int, int] | None:
    """(first, last) hour implied by a YYYYMMDD-YYYYMMDD file name, read as hourly; None if the name has no range."""
    match = _DATES.search(path.name)
    if not match:
        return None
    first, last = (np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}", "s").astype(np.int64) for d in match.groups())
    return int(first), int(last) + 23 * era5.HOUR_S


def catalogue_variable(root: Path, variable: str) -> dict | None:
    """
    The variable's files and the hours each holds. Ranges come from file names when every sampled name agrees with the
    file's own time coordinate; otherwise every file is opened.
    """
    import netCDF4

    directory = find_dir(root, SOURCES[variable])
    if directory is None:
        return None
    files = sorted(directory.rglob("*.nc"))
    if not files:
        return {"dir": str(directory.relative_to(root)), "nc_variable": None, "files": [], "ranges_from": "none"}
    with netCDF4.Dataset(files[0]) as dataset:
        nc_variable = data_variable(dataset)
    named = [name_range(f) for f in files]
    sample = sorted({0, len(files) // 2, len(files) - 1})
    trusted = all(r is not None for r in named) and all(named[i] == time_range(files[i])[:2] for i in sample)
    ranges = named if trusted else [time_range(f)[:2] for f in files]
    return {
        "dir": str(directory.relative_to(root)),
        "nc_variable": nc_variable,
        "files": [[str(f.relative_to(root)), first, last] for f, (first, last) in zip(files, ranges)],
        "ranges_from": "names" if trusted else "time coordinates",
    }


def load_catalogue(paths: Paths, variables) -> dict:
    if not paths.catalogue.exists():
        raise SystemExit(f"FAIL: {paths.catalogue} is missing; run --probe first")
    catalogue = json.loads(paths.catalogue.read_text())
    absent = [v for v in variables if not (catalogue["variables"].get(v) or {}).get("files")]
    if absent:
        raise SystemExit(f"FAIL: the catalogue has no files for {absent}; see the probe")
    return catalogue


def held(entry: dict, hour_s: np.ndarray) -> np.ndarray:
    """Index of the file whose range holds each hour, or -1 (files do not overlap)."""
    timed = [(f[1], f[2]) for f in entry["files"] if f[1] is not None]
    if not timed:
        return np.full(len(hour_s), -1)
    order = np.argsort([t[0] for t in timed])
    first = np.array([timed[i][0] for i in order], dtype=np.int64)
    last = np.array([timed[i][1] for i in order], dtype=np.int64)
    i = np.searchsorted(first, hour_s, side="right") - 1
    ok = (i >= 0) & (hour_s <= last[np.maximum(i, 0)])
    return np.where(ok, order[np.maximum(i, 0)], -1)


# Probe.
def describe_file(path: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(path) as dataset:
        name = data_variable(dataset)
        var = dataset.variables[name]
        lon = era5._coord(dataset, ("longitude", "lon"))
        lat = era5._coord(dataset, ("latitude", "lat"))
        steps = dataset.dimensions[var.dimensions[0]].size if len(var.dimensions) == 3 else 0
        return {
            "nc_variable": name,
            "dimensions": list(var.dimensions),
            "shape": list(var.shape),
            "time_steps": steps,
            "lon": [float(lon[0]), float(lon[-1]), float(lon[1] - lon[0])],
            "lat": [float(lat[0]), float(lat[-1]), float(lat[1] - lat[0])],
            "dtype": str(var.dtype),
            "scale_factor": float(getattr(var, "scale_factor", 1.0)),
            "add_offset": float(getattr(var, "add_offset", 0.0)),
            "fill_value": None if getattr(var, "_FillValue", None) is None else float(var._FillValue),
            "missing_value": None if getattr(var, "missing_value", None) is None else float(var.missing_value),
            "chunking": var.chunking() if var.chunking() == "contiguous" else list(var.chunking()),
            "filters": {k: v for k, v in (var.filters() or {}).items() if v},
        }


def static_invariant(root: Path, entry: dict) -> bool:
    """A static field's first step in its first file equals its last step in its last file."""
    import netCDF4

    def step(rel, last):
        with netCDF4.Dataset(root / rel) as dataset:
            var = dataset.variables[entry["nc_variable"]]
            var.set_auto_maskandscale(False)
            return np.asarray(var[:] if len(var.dimensions) == 2 else var[-1 if last else 0])

    return bool(np.array_equal(step(entry["files"][0][0], False), step(entry["files"][-1][0], True)))


def _pattern(rel: str) -> str:
    return re.sub(r"\d{8}", "YYYYMMDD", re.sub(r"(?<=/)\d{4}(?=/)", "YYYY", rel))


def probe(paths: Paths, *, variables=era5.RAW, probe_variable=PROBE_VARIABLE, probe_month=PROBE_MONTH) -> dict:
    root = paths.root
    if not root.is_dir():
        raise SystemExit(f"FAIL: {root} is not readable (rt52 membership?)")
    listing = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"root             : {root} ({len(listing)} directories)")
    catalogue, report = {"root": str(root), "variables": {}}, {"variables": {}}
    for variable in variables:
        entry = catalogue_variable(root, variable)
        catalogue["variables"][variable] = entry
        if entry is None or not entry["files"]:
            print(f"{variable:<10} NOT FOUND (looked for {SOURCES[variable]})")
            report["variables"][variable] = None
            continue
        files = entry["files"]
        info = describe_file(root / files[0][0])
        timed = [f for f in files if f[1] is not None]
        hours = np.median([(f[2] - f[1]) / era5.HOUR_S + 1 for f in timed]) if timed else 0
        info.update(
            dir=entry["dir"],
            files=len(files),
            pattern=_pattern(files[0][0]),
            ranges_from=entry["ranges_from"],
            median_steps_per_file=float(hours),
            first_month=_month(min(f[1] for f in timed)) if timed else None,
            last_month=_month(max(f[2] for f in timed)) if timed else None,
        )
        if variable in era5.STATIC:
            info["static_layout"] = "single step" if info["time_steps"] <= 1 else "hourly"
            info["invariant"] = static_invariant(root, entry)
        report["variables"][variable] = info
        print(
            f"{variable:<10} {info['pattern']}  files {len(files)}, {info['median_steps_per_file']:.0f} steps/file, "
            f"{info['first_month']} to {info['last_month']}"
        )
        print(
            f"           grid {info['shape'][-2]}x{info['shape'][-1]}, lon {info['lon']}, lat {info['lat']}, "
            f"{info['dtype']} scale {info['scale_factor']:.6g} offset {info['add_offset']:.6g} fill "
            f"{info['fill_value']} missing {info['missing_value']}, chunks {info['chunking']}, {info['filters']}"
            + (f", {info['static_layout']}, invariant {info['invariant']}" if variable in era5.STATIC else "")
        )
    if missing := [v for v in variables if not report["variables"].get(v)]:
        print(f"missing          : {missing}; root directories: {listing}")

    index = load_index(paths)
    hours = index["hour_s"].to_numpy(np.int64)
    months = era5.era5_month(hours).astype(str)
    report["fixes"], report["unheld"] = len(index), {}
    for variable in variables:
        entry = catalogue["variables"][variable]
        if variable in era5.STATIC or not entry or not entry["files"]:
            continue
        unheld = pd.Series(months[held(entry, hours) < 0]).value_counts().sort_index()
        report["unheld"][variable] = {m: int(n) for m, n in unheld.items()}
    lines = report["unheld"]
    if lines and len({json.dumps(v, sort_keys=True) for v in lines.values()}) == 1:
        lines = {"every hourly variable": next(iter(lines.values()))}
    for label, unheld in lines.items():
        print(f"unheld months    : {label}: {len(unheld)} months, {sum(unheld.values()):,} of {len(index):,} fixes "
              + (f"({', '.join(f'{m} {n:,}' for m, n in unheld.items())})" if unheld else ""))

    entry = catalogue["variables"].get(probe_variable)
    if entry and entry["files"]:
        report["timing"] = time_month(root, entry, probe_month, index)
        t = report["timing"]
        print(f"timing           : {probe_variable} {probe_month}: read {t['read_s']:.1f} s, decode {t['decode_s']:.1f} s "
              f"({t['raw_bytes'] / 1e9:.2f} GB raw), sample {t['fixes']:,} fixes {t['sample_s']:.1f} s, "
              f"peak {_gb(_peak_bytes())}")

    _write_atomic(paths.catalogue, lambda h: h.write(json.dumps(catalogue, indent=1).encode()))
    _write_atomic(paths.build / "probe.json", lambda h: h.write(json.dumps(report, indent=1).encode()))
    print(f"wrote            : {paths.catalogue}, {paths.build / 'probe.json'}")
    return report


def time_month(root: Path, entry: dict, month: str, index: pd.DataFrame) -> dict:
    """Read and decode one month of one variable whole, then sample the month's fixes through era5.read_field."""
    import netCDF4

    start = np.datetime64(month, "M").astype("datetime64[s]").astype(np.int64)
    rel = next((f[0] for f in entry["files"] if f[1] is not None and f[1] <= start <= f[2]), None)
    if rel is None:
        return {"month": month, "file": None}
    t0 = time.perf_counter()
    with netCDF4.Dataset(root / rel) as dataset:
        var = dataset.variables[entry["nc_variable"]]
        var.set_auto_maskandscale(False)
        raw = np.asarray(var[:])
        packing = dict(scale_factor=getattr(var, "scale_factor", 1.0), add_offset=getattr(var, "add_offset", 0.0),
                       fill_values=(getattr(var, "_FillValue", None), getattr(var, "missing_value", None)))
    t1 = time.perf_counter()
    era5.decode(raw, **packing)
    t2 = time.perf_counter()
    hours = index["hour_s"].to_numpy(np.int64)
    rows = np.flatnonzero(era5.era5_month(hours).astype(str) == month)
    era5.read_field(root / rel, entry["nc_variable"], hours[rows], index["lon"].to_numpy()[rows],
                    index["lat"].to_numpy()[rows])
    t3 = time.perf_counter()
    return {"month": month, "file": rel, "raw_bytes": int(raw.nbytes), "read_s": t1 - t0, "decode_s": t2 - t1,
            "fixes": int(len(rows)), "sample_s": t3 - t2}


# Tasks: one per (hourly variable, month) and one per static field.
def plan_tasks(index: pd.DataFrame, variables, months=None) -> list[tuple[str, str, np.ndarray]]:
    hours = index["hour_s"].to_numpy(np.int64)
    month_of = era5.era5_month(hours).astype(str)
    wanted = sorted(set(month_of)) if months is None else sorted(months)
    by_month = {m: np.flatnonzero(month_of == m) for m in wanted}
    tasks = []
    for variable in variables:
        if variable in era5.STATIC:
            tasks.append((variable, STATIC_TASK, np.arange(len(index))))
        else:
            tasks += [(variable, m, rows) for m, rows in by_month.items() if len(rows)]
    return tasks


def run_task(root: str, entry: dict, variable: str, month: str, positions, hours, lon, lat, part: str) -> dict:
    """Sample one task's fixes and write its part; `inside` is False where no file holds the fix's hour."""
    start = time.perf_counter()
    root = Path(root)
    values = np.full(len(positions), np.nan)
    inside = np.zeros(len(positions), dtype=bool)
    used = []
    if month == STATIC_TASK:
        rel = entry["files"][0][0]
        values, _ = era5.read_field(root / rel, entry["nc_variable"], hours, lon, lat, static=True)
        inside[:], used = True, [rel]
    else:
        which = held(entry, hours)
        for f in np.unique(which[which >= 0]):
            rows = np.flatnonzero(which == f)
            rel = entry["files"][f][0]
            values[rows], _ = era5.read_field(root / rel, entry["nc_variable"], hours[rows], lon[rows], lat[rows])
            inside[rows] = True
            used.append(rel)
    if variable == "elevation":
        values = era5.elevation_from_geopotential(values)
    seconds = time.perf_counter() - start
    _write_atomic(Path(part), lambda handle: np.savez(
        handle, positions=np.asarray(positions, dtype=np.int64), values=values.astype(np.float32), inside=inside,
        files=np.array(used, dtype=str), seconds=seconds,
    ))
    return {"variable": variable, "month": month, "fixes": len(positions), "outside": int((~inside).sum()),
            "seconds": seconds, "peak_bytes": _peak_bytes()}


def _run_star(arguments: tuple) -> dict:
    return run_task(*arguments)


def _report(result: dict) -> None:
    print(f"task {result['variable']:<10} {result['month']:<7} {result['fixes']:>10,} fixes, "
          f"{result['outside']:>9,} outside, {result['seconds']:7.1f} s, peak {_gb(result['peak_bytes'])}", flush=True)


def extract(paths: Paths, *, variables=era5.RAW, months=None, workers: int = 1) -> dict:
    catalogue = load_catalogue(paths, variables)
    root = catalogue["root"]
    index = load_index(paths)
    hours = index["hour_s"].to_numpy(np.int64)
    lon, lat = index["lon"].to_numpy(float), index["lat"].to_numpy(float)
    tasks = plan_tasks(index, variables, months)
    todo = [t for t in tasks if not paths.part(t[0], t[1]).exists()]
    print(f"tasks            : {len(tasks)} ({len(tasks) - len(todo)} finished, skipped), {len(index):,} fixes, "
          f"{workers} workers")

    def arguments(task):
        variable, month, rows = task
        return (root, catalogue["variables"][variable], variable, month, rows, hours[rows], lon[rows], lat[rows],
                str(paths.part(variable, month)))

    start, results = time.perf_counter(), []
    # static fields cover every fix, so they run here rather than being shipped to a worker
    for task in [t for t in todo if t[1] == STATIC_TASK]:
        results.append(run_task(*arguments(task)))
        _report(results[-1])
    hourly = [t for t in todo if t[1] != STATIC_TASK]
    if workers > 1 and len(hourly) > 1:
        with get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(_run_star, (arguments(t) for t in hourly), chunksize=1):
                results.append(result)
                _report(result)
    else:
        for task in hourly:
            results.append(run_task(*arguments(task)))
            _report(results[-1])
    wall = time.perf_counter() - start
    if results:
        seconds = [r["seconds"] for r in results if r["month"] != STATIC_TASK] or [0.0]
        print(f"done             : {len(results)} tasks in {wall:.1f} s wall; hourly task median "
              f"{np.median(seconds):.1f} s, max {max(seconds):.1f} s; peak {_gb(max(r['peak_bytes'] for r in results))}")
    return {"tasks": len(tasks), "skipped": len(tasks) - len(todo), "results": results, "wall_s": wall}


# Assemble.
def assemble(paths: Paths, *, variables=era5.RAW) -> pd.DataFrame:
    """
    Every part into data/processed_1h_env/{bird}_env.parquet and coverage.csv. Nothing is written unless every index
    row has exactly one value per variable, the segments still match the index, and keys are unique.
    """
    load_catalogue(paths, variables)
    index = load_index(paths)
    n = len(index)
    tasks = plan_tasks(index, variables)
    if missing := [f"{v} {m}" for v, m, _ in tasks if not paths.part(v, m).exists()]:
        raise SystemExit(f"FAIL: {len(missing)} parts missing, e.g. {missing[:5]}; run the extraction first")

    values = {v: np.full(n, np.nan, dtype=np.float32) for v in variables}
    inside = np.ones(n, dtype=bool)  # every hourly variable's file holds the fix's hour
    for variable in variables:
        counts = np.zeros(n, dtype=np.int64)
        for _, month, _ in (t for t in tasks if t[0] == variable):
            with np.load(paths.part(variable, month)) as part:
                positions = part["positions"]
                counts += np.bincount(positions, minlength=n)
                values[variable][positions] = part["values"]
                if variable not in era5.STATIC:
                    inside[positions] &= part["inside"]
        if (counts != 1).any():
            raise ValueError(
                f"{variable}: {int((counts == 0).sum()):,} index rows have no value and {int((counts > 1).sum()):,} "
                "more than one; parts and index are not one to one, nothing written"
            )

    bird_of = index["bird"].astype(str).to_numpy()
    spans = {b: np.flatnonzero(bird_of == b) for b in birds(paths)}
    for bird, rows in spans.items():  # check every bird before writing any
        keys = read_rows(paths, bird, keys=True)
        if keys.duplicated(["segment_id", "bin_index"]).any():
            raise ValueError(f"{bird}: duplicate (segment_id, bin_index) keys in the segments; not one to one")
        same = len(keys) == len(rows) and (
            np.array_equal(era5.era5_hour(keys["epoch_s"].to_numpy(float)), index["hour_s"].to_numpy()[rows])
            and np.array_equal(keys["lon"].to_numpy(float), index["lon"].to_numpy()[rows])
            and np.array_equal(keys["lat"].to_numpy(float), index["lat"].to_numpy()[rows])
        )
        if not same:
            raise ValueError(f"{bird}: the segments ({len(keys):,} rows) no longer match the index ({len(rows):,}); "
                             "not one to one, nothing written")
    if extra := set(index["bird"].astype(str)) - set(spans):
        raise ValueError(f"the index has birds with no segments file: {sorted(extra)}")

    hourly = [v for v in variables if v not in era5.STATIC]
    month_of = era5.era5_month(index["hour_s"].to_numpy(np.int64)).astype(str)
    coverage = []
    for bird, rows in spans.items():
        keys = read_rows(paths, bird, keys=True)
        frame = pd.DataFrame({
            "bird": bird,
            "segment_id": keys["segment_id"],
            "bin_index": keys["bin_index"].astype(np.int64),
            "era5_time": index["hour_s"].to_numpy()[rows].astype("datetime64[s]"),
            **{v: values[v][rows] for v in variables},
        })
        _write_atomic(paths.out / f"{bird}_env.parquet", lambda h: frame.to_parquet(h, index=False))
        finite = np.column_stack([np.isfinite(values[v][rows]) for v in variables])
        coverage.append({
            "bird": bird,
            "rows": len(rows),
            **{f"share_{v}": float(finite[:, i].mean()) for i, v in enumerate(variables)},
            "share_all": float(finite.all(axis=1).mean()),
            "missing_months": ";".join(sorted(set(month_of[rows][~inside[rows]]))) if hourly else "",
        })
    table = pd.DataFrame(coverage)
    paths.build.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths.build / "coverage.csv", index=False)
    for row in coverage:
        months = row["missing_months"].split(";") if row["missing_months"] else []
        print(f"{row['bird']:<26} {row['rows']:>10,} rows, all covariates {row['share_all']:.4f}, "
              f"missing months {len(months)}" + (f" ({', '.join(months[:4])}{', ...' if len(months) > 4 else ''})"
                                                 if months else ""))
    weights = table["rows"] / table["rows"].sum()
    print(f"assembled        : {len(table)} birds, {int(table['rows'].sum()):,} rows, all covariates "
          f"{float((table['share_all'] * weights).sum()):.4f}; {paths.out}, {paths.build / 'coverage.csv'}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--months", help="comma separated YYYY-MM, e.g. 2020-06")
    mode.add_argument("--assemble", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    paths = Paths()
    if args.probe:
        probe(paths)
    elif args.assemble:
        assemble(paths)
    else:
        months = None
        if args.months:
            months = [m.strip() for m in args.months.split(",")]
            if not all(re.fullmatch(r"\d{4}-\d{2}", m) for m in months):
                raise SystemExit(f"FAIL: --months takes YYYY-MM[,YYYY-MM...]; got {args.months!r}")
        extract(paths, months=months, workers=args.workers)


if __name__ == "__main__":
    main()
