# GenAI reference: I specified these tests for era5_extract.py; synthetic missing month setup, end to end checks, resume/error cases and NaN 
# coverage. 
from __future__ import annotations

import netCDF4
import numpy as np
import pandas as pd
import pytest

import experiments.data.era5_extract as ex
from animal_movement_fm.data import era5

VARIABLES = ("u10", "t2m", "lsm")
SHORT = {"u10": "10u", "t2m": "2t", "lsm": "lsm"}
MONTHS = {"2020-01": ("20200101", "20200131"), "2020-03": ("20200301", "20200331")}  # February is not in the tree
UNITS = "hours since 1900-01-01 00:00:00.0"
EPOCH_1900 = np.datetime64("1900-01-01T00:00:00", "s").astype(np.int64)
LON = -180.0 + 10.0 * np.arange(36)
LAT = 90.0 - 10.0 * np.arange(19)


def _nc(path, name, raw, hours=None, scale=0.1, offset=250.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("longitude", len(LON))
        ds.createDimension("latitude", len(LAT))
        ds.createVariable("longitude", "f4", ("longitude",))[:] = LON
        ds.createVariable("latitude", "f4", ("latitude",))[:] = LAT
        dims = ("latitude", "longitude")
        if hours is not None:
            ds.createDimension("time", len(hours))
            time = ds.createVariable("time", "i4", ("time",))
            time.units, time.calendar = UNITS, "gregorian"
            time[:] = (np.asarray(hours) - EPOCH_1900) // 3600
            dims = ("time",) + dims
        var = ds.createVariable(name, "i2", dims, fill_value=np.int16(-32767))
        var.set_auto_maskandscale(False)
        var.scale_factor, var.add_offset = scale, offset
        var[:] = raw


def _month_hours(first, last):
    start = np.datetime64(f"{first[:4]}-{first[4:6]}-{first[6:]}", "s").astype(np.int64)
    end = np.datetime64(f"{last[:4]}-{last[4:6]}-{last[6:]}", "s").astype(np.int64) + 23 * 3600
    return np.arange(start, end + 1, 3600)


def _tree(root, static_layout):
    rng = np.random.default_rng(0)
    for variable in ("u10", "t2m"):
        for first, last in MONTHS.values():
            hours = _month_hours(first, last)
            raw = rng.integers(-2000, 2000, (len(hours), len(LAT), len(LON))).astype(np.int16)
            _nc(root / SHORT[variable] / first[:4] / f"{SHORT[variable]}_era5_oper_sfc_{first}-{last}.nc",
                variable, raw, hours)
    land = rng.integers(0, 10, (len(LAT), len(LON))).astype(np.int16)
    if static_layout == "single step":
        _nc(root / "lsm" / "lsm_era5_invariant.nc", "lsm", land[None], [_month_hours("20200101", "20200101")[0]],
            scale=0.1, offset=0.0)
    else:
        for first, last in MONTHS.values():
            hours = _month_hours(first, last)
            _nc(root / "lsm" / first[:4] / f"lsm_era5_oper_sfc_{first}-{last}.nc", "lsm",
                np.broadcast_to(land, (len(hours),) + land.shape), hours, scale=0.1, offset=0.0)
    (root / "sst").mkdir()  # an unrelated directory the probe lists and ignores


def _segments(directory, extra_duplicate=False):
    """Two birds over January to March 2020; one crosses the antimeridian."""
    rng = np.random.default_rng(1)
    directory.mkdir(parents=True, exist_ok=True)
    start = np.datetime64("2020-01-01T00:00:00", "s").astype(np.int64)
    for bird, lon0 in (("bird_a", 20.0), ("bird_b", 179.5)):
        rows = []
        for s in range(2):
            n = 400
            bins = np.sort(rng.choice(np.arange(2160), n, replace=False))  # 90 days of hourly bins
            ts = start + bins * 3600 + rng.integers(0, 3600, n)
            lon = (lon0 + np.cumsum(rng.normal(0, 0.3, n)) + 180) % 360 - 180
            lat = np.clip(-40 + 30 * s + np.cumsum(rng.normal(0, 0.2, n)), -89, 89)
            rows.append(pd.DataFrame({
                "segment_id": f"{bird}_{s}", "individual_id": bird, "bin_index": bins.astype(np.int64),
                "timestamp": ts.astype("datetime64[s]").astype("datetime64[us]"), "lat": lat, "lon": lon,
            }))
        frame = pd.concat(rows, ignore_index=True)
        if extra_duplicate and bird == "bird_b":
            frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
        frame.to_parquet(directory / f"{bird}_segments.parquet", index=False)


@pytest.fixture(params=["single step", "hourly"])
def built(tmp_path, request):
    root = tmp_path / "rt52"
    _tree(root, request.param)
    paths = ex.Paths(root=root, segments=tmp_path / "segments", out=tmp_path / "env", build=tmp_path / "build")
    _segments(paths.segments)
    report = ex.probe(paths, variables=VARIABLES, probe_month="2020-01")
    return paths, report, request.param


def _expected(paths, bird):
    """era5.py applied directly: each fix's hour, its month's file, bilinear; February absent."""
    rows = ex.read_rows(paths, bird)
    hours = era5.era5_hour(rows["epoch_s"].to_numpy(float))
    months = era5.era5_month(hours).astype(str)
    lon, lat = rows["lon"].to_numpy(float), rows["lat"].to_numpy(float)
    out = {}
    for variable in ("u10", "t2m"):
        values = np.full(len(rows), np.nan)
        for month, (first, last) in MONTHS.items():
            sel = months == month
            path = paths.root / SHORT[variable] / "2020" / f"{SHORT[variable]}_era5_oper_sfc_{first}-{last}.nc"
            values[sel], _ = era5.read_field(path, variable, hours[sel], lon[sel], lat[sel])
        out[variable] = values.astype(np.float32)
    static = sorted((paths.root / "lsm").rglob("*.nc"))[0]
    out["lsm"] = era5.read_field(static, "lsm", hours, lon, lat, static=True)[0].astype(np.float32)
    return out, months


def test_probe_reports_layouts_and_unheld_months(built):
    paths, report, layout = built
    assert set(report["variables"]) == set(VARIABLES)
    u10 = report["variables"]["u10"]
    assert u10["pattern"] == "10u/YYYY/10u_era5_oper_sfc_YYYYMMDD-YYYYMMDD.nc"
    assert (u10["first_month"], u10["last_month"], u10["median_steps_per_file"]) == ("2020-01", "2020-03", 744.0)
    assert u10["lon"] == [-180.0, 170.0, 10.0] and u10["lat"] == [90.0, -90.0, -10.0] and u10["dtype"] == "int16"
    assert report["variables"]["lsm"]["static_layout"] == layout and report["variables"]["lsm"]["invariant"]
    index = ex.load_index(paths)
    february = int((era5.era5_month(index["hour_s"].to_numpy()).astype(str) == "2020-02").sum())
    assert february > 0 and report["unheld"]["u10"] == {"2020-02": february} == report["unheld"]["t2m"]
    assert report["timing"]["fixes"] > 0 and paths.catalogue.exists()


def test_end_to_end_matches_era5_applied_directly(built):
    paths, _, _ = built
    summary = ex.extract(paths, variables=VARIABLES)
    assert summary["skipped"] == 0 and summary["tasks"] == 2 * 3 + 1  # 2 hourly x 3 months + static
    table = ex.assemble(paths, variables=VARIABLES)
    for bird in ("bird_a", "bird_b"):
        env = pd.read_parquet(paths.out / f"{bird}_env.parquet")
        segments = ex.read_rows(paths, bird, keys=True)
        assert env["segment_id"].tolist() == segments["segment_id"].tolist()
        assert env["bin_index"].tolist() == segments["bin_index"].tolist() and (env["bird"] == bird).all()
        expected, months = _expected(paths, bird)
        for variable in VARIABLES:
            np.testing.assert_array_equal(env[variable].to_numpy(), expected[variable])
        assert np.isnan(env.loc[months == "2020-02", ["u10", "t2m"]].to_numpy()).all()
        assert np.isfinite(env.loc[months != "2020-02", list(VARIABLES)].to_numpy()).all()
        hours = era5.era5_hour(segments["epoch_s"].to_numpy(float))
        assert (env["era5_time"].to_numpy().astype("datetime64[s]").astype(np.int64) == hours).all()
        row = table.set_index("bird").loc[bird]
        assert row["missing_months"] == "2020-02" and row["share_lsm"] == 1.0
        assert row["share_u10"] == pytest.approx((months != "2020-02").mean())
    assert (paths.build / "coverage.csv").exists()


def test_months_mode_runs_only_those_months_and_the_static_field(built):
    paths, _, _ = built
    summary = ex.extract(paths, variables=VARIABLES, months=["2020-01"])
    done = sorted((r["variable"], r["month"]) for r in summary["results"])
    assert done == [("lsm", "static"), ("t2m", "2020-01"), ("u10", "2020-01")]
    with pytest.raises(SystemExit, match="parts missing"):
        ex.assemble(paths, variables=VARIABLES)


def test_workers_give_the_same_parts(built, tmp_path):
    paths, _, _ = built
    ex.extract(paths, variables=VARIABLES, workers=2)
    pooled = {p.relative_to(paths.parts): np.load(p)["values"] for p in paths.parts.rglob("*.npz")}
    other = ex.Paths(root=paths.root, segments=paths.segments, out=tmp_path / "env1", build=paths.build)
    ex.extract(other, variables=VARIABLES)
    for rel, values in pooled.items():
        np.testing.assert_array_equal(np.load(other.parts / rel)["values"], values)


def test_resume_skips_finished_parts_and_redoes_a_half_written_one(built):
    paths, _, _ = built
    ex.extract(paths, variables=VARIABLES)
    parts = sorted(paths.parts.rglob("*.npz"))
    stamps = {p: p.stat().st_mtime_ns for p in parts}
    again = ex.extract(paths, variables=VARIABLES)
    assert again["skipped"] == again["tasks"] and not again["results"]
    assert all(p.stat().st_mtime_ns == stamps[p] for p in parts)

    victim = paths.part("t2m", "2020-03")
    victim.unlink()
    victim.with_name(victim.name + ".tmp").write_bytes(b"half written")  # a write that never reached the rename
    redo = ex.extract(paths, variables=VARIABLES)
    assert [(r["variable"], r["month"]) for r in redo["results"]] == [("t2m", "2020-03")]
    assert victim.exists() and not victim.with_name(victim.name + ".tmp").exists()
    assert all(p.stat().st_mtime_ns == stamps[p] for p in parts if p != victim)


def test_assemble_refuses_a_part_missing_a_row(built):
    paths, _, _ = built
    ex.extract(paths, variables=VARIABLES)
    part = paths.part("u10", "2020-01")
    with np.load(part) as data:
        arrays = {k: data[k] for k in data.files}
    np.savez(part, **{k: (v[1:] if k in ("positions", "values", "inside") else v) for k, v in arrays.items()})
    with pytest.raises(ValueError, match="1 index rows have no value"):
        ex.assemble(paths, variables=VARIABLES)
    assert not list(paths.out.glob("*_env.parquet"))


def test_assemble_refuses_a_row_given_twice(built):
    paths, _, _ = built
    ex.extract(paths, variables=VARIABLES)
    part = paths.part("t2m", "2020-03")
    with np.load(part) as data:
        arrays = {k: data[k] for k in data.files}
    doubled = {k: (np.concatenate([v, v[:1]]) if k in ("positions", "values", "inside") else v) for k, v in arrays.items()}
    np.savez(part, **doubled)
    with pytest.raises(ValueError, match="1 more than one"):
        ex.assemble(paths, variables=VARIABLES)


def test_assemble_refuses_segments_that_changed_since_the_index(built):
    paths, _, _ = built
    ex.extract(paths, variables=VARIABLES)
    frame = pd.read_parquet(paths.segments / "bird_a_segments.parquet")
    frame.iloc[:-1].to_parquet(paths.segments / "bird_a_segments.parquet", index=False)  # a key gone
    with pytest.raises(ValueError, match="no longer match the index"):
        ex.assemble(paths, variables=VARIABLES)
    assert not list(paths.out.glob("*_env.parquet"))


def test_assemble_refuses_duplicate_keys(tmp_path):
    root = tmp_path / "rt52"
    _tree(root, "single step")
    paths = ex.Paths(root=root, segments=tmp_path / "segments", out=tmp_path / "env", build=tmp_path / "build")
    _segments(paths.segments, extra_duplicate=True)
    ex.probe(paths, variables=VARIABLES, probe_month="2020-01")
    ex.extract(paths, variables=VARIABLES)
    with pytest.raises(ValueError, match="duplicate"):
        ex.assemble(paths, variables=VARIABLES)
    assert not list(paths.out.glob("*_env.parquet"))


def test_extraction_needs_the_probe_first(tmp_path):
    paths = ex.Paths(root=tmp_path, segments=tmp_path, out=tmp_path / "env", build=tmp_path / "build")
    with pytest.raises(SystemExit, match="run --probe first"):
        ex.extract(paths, variables=VARIABLES)
