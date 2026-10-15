# GenAI reference: written by GenAI from behaviours I gave (note, when I have been saying this, it includes giving the file and 
# sometimes multiple files) windows stay contiguous runs, counts match the registered counter. GenAI added dead channel and 
# elapsed time tests after finding constant channels. Run lengths keep the expected window counts easy to check by hand. I checked 
# all the tests. 

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.split import window_counts_per_individual

C = 12


def _segments(tmp_path, runs, name="seg.parquet", jitter_seed=None):
    """
    `runs` is {individual_id: [run_length, ...]}; runs are separated by a two bin hole.

    With `jitter_seed` set, each fix is offset within its bin and the track is given positional noise,
    which is what a real species looks like: the bin representative is the fix nearest the bin centre,
    so adjacent-bin steps span a range of real elapsed times rather than exactly 6h. The offsets stay
    inside [0, 5h) so timestamps remain strictly increasing and `bin_index` still describes the row.
    Default None keeps the clean straight-line track the counting tests rely on.
    """
    rng = np.random.default_rng(jitter_seed) if jitter_seed is not None else None
    rows, bin_cursor = [], 0
    for individual, lengths in runs.items():
        for run_number, length in enumerate(lengths):
            for step in range(length):
                bin_number = bin_cursor + step
                offset_h = float(rng.uniform(0.0, 5.0)) if rng is not None else 0.0
                wobble = rng.normal(0.0, 0.004, size=2) if rng is not None else (0.0, 0.0)
                rows.append(
                    {
                        "segment_id": f"{individual}_s{run_number}",
                        "individual_id": individual,
                        "bin_index": bin_number,
                        "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                        + pd.Timedelta(hours=6 * bin_number + offset_h),
                        "lon": 10.0 + 0.01 * bin_number + wobble[0],
                        "lat": -40.0 + 0.01 * bin_number + wobble[1],
                    }
                )
            bin_cursor += length + 3  # Hole wider than one bin, so runs never merge. 
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_windows_form_only_inside_one_bin_runs(tmp_path):
    """A 20 bin run gives 20-13+1 = 8 windows; a 13 bin run gives exactly 1; a 12 bin run gives none."""
    path = _segments(tmp_path, {"a": [20], "b": [13], "c": [12]})
    for individual, expected in (("a", 8), ("b", 1), ("c", 0)):
        windows = SpeciesWindows.build(path, [individual], context_length=C)
        assert windows.n_windows == expected


def test_window_count_matches_the_registered_counter(tmp_path):
    """Agreement with the function every registered count in the preregistration came from."""
    runs = {"a": [30, 14], "b": [13], "c": [12, 12], "d": [40]}
    path = _segments(tmp_path, runs)
    ids = list(runs)

    counts = window_counts_per_individual(str(path), context_length=C)
    expected = int(counts.set_index("individual_id")["n_windows"].reindex(ids).fillna(0).sum())

    assert SpeciesWindows.build(path, ids, context_length=C).n_windows == expected


def test_a_window_never_spans_a_gap(tmp_path):
    """Two 12 bin runs cannot combine into one 24 bin stretch, however close they sit."""
    path = _segments(tmp_path, {"a": [12, 12]})
    assert SpeciesWindows.build(path, ["a"], context_length=C).n_windows == 0


def test_string_and_integer_ids_both_resolve(tmp_path):
    path = _segments(tmp_path, {"CAR2013_sel_a_m_03": [20]})
    assert SpeciesWindows.build(path, ["CAR2013_sel_a_m_03"], context_length=C).n_windows == 8

    numeric = _segments(tmp_path, {101: [20]}, name="numeric.parquet")
    assert SpeciesWindows.build(numeric, ["101"], context_length=C).n_windows == 8


def test_manifest_segments_mismatch_raises(tmp_path):
    path = _segments(tmp_path, {"a": [20]})
    with pytest.raises(ValueError, match="manifest/segments mismatch"):
        SpeciesWindows.build(path, ["not_here"], context_length=C)


def test_gather_shapes_and_reversibility(tmp_path):
    from animal_movement_fm.data.enu import N_FEATURES, unscale_prediction

    path = _segments(tmp_path, {"a": [40]})
    windows = SpeciesWindows.build(path, ["a"], context_length=C)
    features, target_scaled, scale = windows.gather(np.arange(5))

    assert features.shape == (5, C - 1, N_FEATURES)
    assert target_scaled.shape == (5, 2)
    assert scale.shape == (5,)
    assert np.isfinite(features).all()
    # The target in metres is recoverable from the normalised target and the scale.
    assert np.isfinite(unscale_prediction(target_scaled, scale)).all()


def test_every_input_channel_varies_within_a_real_batch(tmp_path):
    """
    No channel may be constant across a batch. This is the test that was missing.

    `test_backward_pass_reaches_every_parameter` passed for the entire life of the elapsed-time bug,
    because it checks that every parameter *tensor* receives gradient, not that every input *column*
    carries signal -- and a dead channel hides inside a live `input_proj.weight`. Three of the nine
    channels were constant and nothing failed.

    Note the two failure modes differ: `gap` was 0.0 and so contributed nothing and received no
    gradient, while `log_dt = log(21600)` and `dt_bins = 1.0` were non-zero constants acting as
    redundant bias terms on top of the projection's own bias. A variance assertion catches both.
    """
    from animal_movement_fm.data.enu import FEATURES

    path = _segments(tmp_path, {"a": [200]}, jitter_seed=7)
    windows = SpeciesWindows.build(path, ["a"], context_length=C)
    features, _, _ = windows.gather(np.arange(64))

    dead = [name for i, name in enumerate(FEATURES) if np.ptp(features[..., i]) == 0.0]
    assert not dead, f"channels constant across the batch: {dead}"


def test_elapsed_time_reaches_the_encoder(tmp_path):
    """
    The dt channels must reflect real elapsed times, not the assumed one bin.

    Constructed so the assertion is about the loader rather than the encoder: on the clean fixture every
    step really is 6h, so `dt_bins` is legitimately 1.0 there and only the jittered track can tell a
    working loader from the old default.
    """
    from animal_movement_fm.data.enu import BIN_SECONDS, FEATURES

    dt_i = FEATURES.index("dt_bins")
    clean = SpeciesWindows.build(_segments(tmp_path, {"a": [200]}), ["a"], context_length=C)
    jittered = SpeciesWindows.build(
        _segments(tmp_path, {"a": [200]}, "jit.parquet", jitter_seed=11), ["a"], context_length=C
    )

    assert np.allclose(clean.gather(np.arange(32))[0][..., dt_i], 1.0)

    dt = jittered.gather(np.arange(32))[0][..., dt_i]
    assert dt.min() < 1.0 < dt.max()
    # Elapsed seconds are recoverable from the channel, so the units survived the DuckDB epoch() cast.
    assert (dt * BIN_SECONDS).min() > 0.0


def test_target_perturbation_does_not_move_the_scale(tmp_path):
    """
    The no-leakage guarantee, tested where it actually lives.

    `encode_window` normalises every position it is handed, so nothing in the encoder keeps the target
    out of the normaliser -- `gather`'s slicing does, by passing the first C positions only. Perturb the
    target position and nothing else: the scale must not move, while the target itself must.

    A 13 bin run gives exactly one window, so position 12 is that window's target and is not context for
    any other window. With a longer run the perturbed position would also be history for later windows
    and the test would be measuring the wrong thing.
    """
    path = _segments(tmp_path, {"a": [C + 1]})
    windows = SpeciesWindows.build(path, ["a"], context_length=C)
    assert windows.n_windows == 1

    _, target_before, scale_before = windows.gather(np.array([0]))

    # Copied because the arrays come straight off the parquet frame and are read-only under
    # copy-on-write pandas; the point of the test is the perturbation, not where it is stored.
    windows.lon, windows.lat = windows.lon.copy(), windows.lat.copy()
    windows.lon[C] += 5.0  # ~400 km, and only the target
    windows.lat[C] += 2.0
    _, target_after, scale_after = windows.gather(np.array([0]))

    assert scale_after[0] == scale_before[0]
    assert not np.allclose(target_after, target_before)


def test_sampler_follows_the_registered_probabilities(tmp_path):
    """The sampler must reproduce the registered draw shares, not the raw window shares."""
    big = SpeciesWindows.build(_segments(tmp_path, {"a": [400]}, "big.parquet"), ["a"], context_length=C)
    small = SpeciesWindows.build(_segments(tmp_path, {"b": [40]}, "small.parquet"), ["b"], context_length=C)
    assert big.n_windows > 10 * small.n_windows  # window counts are lopsided

    pool = SourcePool({"big": big, "small": small}, {"big": 0.75, "small": 0.25})
    shares = pool.realised_species_shares(np.random.default_rng(42), draws=200_000)
    assert shares["big"] == pytest.approx(0.75, abs=0.005)
    assert shares["small"] == pytest.approx(0.25, abs=0.005)


def test_weights_are_read_from_the_registered_file_not_recomputed(tmp_path):
    """A rebuilt segments file must not be able to move the registered weights silently."""
    big = SpeciesWindows.build(_segments(tmp_path, {"a": [400]}, "big.parquet"), ["a"], context_length=C)
    small = SpeciesWindows.build(_segments(tmp_path, {"b": [40]}, "small.parquet"), ["b"], context_length=C)

    weights_json = tmp_path / "weights.json"
    weights_json.write_text(
        json.dumps({"source_pool": {"big": 0.6, "small": 0.4},
                    "sub_pools": {"without_big": {"small": 1.0}}})
    )

    pool = SourcePool.from_registered_weights({"big": big, "small": small}, weights_json)
    assert dict(zip(pool.names, pool.probabilities)) == pytest.approx({"big": 0.6, "small": 0.4})

    sub = SourcePool.from_registered_weights({"small": small}, weights_json, pool="sub_pools.without_big")
    assert sub.names == ["small"]


def test_batch_is_a_species_mixture(tmp_path):
    path_a = _segments(tmp_path, {"a": [200]}, "a.parquet")
    path_b = _segments(tmp_path, {"b": [200]}, "b.parquet")
    pool = SourcePool(
        {
            "a": SpeciesWindows.build(path_a, ["a"], context_length=C),
            "b": SpeciesWindows.build(path_b, ["b"], context_length=C),
        },
        {"a": 0.5, "b": 0.5},
    )
    features, targets, scales = pool.sample(np.random.default_rng(0), batch_size=256)
    assert features.shape[0] == 256 and targets.shape == (256, 2) and scales.shape == (256,)


from animal_movement_fm.data.enu import encode_window, enu_steps  # noqa: E402
from animal_movement_fm.data.multispecies import HorizonWindows, history_positions  # noqa: E402
import experiments.eda.cadence_horizons as ch  # noqa: E402

HOUR = 3600
HORIZONS = (1, 6, 24)
# a 14-bin run, then 18, 19 and 36 occupied: anchors at bins 11, 12, 13 (the horizon-count tests' layout)
DIRECT_BINS = np.concatenate((np.arange(14), [18, 19, 36]))


def _hourly(tmp_path, segments: dict, name="hourly.parquet"):
    """`segments` is {segment_id: (individual_id, bins)}; a jittered eastward walk on a 1 h grid."""
    rng = np.random.default_rng(3)
    rows = []
    for segment_id, (individual, bins) in segments.items():
        for b in bins:
            rows.append({"segment_id": segment_id, "individual_id": individual, "bin_index": int(b),
                         "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                         + pd.Timedelta(seconds=HOUR * int(b) + int(rng.integers(0, 3000))),
                         "lon": 10.0 + 0.002 * b + rng.normal(0, 2e-4), "lat": 50.0 + rng.normal(0, 2e-4)})
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def _horizon_windows(path, ids=("a",), context=12):
    return HorizonWindows.build(path, list(ids), context=context, core=12, horizons=HORIZONS, bin_seconds=HOUR)


def test_anchors_and_targets_follow_the_direct_rule(tmp_path):
    windows = _horizon_windows(_hourly(tmp_path, {"a_s0": ("a", DIRECT_BINS)}))
    hit = ch.targets_hit(DIRECT_BINS, 12, HORIZONS)
    assert windows.n_windows == int(hit.any(axis=1).sum()) == 3
    assert np.array_equal(windows.targets >= 0, hit[hit.any(axis=1)])
    assert DIRECT_BINS[windows.anchors].tolist() == [11, 12, 13]
    found = windows.targets >= 0
    assert np.array_equal(DIRECT_BINS[windows.targets[found]] - DIRECT_BINS[np.nonzero(found)[0] + 11],
                          np.broadcast_to(np.array(HORIZONS), found.shape)[found])
    assert windows.for_horizon(6).n_windows == int(hit[:, 1].sum())


def test_no_target_or_history_crosses_a_segment(tmp_path):
    # the second segment continues the bins of the first, so only the segment id keeps them apart
    path = _hourly(tmp_path, {"a_s0": ("a", np.arange(12)), "a_s1": ("a", np.arange(12, 40))})
    windows = _horizon_windows(path, context=24)
    assert windows.n_windows == 28 - 11 - 1  # anchors in a_s1 only, each needs bin t + 1 in a_s1
    assert windows.anchors.min() == 23  # the 12th row of a_s1
    grid = history_positions(windows.keys, windows.anchors[:1], 24)
    assert (grid[grid >= 0] >= 12).all()  # the history never reaches back into a_s0


def test_gather_measures_the_core_and_targets_as_the_main_design(tmp_path):
    path = _hourly(tmp_path, {"a_s0": ("a", np.concatenate(([0, 1, 5], np.arange(10, 60))))})
    main = _horizon_windows(path)
    wide = _horizon_windows(path, context=24)
    assert np.array_equal(main.anchors, wide.anchors)
    ids = np.arange(main.n_windows)
    f12, m12, t12, tm12, s12 = main.gather(ids)
    f24, m24, t24, tm24, s24 = wide.gather(ids)
    assert f12.shape == (len(ids), 11, 12) and f24.shape == (len(ids), 23, 12)
    assert np.array_equal(f24[:, -11:], f12) and np.array_equal(s24, s12) and np.array_equal(t24, t12)
    assert m12.all() and np.array_equal(tm12, tm24) and np.array_equal(tm12, main.targets >= 0)
    a = main.anchors
    core, scale = encode_window(main.lon[a[:, None] + np.arange(-11, 1)], main.lat[a[:, None] + np.arange(-11, 1)],
                                timestamp_s=main.epoch_s[a[:, None] + np.arange(-11, 1)], bin_seconds=HOUR)
    assert np.array_equal(f12, core) and np.array_equal(s12, scale)
    j, rows = 1, np.flatnonzero(tm12[:, 1])
    pair = np.stack([a[rows], main.targets[rows, j]], axis=1)
    expected = enu_steps(main.lon[pair], main.lat[pair])[:, 0, :] / scale[rows, None]
    assert np.array_equal(t12[rows, j], expected)
    assert (t12[~tm12] == 0).all()
    assert m24[0, :12].sum() < 12  # the early history has holes, masked


def test_species_windows_draws_are_unchanged_by_the_generic_sample(tmp_path):
    species = {
        "x": SpeciesWindows.build(_segments(tmp_path, {"a": [40, 30]}, name="x.parquet", jitter_seed=1), ["a"]),
        "y": SpeciesWindows.build(_segments(tmp_path, {"b": [50]}, name="y.parquet", jitter_seed=2), ["b"]),
    }
    pool = SourcePool(species, {"x": 0.3, "y": 0.7})
    drawn = pool.sample(np.random.default_rng(5), 64)
    rng = np.random.default_rng(5)  # the registered rule, step by step
    picks = rng.choice(2, size=64, p=pool.probabilities)
    manual = [species[n].gather(rng.integers(0, species[n].n_windows, size=int((picks == i).sum())))
              for i, n in enumerate(pool.names) if (picks == i).any()]
    assert len(drawn) == 3
    for got, column in zip(drawn, zip(*manual)):
        assert np.array_equal(got, np.concatenate(column))


def test_sample_concatenates_horizon_batches(tmp_path):
    windows = _horizon_windows(_hourly(tmp_path, {"a_s0": ("a", np.arange(80))}))
    batch = SourcePool({"a": windows}, {"a": 1.0}).sample(np.random.default_rng(0), 16)
    assert [x.shape for x in batch] == [(16, 11, 12), (16, 11), (16, 3, 2), (16, 3), (16,)]


from dataclasses import replace  # noqa: E402

import duckdb  # noqa: E402

from animal_movement_fm.data import era5  # noqa: E402
from animal_movement_fm.data.multispecies import _read_segments  # noqa: E402

# _read_segments' query with env off, as it was before the env study
ENV_OFF_SQL = (
    "\n        SELECT s.segment_id, s.individual_id, s.bin_index, epoch(s.timestamp) AS epoch_s, s.lon, s.lat"
    "\n        FROM '{path}' s"
    "\n        SEMI JOIN split_ids ids ON s.individual_id = ids.individual_id"
    "\n        ORDER BY s.segment_id, s.bin_index"
    "\n        "
)
UNIT = (np.zeros(era5.N_STANDARDISED), np.ones(era5.N_STANDARDISED))  # standardised = raw on observed tokens


def _env_parquet(tmp_path, segments_path, *, seed=0, drop_bins=(), nan_bins=(), fixed=None, name="env.parquet"):
    """ERA5 columns per segment row as the assembled file stores them (float32); `fixed` pins columns to constants."""
    frame = pd.read_parquet(segments_path)[["segment_id", "bin_index"]].reset_index(drop=True)
    values = np.random.default_rng(seed).uniform(1.0, 10.0, size=(len(frame), len(era5.RAW)))
    frame = frame.assign(**{name: values[:, i].astype(np.float32) for i, name in enumerate(era5.RAW)})
    for column, value in (fixed or {}).items():
        frame[column] = np.float32(value)
    frame.loc[frame["bin_index"].isin(nan_bins), "tp"] = np.nan
    frame = frame[~frame["bin_index"].isin(drop_bins)]
    path = tmp_path / name
    frame.to_parquet(path, index=False)
    return path


def _env_windows(segments_path, env_path, stats=UNIT):
    windows = HorizonWindows.build(segments_path, ["a"], context=12, core=12, horizons=HORIZONS, bin_seconds=HOUR,
                                   env_parquet=env_path)
    return replace(windows, env_stats=stats)


def _destinations(windows):
    """The row each token lands on; one segment from bin 0, so rows are bins."""
    return windows.anchors[:, None] + np.arange(-10, 1)[None, :]


def test_env_off_query_is_unchanged_and_env_on_only_adds_columns(tmp_path, monkeypatch):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(40))})
    env = _env_parquet(tmp_path, seg)
    seen, connect = [], duckdb.connect

    class Spy:
        def __init__(self, con):
            self.con = con

        def execute(self, sql, *args):
            seen.append(sql)
            return self.con.execute(sql, *args)

        def register(self, *args):
            return self.con.register(*args)

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: Spy(connect(*a, **k)))
    off = _read_segments(seg, ["a"])
    assert seen[-1] == ENV_OFF_SQL.format(path=seg)
    on = _read_segments(seg, ["a"], env_parquet=env)
    assert "LEFT JOIN" in seen[-1] and list(on.columns) == list(off.columns) + list(era5.RAW)
    pd.testing.assert_frame_equal(on[off.columns], off)


def test_env_appends_15_channels_after_the_unchanged_movement_channels(tmp_path):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(60))})
    off, on = _horizon_windows(seg), _env_windows(seg, _env_parquet(tmp_path, seg))
    ids = np.arange(off.n_windows)
    assert on.n_windows == off.n_windows and np.array_equal(on.anchors, off.anchors)
    got, expected = on.gather(ids), off.gather(ids)
    assert got[0].shape == (len(ids), 11, 12 + era5.N_ENV_CHANNELS) == (len(ids), 11, 27)
    assert np.array_equal(got[0][..., :12], expected[0])
    for a, b in zip(got[1:], expected[1:]):
        assert np.array_equal(a, b)
    raw = on.env_tokens(ids)
    assert np.array_equal(got[0][..., 12:], raw) and (raw[..., -1] == 1).all()
    np.testing.assert_array_equal(raw[..., : len(era5.RAW)], on.env[_destinations(on)])
    mean, sd = np.arange(1.0, 15.0), np.arange(2.0, 16.0)
    scaled = replace(on, env_stats=(mean, sd)).gather(ids)[0]
    np.testing.assert_allclose(scaled[..., 12:26], (raw[..., :-1] - mean) / sd, rtol=1e-12)
    assert np.array_equal(scaled[..., 26], raw[..., -1])


def test_env_is_read_at_context_fixes_only(tmp_path):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(60))})
    base = _env_windows(seg, _env_parquet(tmp_path, seg))
    ids = np.arange(base.n_windows)
    reference = base.gather(ids)[0]
    for w in (0, 17, base.n_windows - 1):
        anchor, targets = base.anchors[w], base.targets[w][base.targets[w] >= 0]
        others = np.setdiff1d(np.arange(len(base.env)), _destinations(base)[w])
        # the window's target rows, every row after its anchor and its first context fix (no token lands there)
        assert np.isin(targets, others).all() and np.isin(np.arange(anchor + 1, len(base.env)), others).all()
        assert anchor - 11 in others
        perturbed = base.env.copy()
        perturbed[others] += 1000.0
        assert np.array_equal(replace(base, env=perturbed).gather([w])[0], reference[[w]])
    row = int(base.anchors[5]) - 3
    perturbed = base.env.copy()
    perturbed[row] *= 2.0
    features = replace(base, env=perturbed).gather(ids)[0]
    changed = (features != reference).any(axis=-1)
    assert changed.any() and np.array_equal(changed, _destinations(base) == row)
    assert np.array_equal(features[..., :12], reference[..., :12])


def _straight(tmp_path, d_lon, d_lat, name):
    """A clean track on a 1 h grid: exactly east, exactly north, or still."""
    bins = np.arange(30)
    path = tmp_path / name
    pd.DataFrame({"segment_id": "a_s0", "individual_id": "a", "bin_index": bins,
                  "timestamp": [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=int(b)) for b in bins],
                  "lon": 10.0 + d_lon * bins, "lat": 50.0 + d_lat * bins}).to_parquet(path)
    return path


@pytest.mark.parametrize("track, expected", [
    ((0.01, 0.0), (5.0, 2.0, 7.0, -3.0)),  # east: tail = u, cross = v (positive to the left, north)
    ((0.0, 0.01), (2.0, -5.0, -3.0, -7.0)),  # north: tail = v, cross = -u (left is west)
    ((0.0, 0.0), (0.0, 0.0, 0.0, 0.0)),  # still: heading gated, so no tail or cross wind
])
def test_tail_and_cross_wind_follow_each_tokens_heading(tmp_path, track, expected):
    seg = _straight(tmp_path, *track, name="track.parquet")
    env = _env_parquet(tmp_path, seg, fixed={"u10": 5.0, "v10": 2.0, "u100": 7.0, "v100": -3.0})
    windows = _env_windows(seg, env)
    features = windows.gather(np.arange(windows.n_windows))[0]
    derived = features[..., [12 + era5.ENV_CHANNELS.index(c) for c in era5.DERIVED]]
    assert np.array_equal(derived, np.broadcast_to(np.array(expected), derived.shape))
    assert (features[..., -1] == 1).all()


def test_missing_era5_keeps_the_window_with_zeros_and_flag_0(tmp_path):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(40))})
    env = _env_parquet(tmp_path, seg, drop_bins=(20,), nan_bins=(25,))  # no env row at 20; tp missing at 25
    on = _env_windows(seg, env, stats=(np.full(era5.N_STANDARDISED, 5.0), np.full(era5.N_STANDARDISED, 2.0)))
    assert np.isnan(on.env[20]).all() and np.isnan(on.env[25]).sum() == 1
    assert on.n_windows == _horizon_windows(seg).n_windows
    channels = on.gather(np.arange(on.n_windows))[0][..., 12:]
    missing = np.isin(_destinations(on), [20, 25])
    assert missing.any() and (channels[missing] == 0).all()
    assert (channels[~missing][:, -1] == 1).all() and (channels[~missing][:, :-1] != 0).any(axis=-1).all()


def test_env_refusals(tmp_path):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(40))})
    env = _env_parquet(tmp_path, seg)
    with pytest.raises(ValueError, match="core only"):
        HorizonWindows.build(seg, ["a"], context=24, core=12, horizons=HORIZONS, bin_seconds=HOUR, env_parquet=env)
    with pytest.raises(ValueError, match="env_stats"):
        _env_windows(seg, env, stats=None).gather(np.arange(3))
    with pytest.raises(ValueError, match="no covariates"):
        _horizon_windows(seg).env_tokens(np.arange(3))
    twice = pd.read_parquet(env)
    twice = pd.concat([twice, twice.iloc[:1]])
    twice.to_parquet(tmp_path / "twice.parquet", index=False)
    with pytest.raises(ValueError, match="repeats"):
        _env_windows(seg, tmp_path / "twice.parquet")

def test_env_channels_refuse_a_missing_history_position(tmp_path):
    seg = _hourly(tmp_path, {"a_s0": ("a", np.arange(40))})
    on = _env_windows(seg, _env_parquet(tmp_path, seg))
    ids = np.arange(3)
    features = on.gather(ids)[0][..., :12]
    rows = _destinations(on)[ids]
    np.testing.assert_array_equal(on._env_channels(rows, features), on.env_tokens(ids))
    rows[1, 0] = -1  # direct_anchors never leaves a hole in the core; this stands in for one
    with pytest.raises(ValueError, match="missing history position"):
        on._env_channels(rows, features)
