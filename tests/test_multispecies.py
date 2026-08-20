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
