# GenAI reference: Used GenAI for these tests, from the behaviours I specified; that only steps within a segment and
# exactly one bin apart are counted, that the threshold is 10% of the median over the supplied training individuals
# only, that rho matches the analysis.randomness definition, and the dtype/empty/edge cases on restrict_to_individuals
# and ensure_resampled. Every case moves north only, so expected step lengths are just a latitude difference times
# M_PER_DEG_LAT and can be checked analytically. I validated the correctness of them all.
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.analysis.regime_profile import (
    DESCRIPTOR_TRANSFORMS,
    DESCRIPTORS,
    LOW_DISPLACEMENT_FRACTION,
    LOW_FRACTION_GUARD,
    M_PER_DEG_LAT,
    MIN_MIXTURE_STEPS,
    STORK_REFERENCE_BOUNDARY_M,
    apply_threshold_decision,
    cadence_steps,
    coverage_distances,
    descriptor_standardisation,
    transform_descriptors,
    directional_persistence_from_turns,
    ensure_resampled,
    fit_step_mixture,
    is_low_displacement,
    local_en_metres,
    low_displacement_threshold_m,
    median_step_m,
    normalise_individual_ids,
    profile_species,
    restrict_to_individuals,
    select_threshold_rule,
)
from animal_movement_fm.data.resample import BIN_SECONDS

SIX_HOURS = pd.Timedelta(seconds=BIN_SECONDS)


def make_frame(specs):
    """Build a resampled segment frame from (segment_id, individual_id, bin_offsets, lats) specs.

    `lats` moves the animal north only, so a step's length in metres is just the latitude
    difference times M_PER_DEG_LAT and every expected value below can be written by hand.
    """
    rows = []
    base_bin = 1_000_000
    base_time = pd.Timestamp("2020-01-01 03:00:00")
    for segment_id, individual_id, bin_offsets, lats in specs:
        for offset, lat in zip(bin_offsets, lats, strict=True):
            rows.append(
                {
                    "segment_id": segment_id,
                    "individual_id": individual_id,
                    "bin_index": base_bin + offset,
                    "timestamp": base_time + offset * SIX_HOURS,
                    "lat": lat,
                    "lon": 0.0,
                }
            )
    return pd.DataFrame(rows)


def metres_to_degrees(metres):
    return metres / M_PER_DEG_LAT


# --------------------------------------------------------------------------- geometry


def test_local_en_metres_north_matches_degree_constant():
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 0.0])
    east, north = local_en_metres(lon, lat)
    assert north[0] == pytest.approx(M_PER_DEG_LAT)
    assert east[0] == pytest.approx(0.0)


def test_local_en_metres_longitude_compresses_with_latitude():
    """One degree of longitude at 60 degrees north is about half its equatorial length."""
    equator_east, _ = local_en_metres(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
    high_east, _ = local_en_metres(np.array([0.0, 1.0]), np.array([60.0, 60.0]))
    assert high_east[0] / equator_east[0] == pytest.approx(0.5, abs=1e-3)


def test_local_en_metres_wraps_across_the_antimeridian():
    east, _ = local_en_metres(np.array([179.9, -179.9]), np.array([0.0, 0.0]))
    assert east[0] == pytest.approx(0.2 * 111_320.0)


# ------------------------------------------------------------------- step extraction


def test_cadence_steps_drops_multi_bin_steps():
    """THE load-bearing rule: a step across an internal gap is not a step at the cadence."""
    lats = [0.0, metres_to_degrees(100.0), metres_to_degrees(1100.0)]
    frame = make_frame([("s1", 1, [0, 1, 4], lats)])  # second step spans three bins

    steps = cadence_steps(frame)

    assert len(steps) == 1
    assert steps.mag_m[0] == pytest.approx(100.0, rel=1e-6)


def test_cadence_steps_drops_steps_crossing_segments():
    frame = make_frame(
        [
            ("s1", 1, [0, 1], [0.0, metres_to_degrees(100.0)]),
            ("s2", 1, [2, 3], [10.0, 10.0 + metres_to_degrees(200.0)]),
        ]
    )

    steps = cadence_steps(frame)

    # one step from each segment, and nothing bridging the 10-degree jump between them
    assert sorted(np.round(steps.mag_m, 6)) == [pytest.approx(100.0), pytest.approx(200.0)]


def test_cadence_steps_tags_hour_and_month_on_the_destination_fix():
    frame = make_frame([("s1", 1, [0, 1], [0.0, metres_to_degrees(100.0)])])

    steps = cadence_steps(frame)

    destination = frame["timestamp"].iloc[1]
    assert steps.hour[0] == destination.hour
    assert steps.month[0] == destination.month


def test_cadence_steps_requires_resampling_first():
    frame = make_frame([("s1", 1, [0, 1], [0.0, 1.0])]).drop(columns=["bin_index"])
    with pytest.raises(ValueError, match="resampled"):
        cadence_steps(frame)


def test_cadence_steps_drops_nonfinite_positions():
    frame = make_frame([("s1", 1, [0, 1, 2], [0.0, np.nan, 1.0])])
    assert len(cadence_steps(frame)) == 0


# ----------------------------------------------------------------------- the threshold


def test_low_displacement_boundary_is_a_tenth_of_the_one_bin_median():
    lengths = [100.0, 200.0, 300.0, 400.0, 500.0]
    lats = np.cumsum([0.0] + [metres_to_degrees(v) for v in lengths])
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    assert median_step_m(frame) == pytest.approx(300.0, rel=1e-6)
    assert low_displacement_threshold_m(frame, individual_ids=[1]) == pytest.approx(30.0, rel=1e-6)
    assert LOW_DISPLACEMENT_FRACTION == 0.10


def test_multi_bin_steps_would_inflate_the_median_if_admitted():
    """Documents the distinction amendment A1 turns on, rather than only asserting the fix.

    The gap step is deliberately long, as real ones are. Admitting it would raise the median;
    the registered definition excludes it.
    """
    frame = make_frame(
        [
            (
                "s1",
                1,
                [0, 1, 2, 5],
                [0.0] + list(np.cumsum([metres_to_degrees(v) for v in (100.0, 200.0, 9000.0)])),
            ),
        ]
    )

    steps = cadence_steps(frame)
    assert sorted(np.round(steps.mag_m, 3)) == [pytest.approx(100.0), pytest.approx(200.0)]
    assert median_step_m(frame) == pytest.approx(150.0, rel=1e-6)

    naive_median = float(np.median([100.0, 200.0, 9000.0]))
    assert naive_median > median_step_m(frame)


def test_threshold_uses_only_the_supplied_individuals():
    """Train-only is not cosmetic: a different individual set gives a different threshold."""
    quiet = make_frame(
        [
            (
                "s1",
                1,
                [0, 1, 2],
                [0.0] + list(np.cumsum([metres_to_degrees(v) for v in (100.0, 100.0)])),
            )
        ]
    )
    loud = make_frame(
        [
            (
                "s2",
                2,
                [0, 1, 2],
                [0.0] + list(np.cumsum([metres_to_degrees(v) for v in (900.0, 900.0)])),
            )
        ]
    )
    frame = pd.concat([quiet, loud], ignore_index=True)

    assert median_step_m(frame, individual_ids=[1]) == pytest.approx(100.0, rel=1e-6)
    assert median_step_m(frame, individual_ids=[2]) == pytest.approx(900.0, rel=1e-6)
    assert median_step_m(frame) == pytest.approx(500.0, rel=1e-6)


def test_restrict_to_individuals_matches_across_id_dtypes():
    """A manifest holds ints; a parquet may hold strings. A silent mismatch must not pass."""
    frame = make_frame([("s1", 1, [0, 1], [0.0, metres_to_degrees(100.0)])])
    frame["individual_id"] = frame["individual_id"].astype(str)

    assert len(restrict_to_individuals(frame, [1])) == 2


def test_restrict_to_individuals_raises_rather_than_returning_empty():
    frame = make_frame([("s1", 1, [0, 1], [0.0, metres_to_degrees(100.0)])])
    with pytest.raises(ValueError, match="no rows"):
        restrict_to_individuals(frame, [999])


def test_is_low_displacement_is_strictly_below_the_boundary():
    mags = np.array([9.0, 10.0, 11.0])
    assert list(is_low_displacement(mags, 10.0)) == [True, False, False]


@pytest.mark.parametrize(
    ("turns", "expected"),
    [
        ([0.0, 0.0], 1.0),
        ([np.pi / 2, -np.pi / 2], 0.0),
        ([np.pi, -np.pi], -1.0),
    ],
    ids=("straight", "right-angle", "reversal"),
)
def test_directional_persistence_matches_randomness_definition(turns, expected):
    assert directional_persistence_from_turns(turns) == pytest.approx(expected, abs=1e-12)


def test_directional_persistence_without_turns_is_nan():
    assert np.isnan(directional_persistence_from_turns([]))


# --------------------------------------------------------------------------- profiling


def test_profile_species_returns_every_descriptor():
    rng = np.random.default_rng(0)
    lengths = np.abs(rng.normal(500.0, 200.0, size=400)) + 1.0
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(lengths)]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    # the descriptor quality gates are exercised by their own tests below; here they are lowered so
    # a 400-step fixture does not silently make this a test of the gates instead of the descriptors
    profile = profile_species(
        frame, individual_ids=[1], min_turns_for_rho=1, min_steps_per_amplitude_bin=1
    )

    for key in DESCRIPTORS:
        assert key in profile
        assert not np.isnan(profile[key]), f"{key} came out NaN"
    assert profile["n_steps_at_cadence"] == len(lengths)


def test_intermittency_counts_the_low_displacement_fraction():
    """Half the steps sit far below a tenth of the median, half far above."""
    lengths = [1.0] * 50 + [1000.0] * 50
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(np.array(lengths))]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    profile = profile_species(frame, individual_ids=[1])

    assert profile["intermittency"] == pytest.approx(0.5, abs=0.01)


def test_profile_reports_the_same_boundary_as_the_boundary_function():
    """The profiler and the standalone function must not drift into two copies of the rule."""
    lengths = [100.0, 200.0, 300.0]
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(np.array(lengths))]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    assert profile_species(frame, individual_ids=[1])[
        "median_anchor_boundary_m"
    ] == pytest.approx(low_displacement_threshold_m(frame, individual_ids=[1]))


def test_profile_reports_the_registered_boundary_fields():
    lengths = np.array([10.0, 20.0, 1000.0, 1200.0])
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(lengths)]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    profile = profile_species(frame, individual_ids=[1])

    assert profile["median_anchor_multiplier"] == LOW_DISPLACEMENT_FRACTION
    assert profile["median_anchor_boundary_m"] == pytest.approx(
        LOW_DISPLACEMENT_FRACTION * profile["step_p50_m"]
    )
    assert profile["median_anchor_low_fraction"] == pytest.approx(profile["intermittency"])
    assert profile["step_p50_m"] < profile["step_p75_m"] < profile["step_p90_m"]


def test_profile_without_individual_ids_uses_every_individual():
    """The CLI documents profiling a candidate that has no frozen split yet, so `individual_ids=None`
    must mean "all individuals" rather than raising. Such a row is marked
    threshold_decision_ready=False and cannot feed the A4 decision."""
    quiet = make_frame(
        [("s1", 1, [0, 1, 2], [0.0, *np.cumsum([metres_to_degrees(100.0)] * 2)])]
    )
    loud = make_frame(
        [("s2", 2, [0, 1, 2], [0.0, *np.cumsum([metres_to_degrees(900.0)] * 2)])]
    )
    frame = pd.concat([quiet, loud], ignore_index=True)

    profile = profile_species(frame, min_turns_for_rho=1, min_steps_per_amplitude_bin=1)

    assert profile["n_individuals"] == 2
    assert profile["step_p50_m"] == pytest.approx(median_step_m(frame), rel=1e-6)


def test_profile_emits_no_stationary_named_fields():
    """A4 renamed the stratum; two vocabularies in results/regime_profiles.csv would let a reader
    cite one column and stratify by the other."""
    lengths = np.array([10.0, 20.0, 1000.0, 1200.0])
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(lengths)]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    profile = profile_species(frame, individual_ids=[1])

    assert [key for key in profile if "stationary" in key] == []


# ------------------------------------------------------------------------ resampling in


def test_ensure_resampled_passes_through_a_frame_that_already_has_bins():
    frame = make_frame([("s1", 1, [0, 1], [0.0, 1.0])])
    assert ensure_resampled(frame) is frame


def test_ensure_resampled_builds_bins_from_raw_fixes():
    """Raw fixes go through the pipeline's own resampler, so the profiler cannot drift from it."""
    n = 300
    stamps = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n) * 6, unit="h")
    raw = pd.DataFrame(
        {
            "individual_id": 1,
            "timestamp": stamps,
            "lat": np.cumsum(np.full(n, metres_to_degrees(500.0))),
            "lon": 0.0,
        }
    )

    resampled = ensure_resampled(raw)

    assert "bin_index" in resampled.columns
    assert "segment_id" in resampled.columns
    assert len(resampled) == n


def test_ensure_resampled_rejects_a_frame_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        ensure_resampled(pd.DataFrame({"lat": [0.0], "lon": [0.0]}))


# --------------------------------------------------------------------------- coverage


def _descriptor_columns(levels):
    """
    Descriptor columns placed on each descriptor's own valid domain, driven by one level per species.

    Fixtures used to fill every descriptor with the same arbitrary numbers, including zeros and
    negatives. That stopped being harmless once descriptors are transformed by measurement scale: a
    tail ratio of 0 has no log10 and an intermittency of 1.5 is not a proportion. Driving all eight
    from a single level keeps these tests testing what they are named for -- symmetry, outlier
    ranking, frozen stability -- while every value stays a value the descriptor could actually take.
    """
    levels = np.asarray(levels, dtype=float)
    columns = {}
    for descriptor in DESCRIPTORS:
        kind = DESCRIPTOR_TRANSFORMS[descriptor]
        if kind == "log10":  # ratio scale, strictly positive
            columns[descriptor] = 1.0 + 10.0**levels
        elif kind == "logit":  # a proportion, strictly inside (0, 1)
            columns[descriptor] = 0.05 + 0.9 / (1.0 + np.exp(-levels))
        else:
            columns[descriptor] = levels
    return columns


def test_coverage_distances_is_symmetric_with_a_zero_diagonal():
    rng = np.random.default_rng(1)
    table = pd.DataFrame(
        {"species": ["a", "b", "c"], **_descriptor_columns(rng.normal(size=3))}
    )

    dist = coverage_distances(table)

    assert np.allclose(np.diag(dist.to_numpy()), 0.0)
    assert np.allclose(dist.to_numpy(), dist.to_numpy().T)


def test_coverage_distances_ranks_the_outlier_furthest():
    table = pd.DataFrame(
        {
            "species": ["near_a", "near_b", "outlier"],
            **_descriptor_columns([0.0, 0.1, 9.0]),
        }
    )

    dist = coverage_distances(table)

    assert dist.loc["near_a", "outlier"] > dist.loc["near_a", "near_b"]


def test_coverage_distances_rejects_missing_descriptors_rather_than_imputing():
    """Imputing a NaN to the column mean makes an unmeasurable species look perfectly average.

    That is not a neutral default when the distances feed species selection, so a missing descriptor
    is an eligibility failure that must be raised, not filled.
    """
    table = pd.DataFrame(
        {"species": ["a", "b"], **_descriptor_columns([0.0, 1.0])}
    )
    table.loc[1, "rho"] = np.nan

    with pytest.raises(ValueError, match="missing descriptors"):
        coverage_distances(table)


def test_frozen_standardisation_keeps_a_pair_distance_stable_as_species_are_added():
    """Standardising against whatever rows are present makes a measured quantity move.

    Secondary prediction 2 uses regime distance as the explanatory variable for uneven transfer, so
    the same pair must not change distance when an unrelated candidate is profiled.
    """
    first = pd.DataFrame({"species": ["a", "b"], **_descriptor_columns([0.0, 1.0])})
    later = pd.DataFrame(
        {"species": ["a", "b", "c"], **_descriptor_columns([0.0, 1.0, 50.0])}
    )

    frozen = descriptor_standardisation(first)

    drifting = coverage_distances(later).loc["a", "b"]
    stable = coverage_distances(later, standardisation=frozen).loc["a", "b"]

    assert drifting != pytest.approx(coverage_distances(first).loc["a", "b"], rel=1e-6)
    assert stable == pytest.approx(coverage_distances(first, standardisation=frozen).loc["a", "b"])


# --------------------------------------------------------------- identifiers across sources


def test_normalise_ids_matches_int_float_and_string_spellings():
    assert list(normalise_individual_ids([12345, 12345.0, "12345", " 12345 "])) == ["12345"] * 4


def test_normalise_ids_keeps_distinct_ids_distinct():
    """Truncating non-integral values would silently merge two different animals."""
    assert len(set(normalise_individual_ids([1.5, 1.7, 2]))) == 3


def test_restrict_to_individuals_handles_raatd_string_ids():
    """RAATD ids look like "CAR2013_sel_a_m_03". Integer coercion emptied the frame for the entire
    marine leg, and reported it as a manifest mismatch."""
    frame = make_frame([("s1", "CAR2013_sel_a_m_03", [0, 1], [0.0, metres_to_degrees(100.0)])])

    kept = restrict_to_individuals(frame, ["CAR2013_sel_a_m_03"])

    assert len(kept) == 2
    assert median_step_m(frame, individual_ids=["CAR2013_sel_a_m_03"]) == pytest.approx(100.0)


def test_restrict_to_individuals_matches_a_float_column_against_an_int_manifest():
    frame = make_frame([("s1", 1, [0, 1], [0.0, metres_to_degrees(100.0)])])
    frame["individual_id"] = frame["individual_id"].astype(float)

    assert len(restrict_to_individuals(frame, [1])) == 2


# ------------------------------------------------------------------ descriptor quality gates


def test_rho_is_withheld_when_too_few_turn_pairs_qualify():
    lengths = np.array([100.0, 900.0, 100.0, 900.0, 100.0])
    lats = np.cumsum(np.concatenate([[0.0], metres_to_degrees(lengths)]))
    frame = make_frame([("s1", 1, list(range(len(lats))), lats)])

    profile = profile_species(frame, individual_ids=[1], min_steps_per_amplitude_bin=1)

    assert profile["n_turns_used"] < 100
    assert np.isnan(profile["rho"])


def test_amplitude_ignores_thinly_populated_temporal_bins():
    """max-minus-min is a statistic of the extremes, so one sparse hour would otherwise dominate.

    Two well-populated hour bins carry ordinary steps; a third holds a single wild one. With the
    gate on, the sparse bin is dropped and the amplitude reflects the real diel range.
    """
    specs = []
    # 60 one-step segments landing at 09:00, then 60 landing at 15:00, then ONE at 21:00
    for i in range(60):
        specs.append((f"a{i}", 1, [0, 1], [0.0, metres_to_degrees(500.0)]))
    for i in range(60):
        specs.append((f"b{i}", 1, [1, 2], [0.0, metres_to_degrees(600.0)]))
    specs.append(("c0", 1, [2, 3], [0.0, metres_to_degrees(50_000.0)]))
    frame = make_frame(specs)

    gated = profile_species(frame, individual_ids=[1], min_turns_for_rho=1)
    ungated = profile_species(
        frame, individual_ids=[1], min_turns_for_rho=1, min_steps_per_amplitude_bin=1
    )

    assert gated["diel_amplitude"] < ungated["diel_amplitude"]


# --------------------------------------------------------------------- the mixture diagnostic


def bimodal_steps(n=4000, seed=0):
    """A genuinely two-mode step distribution: a low displacement mode and a travel mode."""
    rng = np.random.default_rng(seed)
    low = rng.lognormal(mean=np.log(5.0), sigma=0.4, size=n // 2)
    travel = rng.lognormal(mean=np.log(2000.0), sigma=0.4, size=n // 2)
    return np.concatenate([low, travel])


def test_mixture_is_supported_and_crosses_between_the_modes():
    mixture = fit_step_mixture(bimodal_steps())

    assert mixture.supported
    assert 5.0 < mixture.crossing_m < 2000.0
    assert mixture.bic_improvement > 10.0
    assert mixture.crossing_fraction == pytest.approx(0.5, abs=0.05)


def test_mixture_is_unsupported_on_a_single_mode():
    rng = np.random.default_rng(1)
    mixture = fit_step_mixture(rng.lognormal(mean=np.log(500.0), sigma=0.5, size=4000))

    assert not mixture.supported
    assert np.isnan(mixture.crossing_m)


def test_mixture_is_unsupported_below_the_registered_step_floor():
    """Rejected for the sample-size reason specifically, not merely rejected.

    The assertion is on the registered constant rather than the wording, so rephrasing the message
    does not fail the test while a change to the threshold itself still does.
    """
    mixture = fit_step_mixture(bimodal_steps(n=MIN_MIXTURE_STEPS - 2))

    assert not mixture.supported
    assert mixture.n_steps == MIN_MIXTURE_STEPS - 2
    assert str(MIN_MIXTURE_STEPS) in mixture.reason
    assert np.isnan(mixture.crossing_m)


def test_mixture_is_unsupported_when_a_component_is_negligible():
    rng = np.random.default_rng(2)
    bulk = rng.lognormal(mean=np.log(500.0), sigma=0.4, size=4000)
    sliver = rng.lognormal(mean=np.log(2.0), sigma=0.1, size=5)  # ~0.1% weight
    mixture = fit_step_mixture(np.concatenate([bulk, sliver]))

    assert not mixture.supported


def test_zero_length_steps_are_floored_not_dropped():
    steps = np.concatenate([np.zeros(50), bimodal_steps(n=2000)])
    mixture = fit_step_mixture(steps)

    assert mixture.n_floor_clipped == 50
    assert mixture.n_steps == len(steps)


# ------------------------------------------------------------- the A4 threshold decision tree


def decision_row(species, p50, p75, p90, low_fraction, *, crossing=np.nan, supported=False):
    return {
        "species": species,
        "step_p50_m": p50,
        "step_p75_m": p75,
        "step_p90_m": p90,
        "median_anchor_low_fraction": low_fraction,
        "mixture_supported": supported,
        "mixture_crossing_m": crossing,
    }


STORK = decision_row("white_stork", 609.5, 1500.0, 4000.0, 0.311)


def test_default_branch_retains_the_registered_median_rule():
    table = pd.DataFrame([STORK, decision_row("walrus", 400.0, 900.0, 2500.0, 0.30)])

    decision = select_threshold_rule(table)

    assert decision.anchor_quantile == 50
    assert decision.multiplier == LOW_DISPLACEMENT_FRACTION
    assert decision.pooled_only_species == ()


# the measured stork training median, from the real profile run against splits/stork_split_seed42.json
# over 879,559 one-bin steps. A1 quotes it rounded to 609.5 m; the unrounded value is what reproduces
# the published 60.9 m boundary, since 0.10 x 609.5 would give 60.95.
STORK_MEASURED_P50_M = 609.4608310684118


def test_stork_boundary_under_the_default_branch_is_the_a1_figure():
    """The default branch must still reproduce 60.9 m, or Tier 1 and Tier 2 stop being comparable."""
    table = pd.DataFrame([{**STORK, "step_p50_m": STORK_MEASURED_P50_M}])
    decision = select_threshold_rule(table)
    applied = apply_threshold_decision(table, decision)

    boundary = float(applied.loc[0, "selected_boundary_m"])

    assert boundary == pytest.approx(60.946083106841186, rel=1e-12)
    assert round(boundary, 1) == STORK_REFERENCE_BOUNDARY_M


def test_low_fraction_guard_moves_every_species_to_p75():
    """One species tripping the guard changes the rule for ALL of them; that is the whole point."""
    table = pd.DataFrame(
        [STORK, decision_row("seal", 100.0, 800.0, 3000.0, LOW_FRACTION_GUARD + 0.01)]
    )

    decision = select_threshold_rule(table)

    assert decision.anchor_quantile == 75
    assert any("low fraction" in r for r in decision.reasons)


def test_the_p75_multiplier_is_calibrated_to_preserve_the_stork_boundary():
    table = pd.DataFrame([STORK, decision_row("seal", 100.0, 800.0, 3000.0, 0.60)])

    decision = select_threshold_rule(table)
    applied = apply_threshold_decision(table, decision).set_index("species")

    assert applied.loc["white_stork", "selected_boundary_m"] == pytest.approx(
        STORK_REFERENCE_BOUNDARY_M
    )
    # and the same universal formula is applied to the other species
    assert applied.loc["seal", "selected_boundary_m"] == pytest.approx(
        decision.multiplier * 800.0
    )


def test_p50_inside_a_supported_low_component_also_trips_the_fallback():
    table = pd.DataFrame(
        [
            STORK,
            decision_row("seal", 100.0, 800.0, 3000.0, 0.20, crossing=150.0, supported=True),
        ]
    )

    decision = select_threshold_rule(table)

    assert decision.anchor_quantile == 75
    assert any("p50 inside" in r for r in decision.reasons)


def test_p75_inside_a_supported_low_component_moves_everyone_to_p90():
    table = pd.DataFrame(
        [
            STORK,
            decision_row("seal", 100.0, 800.0, 3000.0, 0.20, crossing=900.0, supported=True),
        ]
    )

    decision = select_threshold_rule(table)

    assert decision.anchor_quantile == 90
    assert decision.multiplier == pytest.approx(STORK_REFERENCE_BOUNDARY_M / 4000.0)
    assert decision.pooled_only_species == ()


def test_p90_inside_the_low_component_is_reported_pooled_not_given_its_own_rule():
    table = pd.DataFrame(
        [
            STORK,
            decision_row("seal", 100.0, 800.0, 3000.0, 0.20, crossing=5000.0, supported=True),
        ]
    )

    decision = select_threshold_rule(table)
    applied = apply_threshold_decision(table, decision).set_index("species")

    assert decision.anchor_quantile == 90
    assert decision.pooled_only_species == ("seal",)
    assert bool(applied.loc["seal", "pooled_only"]) is True
    assert bool(applied.loc["white_stork", "pooled_only"]) is False
    # the pooled species does NOT get a bespoke formula; it keeps the universal one
    assert applied.loc["seal", "selected_multiplier"] == decision.multiplier


def test_an_unsupported_mixture_never_trips_a_guard():
    """An unsupported fit is not evidence in either direction, so it must not move the rule."""
    table = pd.DataFrame(
        [
            STORK,
            decision_row("seal", 100.0, 800.0, 3000.0, 0.20, crossing=5000.0, supported=False),
        ]
    )

    decision = select_threshold_rule(table)

    assert decision.anchor_quantile == 50


def test_string_booleans_from_a_csv_round_trip_are_read_as_booleans():
    """The table is written to and re-read from CSV, so mixture_supported arrives as text."""
    table = pd.DataFrame(
        [
            STORK,
            decision_row("seal", 100.0, 800.0, 3000.0, 0.20, crossing=150.0, supported=True),
        ]
    )
    table["mixture_supported"] = table["mixture_supported"].astype(str)

    assert select_threshold_rule(table).anchor_quantile == 75


def test_duplicate_species_rows_are_rejected():
    table = pd.DataFrame([STORK, STORK])
    with pytest.raises(ValueError, match="duplicate species"):
        select_threshold_rule(table)


def test_out_of_order_quantiles_are_rejected():
    table = pd.DataFrame([decision_row("white_stork", 900.0, 500.0, 4000.0, 0.20)])
    with pytest.raises(ValueError, match="ordered"):
        select_threshold_rule(table)


def test_a_fallback_without_the_reference_species_is_rejected():
    """The fallback multiplier is calibrated on stork; without it there is no registered constant."""
    table = pd.DataFrame([decision_row("seal", 100.0, 800.0, 3000.0, 0.80)])

    with pytest.raises(ValueError, match="white_stork"):
        select_threshold_rule(table)


def test_an_empty_profile_table_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        select_threshold_rule(pd.DataFrame(columns=list(STORK)))


# ------------------------------------------------------------------------------- naming


def test_the_module_no_longer_exports_the_old_stationary_vocabulary():
    """A4 renamed the stratum because a step-length threshold cannot carry a behavioural claim.

    Tier 2 code has not been written yet, so leaving the old names importable is how the rejected
    vocabulary would get picked up again. Tier 1 prose keeps "stationary"; this module does not.
    """
    import animal_movement_fm.analysis.regime_profile as module

    assert [name for name in vars(module) if "stationary" in name.lower()] == []


# ------------------------------------------------------------------ descriptor transforms


def test_each_descriptor_is_transformed_by_what_the_quantity_is():
    """
    The rule is measurement scale, not outcome: ratio scale -> log10, proportion -> logit, bounded
    correlation -> raw. Pinned as a table because choosing transforms by where they move a distance
    would be supervised feature selection with taxon as the target.
    """
    assert DESCRIPTOR_TRANSFORMS == {
        "log10_step_median_m": "identity",  # the column is already a log10
        "log10_step_p90_m": "identity",
        "tail_ratio": "log10",
        "step_cv": "log10",
        "intermittency": "logit",
        "rho": "identity",  # log is undefined on the negatives, and rho does go negative
        "diel_amplitude": "log10",
        "seasonal_amplitude": "log10",
    }

    values = pd.DataFrame(
        {"tail_ratio": [10.0], "intermittency": [0.25], "rho": [-0.3], "log10_step_p90_m": [4.0]}
    )
    out = transform_descriptors(values)
    assert out.loc[0, "tail_ratio"] == pytest.approx(1.0)
    assert out.loc[0, "intermittency"] == pytest.approx(np.log(0.25 / 0.75))
    assert out.loc[0, "rho"] == pytest.approx(-0.3)
    assert out.loc[0, "log10_step_p90_m"] == pytest.approx(4.0)


def test_a_non_positive_ratio_descriptor_is_rejected_not_floored():
    """Flooring would invent a value for a species whose profile is wrong. Fix the profile."""
    with pytest.raises(ValueError, match="non-positive values"):
        transform_descriptors(pd.DataFrame({"tail_ratio": [0.0]}))


def test_a_standardisation_frozen_before_the_transform_rule_is_refused():
    """Centres taken on raw descriptors do not describe transformed ones; mixing them is silent."""
    table = pd.DataFrame({"species": ["a", "b"], **_descriptor_columns([0.0, 1.0])})
    stale = descriptor_standardisation(table)
    stale.pop("transforms")

    with pytest.raises(ValueError, match="predates the descriptor transform rule"):
        coverage_distances(table, standardisation=stale)


def test_the_a4_branch_cannot_move_when_a_descriptor_transform_changes():
    """
    A4 reads step quantiles, the low fraction and the mixture columns. It never reads the eight
    descriptors, so the transform rule cannot reach it -- asserted here rather than left to a
    re-run, because 'we checked and it did not move' is weaker than 'it cannot'.
    """
    table = pd.DataFrame([STORK, decision_row("walrus", 400.0, 900.0, 2500.0, 0.30)])
    for descriptor, column in _descriptor_columns([0.0, 1.0]).items():
        table[descriptor] = column

    scrambled = table.copy()
    for descriptor in DESCRIPTORS:
        scrambled[descriptor] = scrambled[descriptor] * 1000.0 + 7.0

    before, after = select_threshold_rule(table), select_threshold_rule(scrambled)
    assert before.anchor_quantile == after.anchor_quantile == 50
    assert before.multiplier == after.multiplier
    assert before.pooled_only_species == after.pooled_only_species
