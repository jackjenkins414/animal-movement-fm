# GenAI reference: I used GenAI to write these unit tests to check the movement state labeller's behaviour, in the same 
# style as the other module tests. Every case has an analytically known expected label (chosen to sit clearly 
# inside/outside the hysteresis band), so we can verify correctness by inspection. I gave GenAI the hysteresis band so it 
# could see how it was defined and turn this into analytical test cases. 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.analysis.regime import (
    MIGRATORY_NET_KM,
    REFINED_STATES,
    REGIMES,
    add_step_length_km,
    label_contiguous_regimes,
    label_migratory_launches,
    label_regimes,
    moving_bouts,
    tag_regime,
    _hysteresis_regime,
)

# Great-circle km per degree of latitude near the equator given by (R * pi/180 / 1000); pure latitude displacements at lon=0 let 
# us convert a desired step length in km into a coordinate delta
KM_PER_DEG_LAT = 6_371_000.0 * np.pi / 180.0 / 1000.0


def _segment(seg_id, step_km_list, *, start_lat=0.0, lon=0.0):
    """Build one segment whose consecutive fixes are `step_km_list` km apart in latitude.

    Returns len(step_km_list)+1 fixes, so the incoming steps are [NaN, *step_km_list].
    """
    lats = [start_lat]
    for s in step_km_list:
        lats.append(lats[-1] + s / KM_PER_DEG_LAT)
    ts = pd.date_range("2020-01-01", periods=len(lats), freq="6h")
    return pd.DataFrame(
        {"segment_id": seg_id, "timestamp": ts, "lat": lats, "lon": lon}
    )


class TestHysteresisRegime:
    def test_clear_moving_and_foraging(self):
        # >= 0.10 km is clearly moving, < 0.04 km clearly foraging.
        out = _hysteresis_regime(np.array([0.20, 0.01, 0.20]))
        assert list(out) == ["moving", "foraging", "moving"]

    def test_nan_is_unknown(self):
        out = _hysteresis_regime(np.array([np.nan, 0.20]))
        assert list(out) == ["unknown", "moving"]

    def test_grey_band_inherits_previous_clear_regime(self):
        # 0.05 km sits in the 0.04-0.10 grey band; it should inherit the prior clear state.
        assert list(_hysteresis_regime(np.array([0.20, 0.05]))) == ["moving", "moving"]
        assert list(_hysteresis_regime(np.array([0.01, 0.05]))) == ["foraging", "foraging"]

    def test_grey_band_at_start_falls_back_to_cut(self):
        # With no prior clear state, the grey band falls back to the 0.06 km cut.
        assert list(_hysteresis_regime(np.array([0.08]))) == ["moving"]      # 0.08 >= 0.06
        assert list(_hysteresis_regime(np.array([0.05]))) == ["foraging"]    # 0.05 <  0.06

    def test_boundaries(self):
        # 0.10 is moving (>= high); 0.04 is grey (not < low) -> start fallback vs 0.06 -> foraging.
        assert list(_hysteresis_regime(np.array([0.10]))) == ["moving"]
        assert list(_hysteresis_regime(np.array([0.04]))) == ["foraging"]


class TestTagRegime:
    def test_launch_and_settle_from_direction(self):
        # base:   unknown foraging  moving moving foraging
        # regime: unknown foraging  launch moving settle
        df = pd.DataFrame(
            {
                "segment_id": "a",
                "step_km": [np.nan, 0.01, 0.20, 0.20, 0.01],
            }
        )
        out = tag_regime(df)
        assert list(out["regime"]) == ["unknown", "foraging", "launch", "moving", "settle"]

    def test_unknown_prev_does_not_make_a_transition(self):
        # The first real step of a segment has an 'unknown' predecessor, so it is never a
        # launch/settle even if it is 'moving'.
        df = pd.DataFrame({"segment_id": "a", "step_km": [np.nan, 0.20]})
        out = tag_regime(df)
        assert list(out["regime"]) == ["unknown", "moving"]

    def test_transitions_respect_segment_boundaries(self):
        # Two segments concatenated; the shift must not leak the last state of 'a' into 'b'.
        df = pd.DataFrame(
            {
                "segment_id": ["a", "a", "b", "b"],
                "step_km": [np.nan, 0.20, np.nan, 0.01],
            }
        )
        out = tag_regime(df)
        assert list(out["regime"]) == ["unknown", "moving", "unknown", "foraging"]

    def test_adds_base_regime_and_regime_columns(self):
        df = pd.DataFrame({"segment_id": "a", "step_km": [np.nan, 0.01]})
        out = tag_regime(df)
        assert {"base_regime", "regime"} <= set(out.columns)


class TestAddStepLengthKm:
    def test_first_fix_of_each_segment_is_nan(self):
        df = pd.concat([_segment("a", [0.20, 0.20]), _segment("b", [0.01])], ignore_index=True)
        out = add_step_length_km(df)
        # head(1) keeps the first row per segment positionally (Series.first() would skip NaN).
        firsts = out.groupby("segment_id").head(1)["step_km"]
        assert firsts.isna().all()

    def test_step_length_matches_intended_km(self):
        out = add_step_length_km(_segment("a", [0.20, 0.01]))
        steps = out["step_km"].to_numpy()
        assert np.isnan(steps[0])
        assert steps[1] == pytest.approx(0.20, rel=1e-4)
        assert steps[2] == pytest.approx(0.01, rel=1e-4)

    def test_sorts_by_timestamp_within_segment(self):
        # Shuffled input must still produce time-ordered steps (NaN first).
        df = _segment("a", [0.20, 0.01, 0.20]).sample(frac=1.0, random_state=0)
        out = add_step_length_km(df)
        assert np.isnan(out["step_km"].to_numpy()[0])
        assert out["timestamp"].is_monotonic_increasing


class TestLabelRegimes:
    def test_end_to_end_sequence(self):
        df = _segment("a", [0.01, 0.20, 0.20, 0.01])
        out = label_regimes(df)
        assert list(out["regime"]) == ["unknown", "foraging", "launch", "moving", "settle"]
        assert {"step_km", "base_regime", "regime"} <= set(out.columns)

    def test_first_fix_unknown_per_segment(self):
        df = pd.concat(
            [_segment("a", [0.20, 0.20]), _segment("b", [0.01, 0.01])], ignore_index=True
        )
        out = label_regimes(df)
        firsts = out.groupby("segment_id").head(1)["regime"]
        assert (firsts == "unknown").all()

    def test_all_labels_are_known_regimes_or_unknown(self):
        df = _segment("a", [0.01, 0.20, 0.05, 0.20, 0.01])
        out = label_regimes(df)
        allowed = set(REGIMES) | {"unknown"}
        assert set(out["regime"]) <= allowed

    def test_invariant_to_input_row_order(self):
        df = _segment("a", [0.01, 0.20, 0.20, 0.01])
        shuffled = df.sample(frac=1.0, random_state=1)
        a = label_regimes(df).reset_index(drop=True)
        b = label_regimes(shuffled).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


class TestLabelContiguousRegimes:
    def test_requires_bin_index(self):
        with pytest.raises(ValueError, match="cadence metadata"):
            label_contiguous_regimes(_segment("a", [0.01, 0.20]))

    def test_internal_gap_resets_state_and_adds_run_id(self):
        df = _segment("a", [0.01, 0.20, 0.20, 0.01])
        df["bin_index"] = [0, 1, 3, 4, 5]

        out = label_contiguous_regimes(df)

        assert list(out["contiguous_run_id"]) == [0, 0, 1, 1, 1]
        # The first fix after the gap has no incoming step and cannot inherit
        # a regime or transition from the earlier run.
        assert out.loc[2, "regime"] == "unknown"

    def test_sorts_on_bin_index(self):
        df = _segment("a", [0.01, 0.20, 0.20])
        df["bin_index"] = [10, 11, 12, 13]
        shuffled = df.sample(frac=1.0, random_state=2)

        out = label_contiguous_regimes(shuffled)

        assert list(out["bin_index"]) == [10, 11, 12, 13]


# One track with a long straight-north migratory bout (4 x 10 km = 40 km net) and a short local
# bout (2 x 0.20 km = 0.40 km net), so bout geometry and the 37 km split are analytically known.
#   idx:  0       1        2        3      4      5      6      7       8      9      10
#   step: NaN     0.01     0.01     10     10     10     10     0.01    0.20   0.20   0.01
_MIG_STEPS = [0.01, 0.01, 10.0, 10.0, 10.0, 10.0, 0.01, 0.20, 0.20, 0.01]


def _mig_labelled():
    return label_regimes(_segment("a", _MIG_STEPS))


class TestMovingBouts:
    def test_two_bouts_with_expected_geometry(self):
        b = moving_bouts(_mig_labelled()).sort_values("first_idx").reset_index(drop=True)
        assert len(b) == 2
        # migratory bout: fixes 3-6, anchored at 2; net ~40 km, perfectly straight
        assert b.loc[0, "n_steps"] == 4
        assert (int(b.loc[0, "first_idx"]), int(b.loc[0, "last_idx"])) == (3, 6)
        assert b.loc[0, "net_km"] == pytest.approx(40.0, rel=1e-4)
        assert b.loc[0, "path_km"] == pytest.approx(40.0, rel=1e-4)
        assert b.loc[0, "straightness"] == pytest.approx(1.0, rel=1e-4)
        # local bout: fixes 8-9; net ~0.40 km
        assert b.loc[1, "n_steps"] == 2
        assert (int(b.loc[1, "first_idx"]), int(b.loc[1, "last_idx"])) == (8, 9)
        assert b.loc[1, "net_km"] == pytest.approx(0.40, rel=1e-3)

    def test_segment_start_moving_run_forms_a_bout(self):
        # A segment that opens straight into movement (opener preceded by 'unknown', not 'foraging')
        # still forms a bout; its first moving fix is index 1.
        b = moving_bouts(label_regimes(_segment("a", [10.0, 10.0])))
        assert len(b) == 1
        assert int(b.iloc[0]["first_idx"]) == 1
        assert b.iloc[0]["net_km"] == pytest.approx(20.0, rel=1e-4)


class TestLabelMigratoryLaunches:
    def test_splits_launch_by_bout_scale(self):
        out = label_migratory_launches(_mig_labelled())
        assert list(out["state"]) == [
            "unknown", "foraging", "foraging", "migratory_launch", "moving", "moving",
            "moving", "settle", "local_launch", "moving", "settle",
        ]

    def test_bout_net_km_on_moving_fixes_only(self):
        out = label_migratory_launches(_mig_labelled())
        bn = out["bout_net_km"].to_numpy()
        moving = (out["base_regime"] == "moving").to_numpy()
        assert np.isnan(bn[~moving]).all()
        assert np.isfinite(bn[moving]).all()
        assert bn[3] == pytest.approx(40.0, rel=1e-4)   # migratory bout net on its launch fix
        assert bn[8] == pytest.approx(0.40, rel=1e-3)   # local bout net on its launch fix

    def test_threshold_is_configurable(self):
        # Dropping the cut below the local bout's 0.40 km net promotes it to migratory.
        out = label_migratory_launches(_mig_labelled(), migratory_net_km=0.30)
        assert out.loc[8, "state"] == "migratory_launch"

    def test_only_launches_are_relabelled(self):
        base = _mig_labelled()
        out = label_migratory_launches(base)
        non_launch = (base["regime"] != "launch").to_numpy()
        assert (out["state"].to_numpy()[non_launch] == base["regime"].to_numpy()[non_launch]).all()

    def test_constants(self):
        assert MIGRATORY_NET_KM == 37.0
        assert set(REFINED_STATES) == {
            "foraging", "moving", "local_launch", "migratory_launch", "settle",
        }
