# GenAI reference: I used GenAI to write these unit tests to check the movement state labeller's behaviour, in the same 
# style as the other module tests. Every case has an analytically known expected label (chosen to sit clearly 
# inside/outside the hysteresis band), so we can verify correctness by inspection. I gave GenAI the hysteresis band so it 
# could see how it was defined and turn this into analytical test cases. 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.analysis.regime import (
    REGIMES,
    add_step_length_km,
    label_regimes,
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
