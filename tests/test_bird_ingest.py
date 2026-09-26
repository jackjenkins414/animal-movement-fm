# GenAI reference: GenAI wrote these tests from the guards I approved for the bird only study's ingest; that being every
# species on an extract list has a speed ceiling, the extract only ever adds to species_fixes_summary.json and never
# replaces an existing extract, and regime_profile.py --out-csv writes its own table and refuses the A4 decision flags
import json
import sys

import numpy as np
import pandas as pd
import pytest

import experiments.data.extract_species_fixes as X
import experiments.data.regime_profile as RP
from animal_movement_fm.analysis.regime_profile import M_PER_DEG_LAT
from animal_movement_fm.data.qc import SPEED_CEILINGS, WIND_ASSISTED_FLIER

# the bird study's 14 first-time extracts, passed to the extract with --species
BIRD_STUDY_NEW = (
    "Larus argentatus",
    "Anas platyrhynchos",
    "Branta leucopsis",
    "Streptopelia turtur",
    "Hydroprogne caspia",
    "Anser fabalis",
    "Anser anser",
    "Pelecanus occidentalis",
    "Falco tinnunculus",
    "Phoenicopterus roseus",
    "Platalea leucorodia",
    "Morus bassanus",
    "Ardea alba",
    "Anser brachyrhynchus",
)


# --------------------------------------------------------------------------- speed ceilings

def test_every_species_on_an_extract_list_has_a_speed_ceiling():
    """Without a ceiling the extract silently skips the speed flags, so a missing entry is a wrong extract."""
    listed = {
        *X.TIER2_AND_CANDIDATES,
        *X.TIER3_BIRDS,
        *X.TIER3_CANDIDATES,
        *X.NON_MOVEBANK_SOURCES,
        *BIRD_STUDY_NEW,
    }
    assert sorted(listed - set(SPEED_CEILINGS)) == []


def test_the_bird_study_species_take_the_flier_bound_unchanged():
    for name in BIRD_STUDY_NEW:
        ceiling = SPEED_CEILINGS[name]
        assert (ceiling.instant_kmh, ceiling.sustained_kmh) == (
            WIND_ASSISTED_FLIER.instant_kmh,
            WIND_ASSISTED_FLIER.sustained_kmh,
        ) == (220.0, 180.0), name


# --------------------------------------------------------------------------- extract guards

def _extract(monkeypatch, out_dir, tmp_path, *species):
    argv = ["extract_species_fixes.py", "--out-dir", str(out_dir), "--source", str(tmp_path / "unused.parquet")]
    argv += ["--species", *species]
    monkeypatch.setattr(sys, "argv", argv)
    X.main()


def _stub_extract(monkeypatch, calls):
    def fake(con, name, paths, out_path, **kwargs):
        calls.append(name)
        return True, {"species": name, "fixes_raw": 10, "fixes_out": 9, "individuals_out": 1,
                      "fixes_dropped_by_qc": 1}
    monkeypatch.setattr(X, "extract", fake)


def test_a_species_with_a_summary_entry_is_refused(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "species_fixes_summary.json").write_text(json.dumps([{"species": "Anser fabalis"}]))
    calls = []
    _stub_extract(monkeypatch, calls)

    with pytest.raises(SystemExit, match="already have a summary entry or a fixes file"):
        _extract(monkeypatch, out, tmp_path, "Anser fabalis")
    assert calls == []


def test_a_species_with_a_fixes_file_is_refused_even_without_an_entry(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "anser_fabalis_fixes.parquet").write_bytes(b"existing")
    calls = []
    _stub_extract(monkeypatch, calls)

    with pytest.raises(SystemExit, match="Anser fabalis"):
        _extract(monkeypatch, out, tmp_path, "Ardea alba", "Anser fabalis")
    assert calls == []  # refused before any species is extracted, not part way through
    assert (out / "anser_fabalis_fixes.parquet").read_bytes() == b"existing"


def test_an_unreadable_summary_stops_the_run_and_is_left_alone(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    summary = out / "species_fixes_summary.json"
    summary.write_text("{not json")
    calls = []
    _stub_extract(monkeypatch, calls)

    with pytest.raises(SystemExit, match="could not parse"):
        _extract(monkeypatch, out, tmp_path, "Ardea alba")
    assert calls == []
    assert summary.read_text() == "{not json"


def test_a_new_species_is_added_and_existing_entries_are_untouched(monkeypatch, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    before = [
        {"species": "Anser albifrons", "fixes_out": 2350565, "notes": ["kept"]},
        {"species": "Larus fuscus", "fixes_out": 27582509, "ceiling_multiplier": 1.0},
    ]
    (out / "species_fixes_summary.json").write_text(json.dumps(before, indent=2))
    calls = []
    _stub_extract(monkeypatch, calls)

    _extract(monkeypatch, out, tmp_path, "Ardea alba")

    after = {s["species"]: s for s in json.loads((out / "species_fixes_summary.json").read_text())}
    assert calls == ["Ardea alba"]
    assert set(after) == {"Anser albifrons", "Larus fuscus", "Ardea alba"}
    for entry in before:
        assert after[entry["species"]] == entry


def test_the_role_species_refusal_still_comes_first(monkeypatch, tmp_path):
    calls = []
    _stub_extract(monkeypatch, calls)
    with pytest.raises(SystemExit, match="hold Tier 2 roles"):
        _extract(monkeypatch, tmp_path / "out", tmp_path, "Larus fuscus")
    assert calls == []


# --------------------------------------------------------------------------- regime_profile --out-csv

def _segments(path):
    """A north-only single-segment track, as in test_regime_profile.py."""
    rng = np.random.default_rng(0)
    lats = np.cumsum(np.concatenate([[0.0], (np.abs(rng.normal(500.0, 200.0, 400)) + 1.0) / M_PER_DEG_LAT]))
    offsets = np.arange(len(lats))
    pd.DataFrame({
        "segment_id": "s1",
        "individual_id": 1,
        "bin_index": 1_000_000 + offsets,
        "timestamp": pd.Timestamp("2020-01-01 03:00:00") + offsets * pd.Timedelta(hours=6),
        "lat": lats,
        "lon": 0.0,
    }).to_parquet(path, index=False)


def _profile(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["regime_profile.py", *argv])
    RP.main()


def test_out_csv_writes_its_own_table_and_leaves_the_shared_one_alone(monkeypatch, tmp_path):
    shared = tmp_path / "regime_profiles.csv"
    shared.write_text("species,domain\nwhite_stork,flying\n")
    monkeypatch.setattr(RP, "OUT_CSV", shared)
    segments = tmp_path / "bird_segments.parquet"
    _segments(segments)
    birds = tmp_path / "birds" / "regime_profiles.csv"

    _profile(monkeypatch, "--segments", str(segments), "--name", "ardea_alba", "--domain", "flying",
             "--out-csv", str(birds))

    assert shared.read_text() == "species,domain\nwhite_stork,flying\n"
    table = pd.read_csv(birds)
    assert table["species"].tolist() == ["ardea_alba"]


@pytest.mark.parametrize("flag", ["--decide-threshold", "--reselect"])
def test_out_csv_refuses_the_decision_flags(monkeypatch, tmp_path, flag):
    monkeypatch.setattr(RP, "OUT_CSV", tmp_path / "regime_profiles.csv")
    # the segments path does not exist, so reaching the refusal proves it fires before anything is read
    with pytest.raises(SystemExit, match="run only on"):
        _profile(monkeypatch, "--segments", str(tmp_path / "missing.parquet"), "--name", "x",
                 "--out-csv", str(tmp_path / "birds" / "regime_profiles.csv"), flag)
    assert not (tmp_path / "birds").exists()


def test_the_decision_flags_are_not_refused_on_the_shared_table(monkeypatch, tmp_path):
    monkeypatch.setattr(RP, "OUT_CSV", tmp_path / "regime_profiles.csv")
    with pytest.raises(SystemExit, match="Segments file not found"):
        _profile(monkeypatch, "--segments", str(tmp_path / "missing.parquet"), "--name", "x",
                 "--decide-threshold")
