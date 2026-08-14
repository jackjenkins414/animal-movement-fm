# GenAI reference: Used GenAI to write the following tests from behaviour I outlined; such as ensuring the retag only touches
# the label and overides names and leaves other species in the same taxon alone. Also ensured that an exclusion to a taxon
# spares wild species within that same multi species stidy while an enscoped one removes everything. Study ids and expected 
# taxa are ones Ive chosen from Movebank metadata myself; each of which I have validated the correctness of. 

import pandas as pd
import pytest

from animal_movement_fm.data.inclusion import (
    EXCLUDED_STUDIES,
    STUDY_TAXON_OVERRIDES,
    apply_inclusion_policy,
    is_excluded,
    policy_report,
    resolve_taxon,
)

DINGO_STUDY = 1577975863
ROMANIA_DOGS = 2302317848
HEBBLEWHITE = 209824313  # Genuine Alberta/BC wolves must survive untouched. 
HEUGLINI = 1445058753  # Wrong species label -> retag.
NAV_EXPERIMENT = 2988333  # Translocated gulls -> exclude. 


def _frame():
    """A miniature Canis pool: real wolves, retagged dingoes, excluded dogs, an unrelated species."""
    return pd.DataFrame(
        {
            "individual_id": [1, 2, 3, 4, 5, 6],
            "study_id": [
                HEBBLEWHITE,
                HEBBLEWHITE,
                DINGO_STUDY,
                DINGO_STUDY,
                ROMANIA_DOGS,
                DINGO_STUDY,
            ],
            "taxon_canonical_name": [
                "Canis lupus",
                "Canis lupus",
                "Canis lupus",  # Actually a dingo
                "Canis lupus",  # Actually a dingo
                "Canis lupus",  # Actually a livestock guardian dog
                "Felis catus",  # Same study, different species, must not be retagged
            ],
        }
    )


def test_overrides_and_exclusions_are_disjoint():
    """A study is either relabelled or removed, never both; otherwise the order of operations would matter."""
    assert not set(STUDY_TAXON_OVERRIDES) & set(EXCLUDED_STUDIES)


def test_resolve_taxon_only_touches_the_declared_label():
    assert resolve_taxon(DINGO_STUDY, "Canis lupus") == "Canis dingo"
    # Same study, a species the override does not name.
    assert resolve_taxon(DINGO_STUDY, "Felis catus") == "Felis catus"
    # Same label, a study with no override.
    assert resolve_taxon(HEBBLEWHITE, "Canis lupus") == "Canis lupus"


def test_resolve_taxon_survives_unusable_study_ids():
    """RAATD ids are strings and some corpora carry nulls; neither should raise."""
    assert resolve_taxon(None, "Canis lupus") == "Canis lupus"
    assert resolve_taxon("CAR2013_sel_a_m_03", "Canis lupus") == "Canis lupus"


def test_is_excluded_accepts_string_ids():
    assert is_excluded(ROMANIA_DOGS, "Canis lupus")
    assert is_excluded(str(ROMANIA_DOGS), "Canis lupus")
    assert not is_excluded(HEBBLEWHITE, "Canis lupus")
    assert not is_excluded(None, "Canis lupus")


def test_exclusions_are_scoped_to_taxon_not_to_the_whole_study():
    """Movebank studies are multi-species and a reason rarely covers all of them.

    "Poultry network Thailand 2022" declares fifteen taxa; only the dogs are domestic. The
    other thirteen are wild curlews, owls, monitors, turtles, kites and vultures tracked for
    avian-influenza surveillance, and excluding them would delete wild animals on a ground
    that applies only to the dogs.
    """
    thailand = 1879445455
    assert is_excluded(thailand, "Canis lupus")
    assert is_excluded(thailand, "Homo sapiens")
    for wild in ("Numenius arquata", "Tyto alba", "Chelonia mydas", "Milvus migrans"):
        assert not is_excluded(thailand, wild), wild


def test_a_scoped_exclusion_spares_other_taxa_in_the_same_study():
    frame = pd.DataFrame(
        {
            "individual_id": [1, 2],
            "study_id": [1879445455, 1879445455],
            "taxon_canonical_name": ["Canis lupus", "Numenius arquata"],
        }
    )
    out = apply_inclusion_policy(frame)
    assert list(out["taxon_canonical_name"]) == ["Numenius arquata"]


def test_an_unscoped_exclusion_removes_everything_in_its_study():
    """`None` means every taxon, and is used only where the whole study is one kind of animal."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 2],
            "study_id": [176207818, 176207818],
            "taxon_canonical_name": ["Canis lupus", "Anything else"],
        }
    )
    assert apply_inclusion_policy(frame).empty


def test_apply_removes_dogs_retags_dingoes_and_keeps_wolves():
    out = apply_inclusion_policy(_frame())

    # The excluded study is gone entirely.
    assert ROMANIA_DOGS not in set(out["study_id"])

    # Wolves keep their label.
    wolves = out[out["taxon_canonical_name"] == "Canis lupus"]
    assert sorted(wolves["individual_id"]) == [1, 2]

    # Dingoes are retained under the corrected name rather than dropped.
    dingoes = out[out["taxon_canonical_name"] == "Canis dingo"]
    assert sorted(dingoes["individual_id"]) == [3, 4]

    # The cat in the dingo study is untouched.
    assert list(out.loc[out["individual_id"] == 6, "taxon_canonical_name"]) == ["Felis catus"]


def test_apply_does_not_mutate_the_input():
    frame = _frame()
    before = frame.copy()
    apply_inclusion_policy(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_apply_is_idempotent():
    """Running the policy twice must not retag an already-corrected label or drop more rows."""
    once = apply_inclusion_policy(_frame())
    twice = apply_inclusion_policy(once)
    pd.testing.assert_frame_equal(once, twice)


def test_apply_requires_the_keyed_columns():
    with pytest.raises(KeyError):
        apply_inclusion_policy(_frame().drop(columns=["study_id"]))
    with pytest.raises(KeyError):
        apply_inclusion_policy(_frame().drop(columns=["taxon_canonical_name"]))


def test_a_wrong_label_is_retagged_and_a_manipulation_is_excluded():
    """The two mechanisms must not be confused: heuglini is real data under a wrong name,
    the navigation gulls are the right species doing something a human made them do."""
    gulls = pd.DataFrame(
        {
            "individual_id": [10, 11, 12],
            "study_id": [HEUGLINI, NAV_EXPERIMENT, 2988334],
            "taxon_canonical_name": ["Larus fuscus"] * 3,
        }
    )
    out = apply_inclusion_policy(gulls)

    # Retagged, not dropped: still one row, under the correct species.
    assert list(out.loc[out["individual_id"] == 10, "taxon_canonical_name"]) == ["Larus heuglini"]
    # Excluded entirely.
    assert 11 not in set(out["individual_id"])
    # A neighbouring study id is untouched; exclusion is not a prefix or range match.
    assert list(out.loc[out["individual_id"] == 12, "taxon_canonical_name"]) == ["Larus fuscus"]


def test_retagged_species_leave_the_original_pool():
    """The point of a retag is that the pool no longer contains the animal."""
    gulls = pd.DataFrame(
        {
            "individual_id": [10],
            "study_id": [HEUGLINI],
            "taxon_canonical_name": ["Larus fuscus"],
        }
    )
    out = apply_inclusion_policy(gulls)
    assert (out["taxon_canonical_name"] == "Larus fuscus").sum() == 0


def test_policy_report_explains_every_affected_study():
    report = policy_report(_frame())
    actions = dict(zip(report["study_id"], report["action"], strict=True))
    assert actions[ROMANIA_DOGS] == "excluded"
    assert actions[DINGO_STUDY] == "retagged"
    assert HEBBLEWHITE not in actions  # untouched studies do not appear

    # The retag counts the canids only, not the cat sharing the study.
    assert int(report.loc[report["study_id"] == DINGO_STUDY, "rows"].iloc[0]) == 2
