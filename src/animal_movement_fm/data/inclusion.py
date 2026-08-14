"""
Taxon corrections and exclusions for selected species pools based off the audit 
"""

from __future__ import annotations

import pandas as pd


STUDY_TAXON_OVERRIDES: dict[int, dict[str, str]] = {
    1577975863: {"Canis lupus": "Canis dingo"},
    1445058753: {"Larus fuscus": "Larus heuglini"},
    1215840521: {"Larus fuscus": "Larus heuglini"},
}


# study_id -> (study name, reason, affected taxa);  None = whole study
EXCLUDED_STUDIES: dict[int, tuple[str, str, tuple[str, ...] | None]] = {
    2302317848: (
        "Romania livestock guarding dogs",
        "domestic dogs and sheep",
        ("Canis lupus", "Ovis aries"),
    ),
    5328251712: (
        "Mudumalai free-ranging dogs",
        "domestic dogs labelled Canis lupus",
        None,
    ),
    1879445455: (
        "Poultry network Thailand 2022",
        "village dogs and handler",
        ("Canis lupus", "Homo sapiens"),
    ),
    1942479572: (
        "Kruger Hounds",
        "hunting hounds and handler",
        ("Canis lupus", "Homo sapiens"),
    ),
    176207818: (
        "Stray Dogs Prishtina",
        "urban stray dogs",
        None,
    ),
    2988333: (
        "Lesser black-backed gull navigation experiment",
        "translocated homing birds",
        None,
    ),
}


EXCLUDED_DEPLOYMENTS: dict[str, str] = {
    "2009019": "NES Deployment_Manipulation? = Y (animal M780)",
    "2009020": "NES Deployment_Manipulation? = Y (animal X871)",
    "2009021": "NES Deployment_Manipulation? = Y (animal 2370)",
    "2009023": "NES Deployment_Manipulation? = Y (animal 1234)",
}

# add docstrings later;
def is_excluded_deployment(deployment_id: object) -> bool:
    if deployment_id is None:
        return False

    x = str(deployment_id).strip()
    if x in EXCLUDED_DEPLOYMENTS:
        return True

    # IDs sometimes arrive as floats e.g 2009019.0
    try:
        return str(int(float(x))) in EXCLUDED_DEPLOYMENTS
    except (TypeError, ValueError):
        return False


def resolve_taxon(study_id: object, taxon: object) -> object:
    try:
        sid = int(study_id)
    except (TypeError, ValueError):
        return taxon

    return STUDY_TAXON_OVERRIDES.get(sid, {}).get(taxon, taxon)


def is_excluded(study_id: object, taxon: object = None) -> bool:
    try:
        entry = EXCLUDED_STUDIES.get(int(study_id))
    except (TypeError, ValueError):
        return False

    if entry is None:
        return False

    _, _, taxa = entry
    return True if taxa is None else taxon in taxa


def apply_inclusion_policy(
    df: pd.DataFrame,
    *,
    study_col: str = "study_id",
    taxon_col: str = "taxon_canonical_name",
) -> pd.DataFrame:
    """Apply taxon fixes then exclusions."""

    # failsafe missing cols
    if study_col not in df.columns:
        raise KeyError(f"{study_col!r} not in frame; inclusion policy is keyed on study")
    if taxon_col not in df.columns:
        raise KeyError(f"{taxon_col!r} not in frame; nothing to resolve")

    out = df.copy()

    # keep this vectorised: these tables can contain tens of millions of rows
    study = pd.to_numeric(out[study_col], errors="coerce")

    for sid, mapping in STUDY_TAXON_OVERRIDES.items():
        mask = study.eq(sid)
        if not mask.any():
            continue

        for old, new in mapping.items():
            out.loc[mask & out[taxon_col].eq(old), taxon_col] = new

    drop = pd.Series(False, index=out.index)
    for sid, (_, _, taxa) in EXCLUDED_STUDIES.items():
        mask = study.eq(sid)
        if not mask.any():
            continue

        drop |= mask if taxa is None else mask & out[taxon_col].isin(taxa)

    return out.loc[~drop].reset_index(drop=True)


def policy_report(
    df: pd.DataFrame,
    *,
    study_col: str = "study_id",
    taxon_col: str = "taxon_canonical_name",
) -> pd.DataFrame:
    """Build an audit table for affected studies"""
    rows = []

    for sid, g in df.groupby(study_col, dropna=False):
        declared = sorted({str(x) for x in g[taxon_col].dropna().unique()})
        hit = [x for x in declared if is_excluded(sid, resolve_taxon(sid, x))]

        if hit:
            name, reason, _ = EXCLUDED_STUDIES[int(sid)]
            n = int(
                sum(
                    is_excluded(s, resolve_taxon(s, t))
                    for s, t in zip(g[study_col], g[taxon_col])
                )
            )
            rows.append(
                {
                    "study_id": sid,
                    "study": name,
                    "declared": ", ".join(hit),
                    "action": "excluded",
                    "detail": reason,
                    "rows": n,
                }
            )
            continue

        try:
            fixes = STUDY_TAXON_OVERRIDES.get(int(sid), {})
        except (TypeError, ValueError):
            fixes = {}

        for old, new in fixes.items():
            n = int((g[taxon_col] == old).sum())
            if n:
                rows.append(
                    {
                        "study_id": sid,
                        "study": "",
                        "declared": old,
                        "action": "retagged",
                        "detail": f"{old} -> {new}",
                        "rows": n,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=["study_id", "study", "declared", "action", "detail", "rows"],
    )
