# GenAI reference: added with GenAI since I found the invalid probe exposed that the experiment driver had inherited TrainConfig's 
# baseline safe reconstruction default instead of enabling the registered full FM objective.

import json
from dataclasses import asdict

import pytest

import experiments.training.tier2_inner_loop as inner_loop
from experiments.training.tier2_inner_loop import (
    GRID,
    REGISTERED_PROBE_BUDGETS,
    build_inner_loop_config,
    checkpoint_path,
    probe_budgets,
    rank_probe_budgets,
    validate_probe_request,
)


def test_every_inner_loop_cell_builds_the_complete_registered_fm_package():
    for grid_index, (d_model, layers, lr) in enumerate(GRID):
        config = build_inner_loop_config(grid_index, steps=40_000, seed=0)

        assert config.use_fm_package is True
        assert config.use_reconstruction_head is True
        assert (config.d_model, config.num_layers, config.lr) == (d_model, layers, lr)
        model = config.build()
        assert model.recon_head is not None
        assert model.mask_token is not None


def test_probe_uses_distinct_independent_complete_budgets():
    budgets = probe_budgets(40_000)

    assert budgets == [5_000, 10_000, 20_000, 30_000, 40_000]
    configs = [build_inner_loop_config(6, steps=budget, seed=0) for budget in budgets]
    assert [config.steps for config in configs] == budgets


def test_probe_request_is_locked_to_registered_cell_seed_and_maximum():
    validate_probe_request(grid_index=6, steps=40_000, seed=0)

    for request in (
        {"grid_index": 5, "steps": 40_000, "seed": 0},
        {"grid_index": 6, "steps": 30_000, "seed": 0},
        {"grid_index": 6, "steps": 40_000, "seed": 1},
    ):
        with pytest.raises(SystemExit, match="unregistered step-budget probe"):
            validate_probe_request(**request)

    with pytest.raises(ValueError, match="registered probe maximum"):
        probe_budgets(30_000)


def test_corrected_probe_and_grid_checkpoints_cannot_collide():
    probe = checkpoint_path("fold0_g06", 40_000, probe=True)
    grid = checkpoint_path("fold0_g06", 40_000, probe=False)

    assert probe != grid
    assert "registered_fullfm_probe" in probe.name
    assert "registered_fullfm_grid" in grid.name


def _skill_table(default: list[float]) -> dict[int, list[float]]:
    return {budget: list(default) for budget in REGISTERED_PROBE_BUDGETS}


def test_probe_budget_rule_first_uses_highest_mean_then_highest_minimum():
    skills = _skill_table([-2.0, -2.0, -2.0])
    skills[5_000] = [0.0, 1.0, 2.0]  # mean 1.0, minimum 0.0
    skills[10_000] = [0.5, 0.5, 2.0]  # mean 1.0, minimum 0.5
    skills[20_000] = [0.9, 1.0, 1.2]  # mean 1.033..., so primary criterion wins

    ranked = rank_probe_budgets(skills)

    assert [candidate["steps"] for candidate in ranked[:3]] == [20_000, 10_000, 5_000]


def test_probe_budget_rule_uses_fewer_steps_only_after_exact_metric_ties():
    skills = _skill_table([-2.0, -2.0, -2.0])
    skills[5_000] = [0.0, 1.0, 2.0]
    skills[10_000] = [0.0, 0.5, 2.5]

    ranked = rank_probe_budgets(skills)

    assert ranked[0]["mean_skill"] == ranked[1]["mean_skill"] == 1.0
    assert ranked[0]["min_skill"] == ranked[1]["min_skill"] == 0.0
    assert [candidate["steps"] for candidate in ranked[:2]] == [5_000, 10_000]


def test_probe_report_validates_three_full_fm_folds_and_writes_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(inner_loop, "RESULTS", tmp_path)
    base_skills = {5_000: 0.1, 10_000: 0.2, 20_000: 0.5, 30_000: 0.4, 40_000: 0.3}

    for fold_index, source in enumerate(inner_loop.SOURCES):
        identifier = inner_loop.run_id(6, fold_index, 0)
        payload = {
            "run_id": identifier,
            "grid_index": 6,
            "fold_index": fold_index,
            "fold": source[0],
            "config": asdict(build_inner_loop_config(6, steps=40_000, seed=0)),
            "scores": [
                {"steps": budget, "skill": base_skills[budget] + fold_index * 0.01}
                for budget in REGISTERED_PROBE_BUDGETS
            ],
            "probe": True,
            "probe_design": inner_loop.PROBE_DESIGN,
            "probe_budgets": list(REGISTERED_PROBE_BUDGETS),
        }
        path = tmp_path / f"{identifier}_registered_fullfm_probe.json"
        path.write_text(json.dumps(payload))

    inner_loop.report_probe_selection()

    decision = json.loads((tmp_path / "tier2_probe_selection.json").read_text())
    assert decision["selected_steps"] == 20_000
    assert decision["rule"] == [
        "highest_mean_skill",
        "highest_minimum_skill",
        "fewer_steps",
    ]
    assert len(decision["source_files"]) == 3


# GenAI ref: I gave it this test file to expand on; the registered-artefact tests below were written by GenAI during the A16 follow up
# audit, after a doubled path (registered/registered/...) had silently switched the A16 freeze off with every test green

import pandas as pd

import experiments.training.tier2_final_train as final_train
from animal_movement_fm.data.split import ROLE_STEMS
from experiments.data import regime_profile as regime_profile_script
from experiments.reporting import tier2_prediction2

SEALED_BOUNDARIES_M = {
    "larus_fuscus": 411.451080530957,
    "canis_lupus": 131.3024030590991,
    "mirounga_angustirostris": 1609.4390043586309,
}


def test_every_registered_path_the_code_reads_exists():
    paths = (
        inner_loop.REGISTERED_DECISION,
        inner_loop.REGISTERED_WEIGHTS,
        inner_loop.REGISTERED_SELECTION,
        regime_profile_script.FROZEN_DECISION,
        tier2_prediction2.FROZEN_STANDARDISATION,
    )
    for path in paths:
        assert path.parent == inner_loop.REGISTERED, path
        assert path.exists(), path


def test_registered_weights_reproduce_the_a9_percentages():
    inner_loop.check_registered_weights(json.loads(inner_loop.REGISTERED_WEIGHTS.read_text()))


def test_the_weight_check_fires_on_a_small_drift():
    payload = json.loads(inner_loop.REGISTERED_WEIGHTS.read_text())
    payload["source_pool"]["ciconia_ciconia"] += 1e-4  # one hundredth of a percentage point
    with pytest.raises(SystemExit, match="A9 registers"):
        inner_loop.check_registered_weights(payload)


def test_registered_selection_names_grid_11():
    decision = json.loads(inner_loop.REGISTERED_SELECTION.read_text())
    assert decision["selected_grid_index"] == final_train.REGISTERED_GRID_INDEX == 11
    assert final_train.selected_grid_index(None) == (11, "tier2_selection.json")


def test_registered_boundaries_hold_the_frozen_rule_and_the_sealed_values():
    decision = json.loads(inner_loop.REGISTERED_DECISION.read_text())
    assert (decision["anchor_quantile"], decision["multiplier"]) == (50, 0.1)
    frozen = decision["tier2_boundaries_m"]
    assert {key: frozen[key] for key in SEALED_BOUNDARIES_M} == SEALED_BOUNDARIES_M
    assert set(frozen) == {source[2] for source in inner_loop.SOURCES} | set(SEALED_BOUNDARIES_M)


def _profiles(tmp_path, rows):
    path = tmp_path / "profiles.csv"
    pd.DataFrame(rows, columns=["species", "selected_boundary_m"]).to_csv(path, index=False)
    return path


def test_boundary_m_returns_registered_values_for_roles_and_the_csv_for_candidates(tmp_path):
    profiles = _profiles(tmp_path, [("larus_fuscus", SEALED_BOUNDARIES_M["larus_fuscus"]), ("new_bird", 12.5)])
    assert inner_loop.boundary_m("larus_fuscus", profiles=profiles) == SEALED_BOUNDARIES_M["larus_fuscus"]
    assert inner_loop.boundary_m("new_bird", profiles=profiles) == 12.5


def test_boundary_m_fails_when_the_profile_table_has_moved(tmp_path):
    # the value the 15-09 re-selection wrote for L. fuscus
    profiles = _profiles(tmp_path, [("larus_fuscus", 181.88180190237625)])
    with pytest.raises(SystemExit, match="registered value governs"):
        inner_loop.boundary_m("larus_fuscus", profiles=profiles)


def test_final_train_uses_the_shared_boundary_function():
    assert final_train.boundary_m.__module__ == "tier2_inner_loop"
    assert final_train.boundary_m.__name__ == "boundary_m"


def test_role_stems_cover_every_tier2_role_file():
    stems = {s[0] for s in inner_loop.SOURCES} | {s[1] for s in inner_loop.SOURCES}
    stems |= {h[0] for h in final_train.HELD_OUT} | {h[1] for h in final_train.HELD_OUT}
    assert stems <= ROLE_STEMS
