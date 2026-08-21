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
