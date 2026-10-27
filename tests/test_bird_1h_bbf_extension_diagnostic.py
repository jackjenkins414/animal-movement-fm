# GenAI reference: I asked for tests of the BbF extension's validation diagnostic (its runs and twins, the switch and
# fold guards, an end to end fold and report on toy tracks, and bullets the seal's placeholder rule accepts); GenAI
# wrote them after tests/test_transformer_collapse.py ones
from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.diagnostics.bird_1h_bbf_extension_diagnostic as X
import experiments.reporting.bird_rotation_sealed_eval as RS
from animal_movement_fm.data.multispecies import HorizonWindows
from animal_movement_fm.models.fm import MovementFM

F, TC, b1h = X.F, X.TC, X.b1h  # the modules the diagnostic imported, so monkeypatching reaches them
HOUR = 3_600
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}


def _track(tmp_path, name, bins, *, step_deg=0.01, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"segment_id": f"{name}_s0", "individual_id": name, "bin_index": int(b),
             "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(seconds=HOUR * int(b)
                                                                               + int(rng.integers(0, 3000))),
             "lon": 10.0 + step_deg * b + rng.normal(0, noise), "lat": 50.0 + rng.normal(0, noise)} for b in bins]
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _windows(path, name, context=12):
    return HorizonWindows.build(path, [name], context=context, core=12, horizons=(1, 6, 24), bin_seconds=HOUR,
                                clip=1000.0)


def _toy(trunk, context, seed=0):
    torch.manual_seed(seed)
    flag = trunk == "lstm" and context > 12
    return MovementFM(input_dim=12 + flag, d_model=16, nhead=4, num_layers=2, dropout=0.1, trunk=trunk,
                      n_horizons=3, observed_flag=flag, use_fm_package=False).eval()


# ---------------------------------------------------------------------------------------------------- the runs
@pytest.mark.parametrize("fold", range(10))
def test_each_fold_has_its_nine_extension_runs_and_their_two_ablation_twins(fold):
    entries = X.entries(fold)
    assert len(entries) == 27 and len({e["label"] for e in entries}) == 27
    want = {(arm, s, c) for arm in X.ARMS for s in (0, 1, 2) for c in (24, 48, 72)}
    assert {X.parse(e["label"]) for e in entries} == want
    for e in entries:
        arm, seed, context = X.parse(e["label"])
        assert e["run_id"] == f"{arm}_f{fold}_s{seed}_C{context}" and e["context"] == context
        assert e["transformer"] == (arm != "LSTMBb") and e["pair"] == f"{arm}_s{seed}"
    assert {e["run_id"] for e in entries if e["label"].startswith("BbF")} == {
        r.run_id for r in F.EXTENSION if r.fold == fold}
    assert {e["run_id"] for e in entries if not e["label"].startswith("BbF")} <= {r.run_id for r in b1h.PLAN}


def test_only_the_ten_folds_are_accepted_and_own_entries_need_a_mode():
    with pytest.raises(SystemExit, match="fold 10 is not one of"):
        X.run_fold(10)
    with pytest.raises(SystemExit, match="need a mode"):
        TC.run_fold(0, entries=X.entries(0))
    with pytest.raises(SystemExit, match="need a mode"):
        TC.run_fold(0, entries=X.entries(0), mode="x", repair=True)


def test_the_load_checks_the_switches_on_the_record_and_on_the_model(monkeypatch):
    calls = []

    def fake(switches_recorded, switches_built):
        def load(run, *args):
            calls.append((run.run_id, args))
            return SimpleNamespace(**switches_built), np.ones(3), {"config": switches_recorded}
        return load

    bbf = F.EXTENSION[0]
    monkeypatch.setattr(TC, "load_run", fake(V4, V4))
    X.checked_load("BbF", bbf)
    assert calls[-1] == (bbf.run_id, (F.RESULTS, F.TAG))  # BbF from its own results, with its tag checked
    monkeypatch.setattr(TC, "load_run", fake({}, X.DEFAULTS))  # a 1 h record from before the switches existed
    X.checked_load("Bb", bbf.base)
    assert calls[-1] == (bbf.base.run_id, ())
    for arm, recorded, built in (("BbF", {}, X.DEFAULTS), ("BbF", V4, X.DEFAULTS), ("Bb", V4, V4),
                                 ("LSTMBb", {}, V4 | {"norm_first": False})):
        monkeypatch.setattr(TC, "load_run", fake(recorded, built))
        with pytest.raises(SystemExit, match="switches"):
            X.checked_load(arm, X.twin_runs(bbf)[arm])


def test_record_status_counts_what_is_on_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(F, "RESULTS", tmp_path / "bbf")
    monkeypatch.setattr(b1h, "RESULTS", tmp_path / "1h")
    F.RESULTS.mkdir()
    b1h.RESULTS.mkdir()
    (F.RESULTS / "BbF_f2_s1_C48.json").write_text("{}")
    (b1h.RESULTS / "LSTMBb_f2_s0_C24.json").write_text("{}")
    assert X.record_status(2) == {"BbF": {"planned": 9, "present": 1}, "Bb": {"planned": 9, "present": 0},
                                  "LSTMBb": {"planned": 9, "present": 1}}


# ---------------------------------------------------------------------------------------------------- end to end
def test_a_fold_and_the_report_run_end_to_end_on_toy_data(tmp_path, monkeypatch, capsys):
    bins = np.concatenate(([0, 3, 9], np.arange(20, 300)))
    paths = {n: _track(tmp_path, n, bins, step_deg=0.002 * (i + 1), noise=2e-4, seed=i) for i, n in enumerate("abc")}

    def build(stem, partition, payload, context=12):
        assert partition in ("train", "val")
        return _windows(paths[stem], stem, context)

    models = {(arm, c): _toy("lstm" if arm == "LSTMBb" else "transformer", c, seed=c) for arm in X.ARMS
              for c in X.CONTEXTS}
    record = {"loss_history": np.linspace(1.0, 0.5, 40).tolist(), "head_loss_history": [[1.0, 0.5, 0.2]] * 40}
    monkeypatch.setattr(b1h, "load_registered", lambda: {})
    monkeypatch.setattr(b1h, "fold_probabilities", lambda fold, payload: {"a": 0.6, "b": 0.4})
    monkeypatch.setattr(b1h, "assert_fold_sources", lambda fold, stems: None)
    monkeypatch.setattr(b1h, "check_window_counts", lambda pool: None)
    monkeypatch.setattr(TC, "FOLDS", {0: ("c",)})
    monkeypatch.setattr(TC, "BOUNDARY_M", defaultdict(lambda: 30.0))
    monkeypatch.setattr(TC, "build", build)
    monkeypatch.setattr(X, "checked_load", lambda arm, run: (models[(arm, run.context)], np.array([0.7, 6.0, 5.0]),
                                                             record))
    for name in ("OUT", "REPORT_JSON", "REPORT_MD"):
        monkeypatch.setattr(X, name, tmp_path / "out" / getattr(X, name).name if name != "OUT" else tmp_path / "out")
    monkeypatch.setattr(X, "FOLDS", (0,))

    result = X.run_fold(0, n_train=300, max_val=150, n_check=60)
    assert result["mode"] == X.MODE and result["seed"] is None and result["opened_test_partition"] is False
    assert result["contexts"] == [24, 48, 72] and set(result["checks"]) == {e["label"] for e in X.entries(0)}
    assert result["checks"]["BbF_s0_C72"]["masked_noise"]["max_abs_diff"] < 1e-5
    assert set(result["core_only"]) == set(result["checks"])

    rows = X.run_rows(result)
    assert len(rows) == 27 and all(r["train_loss_over_zero"] > 0 for r in rows)
    row = next(r for r in rows if r["label"] == "Bb_s1_C48")
    sens = result["checks"]["Bb_s1_C48"]["token_sensitivity"]
    assert row["swap_over_output_h6"] == pytest.approx(sens["last_swapped"]["h6"] / sens["median_abs_output"]["h6"])

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "fold0.json").write_text(json.dumps(result, default=str))
    X.report()
    markdown = X.REPORT_MD.read_text()
    for heading in ("## 1. Loss", "## 2. Input sensitivity", "## 3. Validation behaviour", "## 4. Bug checks",
                    "## 5. BbF training loss", "## 6. For the freeze"):
        assert heading in markdown
    assert "| BbF_C24 | 3 |" in markdown and "| LSTMBb_C72 |" in markdown
    bullets = capsys.readouterr().out.split("\n\n", 1)[1]
    assert bullets.count("\n- **") == 4 and "of 30 each" in bullets
    assert not RS.PLACEHOLDER.search(bullets)  # the bullets can go into the freeze as printed


def test_the_report_refuses_a_dry_run_or_foreign_fold_file(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "OUT", tmp_path)
    monkeypatch.setattr(X, "FOLDS", (3,))
    with pytest.raises(SystemExit, match="missing fold3.json"):
        X.report()
    good = {"fold": 3, "mode": X.MODE, "labels": [e["label"] for e in X.entries(3)], "opened_test_partition": False}
    for bad in (good | {"mode": "diagnosis"}, good | {"labels": good["labels"][:-1]}, good | {"dry_run": True}):
        (tmp_path / "fold3.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="fold3.json"):
            X.report()
