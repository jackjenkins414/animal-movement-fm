"""Run all four baselines on all three synthetic datasets.

Produces:
- results/synthetic_baselines.csv      (long format, one row per model x dataset)
- results/feature_importances.csv      (XGBoost feature importances, top N per axis x dataset)
- results/raw_errors_<dataset>.csv     (per-prediction errors for plotting/re-analysis)
- results/plots/bar_<metric>.png       (one bar chart per metric)
- results/plots/cdf_<dataset>.png      (one error CDF per dataset)
- results/plots/summary.png            (everything on one figure for the supervisor)
- A Markdown table printed to stdout, paste-able into methods.md.

GenAI Reference: We used GenAI to help generate parts of the code in this script. 
Will add the DETAILS later (full in depth prompt/verification of correctness) but 
for now I have this. 
"""

import logging
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from animal_movement_fm.data.synthetic import (
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.models.arima import ARIMA
from animal_movement_fm.models.averaging import DisplacementAveraging
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.xgboost import XGBoost

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_INDIVIDUALS = 20
N_STEPS = 500
TEST_FRAC = 0.2
CONTEXT_LENGTH = 24
SEED = 0
TOP_N_FEATURES = 10  # how many top features to report per axis

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_PATH = RESULTS_DIR / "synthetic_baselines.csv"
IMPORTANCES_PATH = RESULTS_DIR / "feature_importances.csv"

# Consistent colours across plots so the same model is the same colour everywhere.
MODEL_COLOURS = {
    "persistence":   "#999999",
    "averaging_k1":  "#a1d99b",
    "averaging_k10": "#31a354",
    "arima_202":     "#fdae6b",
    "xgboost_lag5":  "#3182bd",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("animal_movement_fm.models.xgboost").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_by_individual(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long-format trajectory DataFrame into train/test by individual."""
    rng = np.random.default_rng(seed)
    individuals = df["individual_id"].unique()
    rng.shuffle(individuals)
    n_test = max(1, int(round(len(individuals) * test_frac)))
    test_ids = set(individuals[:n_test])
    df_test = df[df["individual_id"].isin(test_ids)].copy()
    df_train = df[~df["individual_id"].isin(test_ids)].copy()
    return df_train, df_test


def summarise(results: pd.DataFrame) -> dict[str, float]:
    """Compute headline error statistics from per-prediction results."""
    errors = results["error_m"].to_numpy()
    return {
        "n": len(errors),
        "median_m": float(np.median(errors)),
        "mean_m": float(np.mean(errors)),
        "p99_m": float(np.percentile(errors, 99)),
        "std_m": float(np.std(errors)),
    }


def evaluate_model(
    model,
    df_test: pd.DataFrame,
    context_length: int,
    *,
    needs_fit: bool = False,
    df_train: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Optionally fit, then evaluate. Returns (summary, per-prediction results)."""
    fit_seconds = 0.0
    if needs_fit:
        t0 = perf_counter()
        model.fit(df_train)
        fit_seconds = perf_counter() - t0

    t0 = perf_counter()
    results = evaluate(
        model, df_test, context_length=context_length, show_progress=False
    )
    eval_seconds = perf_counter() - t0

    stats = summarise(results)
    stats["fit_seconds"] = fit_seconds
    stats["eval_seconds"] = eval_seconds
    return stats, results


def make_models() -> list[tuple[str, object, bool]]:
    """Return list of (display_name, model_instance, needs_fit) tuples.

    Fresh instances are needed per dataset because XGBoost retains state.
    """
    return [
        ("persistence", Persistence(), False),
        ("averaging_k1", DisplacementAveraging(k=1), False),
        ("averaging_k10", DisplacementAveraging(k=10), False),
        ("arima_202", ARIMA(order=(2, 0, 2)), False),
        # Time features disabled for synthetic-data experiments: the harness
        # Predictor protocol does not currently propagate timestamps to predict(),
        # so fitting with time features causes a train/predict distribution shift
        # (training sees varying sin/cos hour; predict sees frozen midnight values).
        # Real-data experiments may revisit this once the protocol is extended.
        ("xgboost_lag5", XGBoost(
            n_lags=5, n_estimators=200, max_depth=6,
            use_time_features=False,
        ), True),
    ]


def make_datasets() -> list[tuple[str, pd.DataFrame]]:
    """Return list of (dataset_name, df) tuples."""
    return [
        ("brownian_motion", generate_brownian_motion(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        )),
        ("crw", generate_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        )),
        ("switching_crw", generate_switching_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        )),
    ]


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format the results DataFrame as a Markdown table, dataset-grouped."""
    lines = []
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        lines.append(f"\n### {dataset}\n")
        lines.append("| Model | n | Median (m) | Mean (m) | p99 (m) | Std (m) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['model']:<14} | {int(row['n']):>5} | "
                f"{row['median_m']:>9.1f} | {row['mean_m']:>9.1f} | "
                f"{row['p99_m']:>9.1f} | {row['std_m']:>9.1f} |"
            )
    return "\n".join(lines)


def extract_feature_importances(
    xgb_model: XGBoost, dataset_name: str, top_n: int = 10
) -> pd.DataFrame:
    """Extract top-N feature importances for both per-axis regressors."""
    rows = []
    feature_names = xgb_model.feature_names_
    for axis, regressor in [
        ("dlon", xgb_model.model_dlon_),
        ("dlat", xgb_model.model_dlat_),
    ]:
        importances = regressor.feature_importances_
        # Pair names with importances, sort descending, take top N.
        paired = sorted(
            zip(feature_names, importances), key=lambda x: -x[1]
        )[:top_n]
        for rank, (name, imp) in enumerate(paired, start=1):
            rows.append({
                "dataset": dataset_name,
                "axis": axis,
                "rank": rank,
                "feature": name,
                "importance": float(imp),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _model_colour(model: str) -> str:
    return MODEL_COLOURS.get(model, "#666666")


def plot_metric_bars(
    results_df: pd.DataFrame, metric: str, ylabel: str, path: Path
) -> None:
    """Grouped bar chart: one group per dataset, bars per model."""
    datasets = list(results_df["dataset"].unique())
    models = list(results_df["model"].unique())

    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        values = [
            results_df[(results_df["dataset"] == d) & (results_df["model"] == model)]
            [metric].iloc[0]
            for d in datasets
        ]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=model,
               color=_model_colour(model))

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Synthetic baselines — {metric}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_cdf(
    raw_errors: dict[str, dict[str, np.ndarray]],
    dataset: str,
    path: Path,
) -> None:
    """Error CDF for one dataset, all models on one plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, errors in raw_errors[dataset].items():
        sorted_errors = np.sort(errors)
        y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, y, label=model, color=_model_colour(model),
                linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Prediction error (m, log scale)")
    ax.set_ylabel("Fraction of predictions with error ≤ x")
    ax.set_title(f"Error CDF — {dataset}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary(
    results_df: pd.DataFrame,
    raw_errors: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    """One big figure: bars (median, mean, p99) on top, CDFs on bottom."""
    datasets = list(results_df["dataset"].unique())
    models = list(results_df["model"].unique())
    n_models = len(models)
    bar_width = 0.8 / n_models
    x = np.arange(len(datasets))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.3)

    # Top row: three bar charts (one per metric).
    for col, (metric, title) in enumerate([
        ("median_m", "Median error (m)"),
        ("mean_m", "Mean error (m)"),
        ("p99_m", "p99 error (m)"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for i, model in enumerate(models):
            values = [
                results_df[
                    (results_df["dataset"] == d) & (results_df["model"] == model)
                ][metric].iloc[0]
                for d in datasets
            ]
            offset = (i - n_models / 2 + 0.5) * bar_width
            ax.bar(x + offset, values, bar_width, label=model,
                   color=_model_colour(model))
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

    # Bottom row: three CDFs (one per dataset).
    for col, dataset in enumerate(datasets):
        ax = fig.add_subplot(gs[1, col])
        for model, errors in raw_errors[dataset].items():
            sorted_errors = np.sort(errors)
            y = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax.plot(sorted_errors, y, label=model, color=_model_colour(model),
                    linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Prediction error (m, log scale)")
        if col == 0:
            ax.set_ylabel("Fraction of predictions ≤ x")
        ax.set_title(f"CDF — {dataset}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        if col == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Baseline ladder — synthetic data\n"
        f"({N_INDIVIDUALS} individuals × {N_STEPS} steps, "
        f"{int(TEST_FRAC * 100)}% per-individual test split)",
        fontsize=12, y=0.995,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info(
        "Generating synthetic datasets (n_individuals=%d, n_steps=%d)...",
        N_INDIVIDUALS, N_STEPS,
    )
    datasets = make_datasets()

    rows = []
    raw_errors: dict[str, dict[str, np.ndarray]] = {}
    importance_rows = []

    for dataset_name, df in datasets:
        df_train, df_test = split_by_individual(
            df, test_frac=TEST_FRAC, seed=SEED
        )
        log.info(
            "\n=== %s ===  train: %d individuals (%d rows) | "
            "test: %d individuals (%d rows)",
            dataset_name,
            df_train["individual_id"].nunique(), len(df_train),
            df_test["individual_id"].nunique(), len(df_test),
        )

        raw_errors[dataset_name] = {}

        for model_name, model, needs_fit in make_models():
            t0 = perf_counter()
            try:
                stats, results = evaluate_model(
                    model, df_test, CONTEXT_LENGTH,
                    needs_fit=needs_fit, df_train=df_train,
                )
            except Exception as exc:
                log.warning("  %-14s FAILED: %s", model_name, exc)
                continue
            total = perf_counter() - t0
            log.info(
                "  %-14s n=%5d  median=%7.1f m  mean=%7.1f m  "
                "p99=%8.1f m  (%.1fs)",
                model_name, stats["n"], stats["median_m"],
                stats["mean_m"], stats["p99_m"], total,
            )
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                **stats,
            })
            raw_errors[dataset_name][model_name] = (
                results["error_m"].to_numpy()
            )

            # Capture XGBoost feature importances per dataset.
            if isinstance(model, XGBoost) and model.is_fitted_:
                importance_rows.append(
                    extract_feature_importances(model, dataset_name,
                                                top_n=TOP_N_FEATURES)
                )

    results_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    # ---- save numeric results ----
    results_df.to_csv(RESULTS_PATH, index=False)
    log.info("\nWrote %s", RESULTS_PATH)

    if importance_rows:
        importance_df = pd.concat(importance_rows, ignore_index=True)
        importance_df.to_csv(IMPORTANCES_PATH, index=False)
        log.info("Wrote %s", IMPORTANCES_PATH)

    # ---- save raw errors per dataset (useful for re-analysis) ----
    for dataset_name, model_errors in raw_errors.items():
        long_rows = []
        for model_name, errors in model_errors.items():
            for err in errors:
                long_rows.append({"model": model_name, "error_m": err})
        pd.DataFrame(long_rows).to_csv(
            RESULTS_DIR / f"raw_errors_{dataset_name}.csv", index=False
        )

    # ---- plots ----
    plot_metric_bars(results_df, "median_m", "Median error (m)",
                     PLOTS_DIR / "bar_median.png")
    plot_metric_bars(results_df, "mean_m", "Mean error (m)",
                     PLOTS_DIR / "bar_mean.png")
    plot_metric_bars(results_df, "p99_m", "p99 error (m)",
                     PLOTS_DIR / "bar_p99.png")
    for dataset in results_df["dataset"].unique():
        plot_error_cdf(raw_errors, dataset,
                       PLOTS_DIR / f"cdf_{dataset}.png")
    plot_summary(results_df, raw_errors, PLOTS_DIR / "summary.png")
    log.info("Wrote plots to %s/", PLOTS_DIR)

    # ---- Markdown table ----
    print(format_markdown_table(results_df))

    # ---- feature importance summary ----
    if importance_rows:
        print("\n## XGBoost top features per dataset\n")
        importance_df = pd.concat(importance_rows, ignore_index=True)
        for dataset in importance_df["dataset"].unique():
            print(f"\n### {dataset}")
            for axis in ["dlon", "dlat"]:
                sub = importance_df[
                    (importance_df["dataset"] == dataset)
                    & (importance_df["axis"] == axis)
                ]
                print(f"  {axis} top features:")
                for _, row in sub.iterrows():
                    print(f"    {int(row['rank']):>2}. "
                          f"{row['feature']:<14} {row['importance']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())