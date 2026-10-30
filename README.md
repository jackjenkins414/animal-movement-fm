# animal-movement-fm

Code for our ANU honours thesis in machine learning (2026), investigating animal movement forecasting within and across species.

The project compares persistence, averaging, ARIMA, XGBoost, LSTM and Transformer models using a common forecasting and evaluation framework. Our central question is whether models trained across species can forecast a species excluded from training, using its recent observed trajectory. Separate adaptation experiments compare fine-tuning with training from scratch on the same target individuals.

## For readers and markers

Read this repository alongside the submitted thesis, which presents the findings and their limitations. The main entry points are:

| Material | Location |
| --- | --- |
| Data preparation, movement representations and synthetic processes | [src/animal_movement_fm/data/](src/animal_movement_fm/data/) |
| Model implementations and training loops | [src/animal_movement_fm/models/](src/animal_movement_fm/models/), [src/animal_movement_fm/train/](src/animal_movement_fm/train/) |
| Evaluation metrics and supporting analyses | [src/animal_movement_fm/eval/](src/animal_movement_fm/eval/), [src/animal_movement_fm/analysis/](src/animal_movement_fm/analysis/) |
| Synthetic and single species development experiments | [experiments/baselines/](experiments/baselines/) |
| Multi-species training and evaluation drivers | [experiments/training/](experiments/training/), [experiments/reporting/](experiments/reporting/) |
| Cadence, movement structure, environmental inputs and model diagnostics | [experiments/cadence/](experiments/cadence/), [experiments/eda/](experiments/eda/), [experiments/data/](experiments/data/), [experiments/diagnostics/](experiments/diagnostics/) |
| Study protocols, amendments and evaluation freezes | [preregistration/](preregistration/) |
| Recorded source pools, sampling weights and bird rotation folds | [registered/](registered/) |
| Individual split manifests | [splits/](splits/) |
| Implementation checks | [tests/](tests/) |

Our [methods log](methods.md) records decisions during development. Earlier entries describe the study at that time and may be superseded by later entries or the final thesis.

### Names used in the code

- **Tier 2** and **Tier 3** refer to the original three and nine source species comparisons.
- **Tier 4** refers to the exploratory 26 source species study.
- The **bird study** uses 22 source species; the **bird rotation** holds out each of 30 bird species in turn.
- **BbF** and related names identify repaired Transformer arms. The matching protocols document their settings.

Here, *registered* refers to methods and evaluation criteria documented in internal repository protocols. The hourly, expanded pool and repaired model studies are exploratory. Several extensions reuse previously evaluated populations, so these are not independent replications.

## Status

The experiments reported in the thesis are complete. They support useful forecasting for many excluded bird species with pooled LSTM, XGBoost and repaired Transformer models, while larger mixed source pools and the original fine-tuning procedure gave no consistent benefit. Longer histories helped six hour forecasts in the hourly studies; the tested ERA5 inputs gave small median changes. The project remains ongoing, and we have planned further development beyond the thesis.

## Data and reproducibility

This repository contains the code, protocols and recorded study artefacts. Tracking datasets, processed trajectories, trained weights, prediction files and generated result tables are not included. Data access remains subject to the original providers' conditions.

Our real data experiments require those external inputs and the corresponding study settings. Many training drivers and batch scripts target NCI's Gadi computing system and contain environment specific paths. Installing the package alone does not reproduce the thesis results. Reporting scripts also require the generated evaluation outputs.

However, our synthetic movement generators and their benchmark driver are available in [src/animal_movement_fm/data/](src/animal_movement_fm/data/) and [experiments/baselines/run_baselines_synthetic.py](experiments/baselines/run_baselines_synthetic.py); they do not require the original tracking datasets.

## Local setup

Requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jackjenkins414/animal-movement-fm.git
cd animal-movement-fm
uv sync --extra dev
```

To run the implementation checks from the repository directory:

```bash
uv run pytest -q
```

## Permissions

See [LICENSE](LICENSE) for the permissions and restrictions that apply to this repository.
