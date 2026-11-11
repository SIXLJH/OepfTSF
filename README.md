# OepfTSF

**Reliability-Driven Probabilistic Time-Series Forecasting with Interleaved Supports**

Anonymous code release for review.

## Overview

OepfTSF combines history normalization, temporal decomposition, predictions on
interleaved discrete supports, and reliability-based fusion for time-series
forecasting. This package contains the model, data loaders, training loop,
MSE/MAE evaluation, and an ETTh1 launch script.

## Installation

Use the original experiment environment when reproducing existing runs. The
dependency list below contains the packages imported by the code; it is not a
record of the original experiment's exact package versions.

For a new environment:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your device. The provided launch script
requests bfloat16 AMP on CUDA; this setting requires suitable hardware and a
compatible PyTorch installation. `run.py` selects CPU automatically when CUDA
is unavailable, and the training loop disables AMP on CPU.

Some optional data-loading paths use the older pandas positional `drop` API.
Use pandas 1.x for those paths. Retain the original package versions and device
settings when comparing results from existing experiments.

## Data preparation

Dataset files are not included. Place `ETTh1.csv` at:

```text
dataset/ETT-small/ETTh1.csv
```

The CSV must contain a `date` column followed by the seven numeric ETTh1
variables. Keep the original timestamps, column order, and values.
See [dataset/README.md](dataset/README.md) for other supported input formats.

## Training and evaluation

Run commands from the repository root. On Linux, macOS, or a Bash-compatible
environment:

```bash
bash scripts/OepfTSF/ETTh1.sh
```

The script launches training and then evaluates the selected model on the test
split. Standard output is written under `logs/ema/`; error messages remain in
the terminal. The script uses an input length of **512**, a prediction length
of **96**, batch size **1024**, learning rate **0.0002**, up to **100** epochs,
and early-stopping patience **15**. The random seed in `run.py` is **2021**.
These values describe the bundled code and script.

For command-line options:

```bash
python run.py --help
```

Keep the same command-line arguments, data, package versions, and hardware
settings when comparing with previous runs. `run.py` defaults and the bundled
shell script specify different values for several training options; use the
same launch command as the experiment being compared.

## Outputs

| Path | Content |
| --- | --- |
| `logs/ema/` | Training output from the ETTh1 shell script |
| `Checkpoints/` | Temporary training checkpoints by default |
| `test_results/` | Forecast plots generated during testing |
| `result.txt` | Appended MSE and MAE summaries |

The existing training loop reloads its best checkpoint and then removes that
checkpoint file. Evaluation immediately following training uses the model
already loaded in memory. Standalone evaluation with `--is_training 0` expects
a separately retained checkpoint at `checkpoints/<setting>/checkpoint.pth`;
this path is lowercase and differs from the default training directory.

## Code layout

| Path | Purpose |
| --- | --- |
| `run.py` | Argument parsing and experiment entry point |
| `models/OepfTSF.py` | Forecasting model and support-based fusion |
| `layers/` | Normalization, decomposition, and prediction networks |
| `data_provider/` | Dataset loading and batching |
| `exp/` | Optimization, validation, and evaluation |
| `scripts/OepfTSF/ETTh1.sh` | ETTh1 launch configuration |
| `utils/` | Metrics, time features, learning-rate schedule, and plotting |
| `dataset/` | Local dataset placement |

The architecture figure already present in the repository can remain at the
repository root as `model.png`.
