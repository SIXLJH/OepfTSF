# Dataset placement

Dataset files are local inputs and are not distributed with this code release.

| `--data` | Expected input | Typical path |
| --- | --- | --- |
| `ETTh1`, `ETTh2` | ETT hourly CSV with a `date` column | `dataset/ETT-small/ETTh1.csv`, `ETTh2.csv` |
| `ETTm1`, `ETTm2` | ETT minute CSV with a `date` column | `dataset/ETT-small/ETTm1.csv`, `ETTm2.csv` |
| `custom` | CSV with a `date` column and numeric variables | Set `--root_path` and `--data_path` |
| `Solar` | Comma-separated numeric rows without a header | Set `--root_path` and `--data_path` |
| `PEMS` | NumPy `.npz` archive with a `data` array | Set `--root_path` and `--data_path` |

Match `--enc_in` to the number of modeled variables. `--features M` uses all
numeric columns; `S` uses the target; `MS` evaluates the last modeled column as
the target, so preserve the required column order. Use the same `--freq` as the
experiment being reproduced.

The data loaders keep their existing chronological splits and scaling:

- ETT hourly: the fixed 12/4/4-month boundaries defined in `data_loader.py`.
- ETT minute: the corresponding boundaries at four samples per hour.
- Custom and Solar: the existing 70%/10%/20% logic.
- PEMS: the existing 60%/20%/20% logic, using `data[:, :, 0]`.

The existing batching implementation uses `drop_last=True` for training,
validation, and test data. Keep the same batch size when comparing metrics.
