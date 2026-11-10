# ln2149-tactile-slip

Official implementation for the paper **Tactile Slip Detection with Low-Cost Optical Sensors**.

## Citation

> Jennifer Baker, Kenji Mori, Sofia Neves. "Tactile Slip Detection with Low-Cost Optical Sensors." Accepted at ICRA 2026.

## What is included

- `src/model.py` — model definition
- `src/train.py` — training entry point
- `src/rollout.py` — evaluation / rollout utilities
- `src/dataset.py`, `src/metrics.py`, `src/utils.py` — data and metric helpers
- `configs/basins/` — per-basin configs used for the reported results
- `configs/` — experiment configuration
- `scripts/prepare_data.py`, `scripts/run_experiments.sh` — data prep and experiment driver
- `tests/` — smoke tests for the model and data pipeline

The repository contains the complete training and evaluation pipeline used for the
results reported in the paper. See `docs/setup.md` for environment setup and
`scripts/run_experiments.sh` for the full experiment driver.
