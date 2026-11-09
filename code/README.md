# 10-703 Homework 5: Model-Based and Offline RL — Starter Code

## Setup

Requires Python 3.10–3.12. Everything installs from pip (no conda, swig, or compilers needed).

```bash
conda create -n 703hw5 python=3.12 && conda activate 703hw5   # or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For Problem 1.4, download the pretrained dynamics model `model.pt` (link in the handout) into this directory.
For Problems 2–3, the Minari dataset is downloaded automatically on first use (~150 MB, stored in `~/.minari`).

All commands below are run from this `code/` directory.

## File structure

Files marked ✏️ contain code you must write; search for `BEGIN STUDENT SOLUTION`.

```
common/                 shared by both parts
  td3.py            ✏️  TD3 (used in 1.4 and 2-3); BC regularization (2.3) and CQL loss (3.1)
  replay_buffer.py      replay buffer used by every TD3 run
pets/                   Problem 1 (model-based RL)
  envs/                 2D pushing environment (Box2D) and its renderer
  agent.py              rollout helper
  optimizers.py         CEM and random-shooting optimizers
  model.py          ✏️  probabilistic ensemble: loss and training (1.2)
  mpc.py            ✏️  MPC: ground-truth (1.1) and model-based (1.2) next-state prediction
offline/                Problems 2-3 (offline RL)
  data.py           ✏️  load the Minari dataset into the replay buffer (2.1)
  evaluation.py         evaluation, videos, training-curve plots
run_pets.py             entry point for 1.1-1.3
run_synthetic.py  ✏️  entry point for 1.4; get_synthetic_transition (1.4)
run_offline.py          entry point for 2-3
```

## Running

| Problem | Command | Outputs |
|---|---|---|
| 1.1 | `python run_pets.py --question 1.1` | `out/log.txt` |
| 1.2 | `python run_pets.py --question 1.2` | `out/log.txt`, `out/1.2.1-loss.png` |
| 1.3 | `python run_pets.py --question 1.3` (~30 min) | `out/1.3.1-loss.png`, `out/1.3.2-*.png` |
| 1.4.1–1.4.3 | `python run_synthetic.py --question 1.4.1` (or `1.4.2`, `1.4.3`; ~1 h total) | `out/1.4.*.png` |
| 2.2 | `python run_offline.py` (~30 min) | `runs/<config>/<time>/training_curves.png` |
| 2.3 | `python run_offline.py --bc_regularization_weight <alpha>` | same |
| 3.1 | `python run_offline.py --cql_alpha 5` | same |

Use `python <script> --help` for all options.
