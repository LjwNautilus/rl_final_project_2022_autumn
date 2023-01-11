# rl_final_project_2022_autumn

This repository is the codebase of *Reinforcement Learning and Game Theory*  course in SYSU in 2022 autumn. Author of this repository is Jingwu Luo 罗京武 with student number 20337087.

## File Structure

- Directory `/comments/` stores comments of key source code. Three scenario files are commented: `balance.py`, `transport.py`, `wheel.py`.
- Directory `/utils/` stores utilitiy code shared by algorithms.
  - `network.py` defines network architectures.
  - `replay_memory.py` implements experience replay.
  - `run_env.py` defines a function `run_env` to test policies.
- `cppo.py`, `ippo.py` defines PPO-based algorithms.
- The main function lies in `main.py`.

## Usage

```bash
# python main.py [balance | transport | wheel] [ippo | cppo] [cpu | cuda:$GPU_ID]
python main.py
```

- The first optional argument is name of the scenario (default: balance).
- The second optional argument is the PPO-based algorithm (default: ippo).
- The third optional argument is the device (default: cuda if it is available else cpu).

## Requirements

```requirements
vmas
torch
```
