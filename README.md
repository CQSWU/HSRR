# HSRR

**Hybrid Switching with Anti-Oscillation Replanning and Action Reweighting
for Lifelong Partially Observable Multi-Agent Pathfinding.**

HSRR combines AORePlan and ARPE with a learned Switcher. When AORePlan
returns wait, ARPE is used directly; otherwise, the Switcher selects a branch.

| RePlan | AORePlan |
| :---: | :---: |
| <img src="docs/assets/replan.svg" alt="RePlan animation" width="420"> | <img src="docs/assets/aoreplan.svg" alt="AORePlan animation" width="420"> |

## Install

Python 3.10 or 3.11 on Windows or Linux. Linux also needs a C++ compiler.

```bash
git clone https://github.com/CQSWU/HSRR.git
cd HSRR
uv sync
```

## Run

```bash
uv run python evaluate.py --algorithms AORePlan
```

Evaluation uses all 36 maps in `maps/test.yaml` by default.

Download the [model weights](https://github.com/CQSWU/HSRR/releases/tag/weights-v1)
and extract them into `weights/`. ARPE needs `EPOM-L` and `ARPE-Final-1B`;
HSRR also needs `HSRR-Switcher-Final-1B`. Experiment data is not included.

```bash
uv run python evaluate.py --algorithms HSRR
```

## Train

```bash
uv run python train_arpe.py --config_path configs/train_arpe.yaml
uv run python train_switcher.py --config_path configs/train_switcher.yaml
```

Both models train with 200 agents and individual rewards. ARPE uses the same
entropy gate during training and inference. Settings are in the YAML files.
Training and test maps are in `maps/train.yaml` and `maps/test.yaml`.

## License

[MIT](LICENSE). AORePlan builds on RePlan, and ARPE uses EPOM as its base policy;
the original authors' notices are preserved in the license.
