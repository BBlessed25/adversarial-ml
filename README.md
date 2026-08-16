# Adversarial Attacks and Defenses


**Framework:** PyTorch

This project trains an image classifier, attacks it with white-box adversarial examples, and evaluates defenses. Quantitative security metrics are reported throughout.

## Repository layout

```text
├── attacks/           # FGSM, PGD
├── defenses/          # Adversarial training, JPEG compression
├── models/            # CNN architecture + training utilities
├── utils/             # Data, metrics, visualization
├── notebooks/         # Exploratory notebooks
├── figures/           # Confusion matrices, adv examples, results table
├── results/           # CSV / JSON metrics
├── requirements.txt
└── run_project.py     # End-to-end pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick demo (smoke test)

Uses a data subset and fewer epochs so you can verify the pipeline:

```bash
python run_project.py --quick
```

## Full experiment

```bash
python run_project.py --epochs 15 --adv-epochs 8
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--quick` | Subset + short training for a fast run |
| `--skip-adv-train` | Evaluate JPEG defense only |
| `--epsilon 0.031` | L∞ budget (default `8/255`) |
| `--pgd-steps 10` | PGD iterations |
| `--jpeg-quality 75` | JPEG defense quality |
| `--baseline-ckpt PATH` | Reuse a saved baseline |

Checkpoints are written to `models/checkpoints/`. Re-running loads them automatically.

## What the pipeline does

1. **Baseline** — Train `SimpleCNN` on CIFAR-10; report clean accuracy / P / R / F1 / confusion matrix  
2. **Attacks** — FGSM and PGD; report robust accuracy, ASR, mean L∞ / L2 perturbation  
3. **Defenses** — JPEG compression at inference; Madry-style adversarial training; optional combined defense  
4. **Exports** — `results/results_table.csv`, `results/metrics_detail.json`, plots under `figures/`

## Required metrics table

Produced automatically as `results/results_table.csv` and `figures/results_table.png`:

| Metric | Baseline | After Attack | After Defense |
|---|---|---|---|
| Accuracy | ✓ | ✓ | ✓ |
| Robust Accuracy | ✓ | ✓ | ✓ |
| ASR | ✓ | ✓ | ✓ |
| Precision | ✓ | ✓ | ✓ |
| Recall | ✓ | ✓ | ✓ |
| F1-score | ✓ | ✓ | ✓ |

pFool or CW) (`+2`)  
- Small Streamlit dashboard over `results/metrics_detail.json` (`+2`)
