"""
End-to-end runner for the adversarial ML course project.

Phases:
  1. Train / load baseline CIFAR-10 CNN
  2. Evaluate clean metrics
  3. Run FGSM + PGD attacks
  4. Apply JPEG defense + adversarial training
  5. Export metrics table, confusion matrices, and example images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from attacks import make_fgsm, make_pgd
from defenses import adversarial_train, make_jpeg_defense
from models import build_model, count_parameters, get_device, load_checkpoint, train_model
from utils.data import get_dataloaders
from utils.metrics import evaluate_attack, evaluate_clean, metrics_to_row
from utils.visualization import (
    save_adversarial_grid,
    save_confusion_matrix,
    save_results_table_figure,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adversarial ML project runner")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--figures", type=str, default="figures")
    p.add_argument("--epochs", type=int, default=10, help="Baseline training epochs")
    p.add_argument("--adv-epochs", type=int, default=5, help="Adversarial training epochs")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epsilon", type=float, default=8 / 255, help="L∞ attack budget")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--pgd-alpha", type=float, default=2 / 255)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Optional train subset size for quick smoke tests",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast demo: fewer epochs, subset data, fewer PGD steps",
    )
    p.add_argument(
        "--skip-adv-train",
        action="store_true",
        help="Skip adversarial training (use baseline + JPEG only)",
    )
    p.add_argument(
        "--baseline-ckpt",
        type=str,
        default="models/checkpoints/baseline.pt",
    )
    p.add_argument(
        "--robust-ckpt",
        type=str,
        default="models/checkpoints/robust.pt",
    )
    p.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Limit attack evaluation to N batches (speed)",
    )
    return p.parse_args()


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_attack_examples(
    model,
    loader,
    attack_fn,
    device,
    figures_dir: Path,
    name: str,
) -> None:
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
    adv = attack_fn(model, images, labels)
    with torch.no_grad():
        adv_preds = model(adv).argmax(dim=1)
    save_adversarial_grid(
        images,
        adv,
        labels,
        clean_preds,
        adv_preds,
        figures_dir / f"adv_examples_{name}.png",
        n=8,
        title=f"Clean vs {name.upper()} Adversarial",
    )


def main() -> None:
    args = parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 2)
        args.adv_epochs = min(args.adv_epochs, 1)
        args.subset = args.subset or 2000
        args.pgd_steps = min(args.pgd_steps, 3)
        args.eval_batches = args.eval_batches or 5
        args.batch_size = min(args.batch_size, 64)

    outdir = Path(args.outdir)
    figures = Path(args.figures)
    ensure_dirs(outdir, figures, "models/checkpoints", "data")

    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        subset_size=args.subset,
    )

    # ------------------------------------------------------------------
    # Phase 1: Baseline model
    # ------------------------------------------------------------------
    baseline = build_model().to(device)
    print(f"Model parameters: {count_parameters(baseline):,}")

    baseline_path = Path(args.baseline_ckpt)
    if baseline_path.exists():
        print(f"Loading baseline from {baseline_path}")
        load_checkpoint(baseline, baseline_path, device)
    else:
        print("Training baseline model...")
        train_model(
            baseline,
            train_loader,
            test_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            checkpoint_path=baseline_path,
        )

    clean = evaluate_clean(baseline, test_loader, device)
    print(f"Baseline clean accuracy: {clean['accuracy']:.4f}")
    save_confusion_matrix(
        clean["confusion_matrix"],
        figures / "cm_baseline_clean.png",
        title="Baseline — Clean",
    )

    # ------------------------------------------------------------------
    # Phase 3: Attacks
    # ------------------------------------------------------------------
    fgsm = make_fgsm(epsilon=args.epsilon)
    pgd = make_pgd(epsilon=args.epsilon, alpha=args.pgd_alpha, steps=args.pgd_steps)

    print("Evaluating FGSM on baseline...")
    fgsm_metrics = evaluate_attack(
        baseline, test_loader, device, fgsm, max_batches=args.eval_batches
    )
    print(
        f"  FGSM  robust_acc={fgsm_metrics['robust_accuracy']:.4f}  "
        f"ASR={fgsm_metrics['asr']:.4f}  "
        f"L∞̄={fgsm_metrics['perturbation_linf_mean']:.4f}"
    )
    save_confusion_matrix(
        fgsm_metrics["confusion_matrix"],
        figures / "cm_baseline_fgsm.png",
        title="Baseline — FGSM",
    )
    save_attack_examples(baseline, test_loader, fgsm, device, figures, "fgsm")

    print("Evaluating PGD on baseline...")
    pgd_metrics = evaluate_attack(
        baseline, test_loader, device, pgd, max_batches=args.eval_batches
    )
    print(
        f"  PGD   robust_acc={pgd_metrics['robust_accuracy']:.4f}  "
        f"ASR={pgd_metrics['asr']:.4f}  "
        f"L∞̄={pgd_metrics['perturbation_linf_mean']:.4f}"
    )
    save_confusion_matrix(
        pgd_metrics["confusion_matrix"],
        figures / "cm_baseline_pgd.png",
        title="Baseline — PGD",
    )
    save_attack_examples(baseline, test_loader, pgd, device, figures, "pgd")

    # Primary attack for "After Attack" column = stronger PGD
    after_attack = pgd_metrics

    # ------------------------------------------------------------------
    # Phase 4: Defenses
    # ------------------------------------------------------------------
    jpeg = make_jpeg_defense(quality=args.jpeg_quality)

    print("Evaluating PGD + JPEG defense on baseline...")
    jpeg_metrics = evaluate_attack(
        baseline,
        test_loader,
        device,
        pgd,
        defense_fn=jpeg,
        max_batches=args.eval_batches,
    )
    print(
        f"  JPEG  robust_acc={jpeg_metrics['robust_accuracy']:.4f}  "
        f"ASR={jpeg_metrics['asr']:.4f}"
    )
    save_confusion_matrix(
        jpeg_metrics["confusion_matrix"],
        figures / "cm_jpeg_pgd.png",
        title="JPEG Defense — PGD",
    )

    jpeg_clean = evaluate_clean(baseline, test_loader, device, defense_fn=jpeg)
    print(f"  JPEG clean accuracy: {jpeg_clean['accuracy']:.4f}")

    robust_metrics = None
    if not args.skip_adv_train:
        robust_path = Path(args.robust_ckpt)
        robust = build_model().to(device)
        if robust_path.exists():
            print(f"Loading robust model from {robust_path}")
            load_checkpoint(robust, robust_path, device)
        else:
            print("Adversarial training (PGD-in-the-loop)...")
            # Warm-start from baseline weights when available
            if baseline_path.exists():
                load_checkpoint(robust, baseline_path, device)
            adversarial_train(
                robust,
                train_loader,
                test_loader,
                device,
                epochs=args.adv_epochs,
                lr=args.lr,
                epsilon=args.epsilon,
                alpha=args.pgd_alpha,
                pgd_steps=min(5, args.pgd_steps),
                checkpoint_path=robust_path,
            )

        robust_clean = evaluate_clean(robust, test_loader, device)
        print(f"Robust model clean accuracy: {robust_clean['accuracy']:.4f}")

        print("Evaluating PGD on adversarially trained model...")
        robust_metrics = evaluate_attack(
            robust, test_loader, device, pgd, max_batches=args.eval_batches
        )
        print(
            f"  AdvTrain robust_acc={robust_metrics['robust_accuracy']:.4f}  "
            f"ASR={robust_metrics['asr']:.4f}"
        )
        save_confusion_matrix(
            robust_metrics["confusion_matrix"],
            figures / "cm_advtrain_pgd.png",
            title="Adversarial Training — PGD",
        )

        # Combined defense: adv-trained model + JPEG at inference
        print("Evaluating PGD on AdvTrain + JPEG...")
        combined = evaluate_attack(
            robust,
            test_loader,
            device,
            pgd,
            defense_fn=jpeg,
            max_batches=args.eval_batches,
        )
        print(
            f"  Combined robust_acc={combined['robust_accuracy']:.4f}  "
            f"ASR={combined['asr']:.4f}"
        )
        after_defense = combined
        after_defense_clean = evaluate_clean(robust, test_loader, device, defense_fn=jpeg)
    else:
        after_defense = jpeg_metrics
        after_defense_clean = jpeg_clean

    # ------------------------------------------------------------------
    # Phase 5: Required results table
    # ------------------------------------------------------------------
    # "Accuracy" column uses clean accuracy for baseline / defended model;
    # after-attack accuracy is the robust accuracy under PGD.
    table = {
        "Baseline": {
            **metrics_to_row(clean),
            "asr": 0.0,
        },
        "After Attack (PGD)": metrics_to_row(after_attack),
        "After Defense": {
            **metrics_to_row(after_defense),
            # Prefer clean accuracy of the defended system when available
            "accuracy": after_defense_clean["accuracy"],
        },
    }

    # Extra detail rows for the report
    detail = {
        "baseline_clean": metrics_to_row(clean),
        "baseline_fgsm": metrics_to_row(fgsm_metrics),
        "baseline_pgd": metrics_to_row(pgd_metrics),
        "jpeg_vs_pgd": metrics_to_row(jpeg_metrics),
        "jpeg_clean": metrics_to_row(jpeg_clean),
    }
    if robust_metrics is not None:
        detail["advtrain_vs_pgd"] = metrics_to_row(robust_metrics)
        detail["advtrain_jpeg_vs_pgd"] = metrics_to_row(after_defense)

    df = pd.DataFrame(table).T
    csv_path = outdir / "results_table.csv"
    df.to_csv(csv_path)
    print("\n=== Required Results Table ===")
    print(df.round(3).to_string())

    with open(outdir / "metrics_detail.json", "w") as f:
        json.dump(
            {
                "table": table,
                "detail": detail,
                "config": {
                    "epochs": args.epochs,
                    "adv_epochs": args.adv_epochs,
                    "epsilon": args.epsilon,
                    "pgd_steps": args.pgd_steps,
                    "jpeg_quality": args.jpeg_quality,
                    "subset": args.subset,
                    "device": str(device),
                },
                "perturbation": {
                    "fgsm_linf_mean": fgsm_metrics.get("perturbation_linf_mean"),
                    "fgsm_l2_mean": fgsm_metrics.get("perturbation_l2_mean"),
                    "pgd_linf_mean": pgd_metrics.get("perturbation_linf_mean"),
                    "pgd_l2_mean": pgd_metrics.get("perturbation_l2_mean"),
                },
            },
            f,
            indent=2,
        )

    save_results_table_figure(table, figures / "results_table.png")
    print(f"\nSaved CSV → {csv_path}")
    print(f"Saved figures → {figures}/")
    print("Done.")


if __name__ == "__main__":
    main()
