"""
Training loop for elution sequence prediction models.

Supports LSTM and Transformer architectures.
Implements early stopping, cosine annealing LR, and periodic evaluation.
Logs metrics to JSON for reproducibility.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    RANDOM_SEED,
    OUTPUTS,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    NUM_HEADS,
    FF_DIM,
    CONTEXT_LENGTH,
    BATCH_SIZE,
)
from datasets import create_dataloaders
from models import LSTMModel, TransformerModel, count_parameters


def set_seeds(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device, top_k=(1, 3, 5, 10)):
    """Evaluate model on a dataloader. Returns metrics dict."""
    model.eval()
    total_loss = 0
    total_correct = {k: 0 for k in top_k}
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            mz = batch["ctx_mz"].to(device)
            md = batch["ctx_md"].to(device)
            gap = batch["ctx_gap"].to(device)
            pol = batch["ctx_pol"].to(device)
            inten = batch["ctx_int"].to(device)
            target = batch["target_mz"].to(device)

            logits = model(mz, md, gap, pol, inten)
            loss = criterion(logits, target)
            total_loss += loss.item() * target.size(0)

            # Top-k accuracy
            for k in top_k:
                _, topk_preds = logits.topk(k, dim=1)
                correct = topk_preds.eq(target.unsqueeze(1)).any(dim=1).sum().item()
                total_correct[k] += correct

            # MAE
            pred_mz = logits.argmax(dim=1)
            all_preds.append(pred_mz.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            total_samples += target.size(0)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae_bins = np.mean(np.abs(all_preds.astype(float) - all_targets.astype(float)))

    metrics = {
        "loss": total_loss / total_samples,
        "mae_bins": float(mae_bins),
        "mae_da": float(mae_bins * 10),  # 10 Da per bin
        "n_samples": total_samples,
    }
    for k in top_k:
        metrics[f"top{k}_accuracy"] = total_correct[k] / total_samples

    return metrics


def train_model(model_type="lstm", max_epochs=MAX_EPOCHS, patience=PATIENCE,
                lr=LEARNING_RATE, device=None):
    """Train a model end-to-end. Returns trained model and metrics history."""
    set_seeds()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, meta = create_dataloaders(
        context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
    )
    num_classes = meta["max_mz_bin"]

    # Model
    embed_kwargs = dict(
        max_mz_bin=num_classes,
        max_md_bin=20,
        max_rt_gap=7,
        max_polarity=3,
        max_intensity=5,
    )

    if model_type == "lstm":
        model = LSTMModel(
            num_classes,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            **embed_kwargs,
        )
    elif model_type == "transformer":
        model = TransformerModel(
            num_classes,
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            context_length=CONTEXT_LENGTH,
            **embed_kwargs,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    print(f"\n{model_type.upper()} model: {count_parameters(model):,} parameters")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    history = []
    checkpoint_path = OUTPUTS / "checkpoints" / f"{model_type}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...\n")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        t0 = time.time()

        for batch in train_loader:
            mz = batch["ctx_mz"].to(device)
            md = batch["ctx_md"].to(device)
            gap = batch["ctx_gap"].to(device)
            pol = batch["ctx_pol"].to(device)
            inten = batch["ctx_int"].to(device)
            target = batch["target_mz"].to(device)

            optimizer.zero_grad()
            logits = model(mz, md, gap, pol, inten)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            train_correct += (logits.argmax(1) == target).sum().item()
            train_samples += target.size(0)

        scheduler.step()
        train_time = time.time() - t0

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_samples,
            "train_acc": train_correct / train_samples,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_accuracy"],
            "val_top5": val_metrics["top5_accuracy"],
            "val_mae_da": val_metrics["mae_da"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_s": train_time,
        }
        history.append(record)

        # Print progress
        print(f"Epoch {epoch:3d}/{max_epochs} | "
              f"train_loss={record['train_loss']:.4f} train_acc={record['train_acc']:.4f} | "
              f"val_loss={record['val_loss']:.4f} val_top1={record['val_top1']:.4f} "
              f"val_top5={record['val_top5']:.4f} val_MAE={record['val_mae_da']:.0f}Da | "
              f"{train_time:.0f}s")

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "model_type": model_type,
                "config": {
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS,
                    "ff_dim": FF_DIM,
                    "context_length": CONTEXT_LENGTH,
                    "num_classes": num_classes,
                },
            }, checkpoint_path)
        elif epoch - best_epoch >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Final test evaluation with best model
    print(f"\nLoading best model from epoch {best_epoch}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"  Loss:         {test_metrics['loss']:.4f}")
    print(f"  Top-1 acc:    {test_metrics['top1_accuracy']:.4f}")
    print(f"  Top-3 acc:    {test_metrics['top3_accuracy']:.4f}")
    print(f"  Top-5 acc:    {test_metrics['top5_accuracy']:.4f}")
    print(f"  Top-10 acc:   {test_metrics['top10_accuracy']:.4f}")
    print(f"  MAE:          {test_metrics['mae_da']:.0f} Da")

    # Save results
    results_dir = OUTPUTS / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_type": model_type,
        "best_epoch": best_epoch,
        "parameters": count_parameters(model),
        "test_metrics": test_metrics,
        "history": history,
    }
    with open(results_dir / f"{model_type}_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_dir / f'{model_type}_results.json'}")

    return model, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    model, results = train_model(
        model_type=args.model,
        max_epochs=args.epochs,
        patience=args.patience,
    )
