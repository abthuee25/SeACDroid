"""Train SeACDroid with a same-year stratified split."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from seacdroid.data import load_dataset
from seacdroid.models import GatedAttentionMIL
from seacdroid.training.common import (
    build_model_config,
    evaluate,
    make_loader,
    split_dataset,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SeACDroid on one year with train/validation/test splits")
    parser.add_argument("--data_dir", required=True, help="Feature root containing <year>/{benign,malware}/*.pkl")
    parser.add_argument("--year", required=True, help="Year to split, e.g., 2021")
    parser.add_argument("--output_dir", default="outputs/same_year")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio within the non-test split")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Held-out test ratio within the selected year")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=1228,
        help="Seed for the stratified data split only",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--malware_weight", type=float, default=1.2)
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--max_nodes", type=int, default=0)
    parser.add_argument("--directed", action="store_true", help="Use directed graph edges instead of the default undirected edges")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--context_dim", type=int, default=64)
    parser.add_argument("--mil_hidden_dim", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=32)
    parser.add_argument("--gat_layers", type=int, default=3)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_checkpoint", action="store_true", help="Save the validation-selected model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"same_year_{args.year}"

    undirected = not args.directed
    samples = load_dataset(args.data_dir, [args.year], max_nodes=args.max_nodes, undirected=undirected)
    train_samples, val_samples, test_samples = split_dataset(samples, args.val_ratio, args.test_ratio, args.split_seed)
    LOGGER.info("Same-year split %s: train=%d validation=%d test=%d", args.year, len(train_samples), len(val_samples), len(test_samples))

    train_loader = make_loader(train_samples, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_samples, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_samples, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_config = build_model_config(args)
    model = GatedAttentionMIL(**model_config).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.malware_weight], device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_metrics = None
    best_state = None
    best_checkpoint = None
    history = []
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        update_steps = 0
        for batch_data in train_loader:
            graph_batch = batch_data["batch"].to(device)
            graph_to_apk = batch_data["graph_to_apk"].to(device)
            labels = batch_data["labels"].to(device)
            if (graph_to_apk >= 0).sum() == 0:
                continue

            optimizer.zero_grad()
            logits, unique_apks, _ = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, graph_to_apk)
            loss = criterion(logits, labels[unique_apks])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            update_steps += 1

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1"])
        avg_loss = total_loss / max(update_steps, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "validation": val_metrics})
        LOGGER.info("Epoch %02d/%02d | loss=%.4f | val_f1=%.4f", epoch, args.epochs, avg_loss, val_metrics["f1"])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "train_mode": "same_year",
            "train_year": args.year,
            "split_seed": args.split_seed,
            "max_nodes": args.max_nodes,
            "undirected": undirected,
            "validation_metrics": val_metrics,
        }
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint = {**checkpoint, "model_state_dict": best_state}
            no_improve = 0
            if args.save_checkpoint:
                torch.save(best_checkpoint, output_dir / f"{run_name}_best_validation.pth")
        else:
            no_improve += 1
            if args.early_stop > 0 and no_improve >= args.early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    report = {
        "run_name": run_name,
        "mode": "same_year",
        "year": args.year,
        "split_seed": args.split_seed,
        "split_counts": {
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
        },
        "best_validation": best_metrics,
        "heldout_test": test_metrics,
        "history": history,
        "checkpoint_saved": bool(args.save_checkpoint and best_checkpoint is not None),
    }
    (output_dir / f"{run_name}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Same-year test metrics: %s", test_metrics)
    if not args.save_checkpoint:
        LOGGER.info("Same-year checkpoint was not saved; use --save_checkpoint to keep it.")


if __name__ == "__main__":
    main()
