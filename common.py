"""Shared training utilities for SeACDroid training entry points."""

from __future__ import annotations

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from seacdroid.data import collate_apk_batch
from seacdroid.metrics import binary_metrics
from seacdroid.models import GatedAttentionMIL, default_model_config


def evaluate(model: GatedAttentionMIL, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate one model on a DataLoader."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch_data in loader:
            if batch_data["empty_labels"] is not None:
                labels.extend(batch_data["empty_labels"].tolist())
                preds.extend([0] * len(batch_data["empty_labels"]))

            graph_batch = batch_data["batch"].to(device)
            graph_to_apk = batch_data["graph_to_apk"].to(device)
            if (graph_to_apk >= 0).sum() == 0:
                continue

            logits, unique_apks, _ = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, graph_to_apk)
            batch_labels = batch_data["labels"][unique_apks.cpu()].tolist()
            labels.extend(batch_labels)
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    return binary_metrics(labels, preds)


def make_loader(samples: list[dict], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create a PyTorch DataLoader for APK bags."""
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_apk_batch,
        num_workers=num_workers,
    )


def split_dataset(samples: list[dict], val_ratio: float, test_ratio: float, split_seed: int):
    """Create stratified train/validation(/test) splits."""
    labels = [sample["label"] for sample in samples]

    test_samples: list[dict] = []
    train_val_samples = samples
    if test_ratio > 0:
        train_val_samples, test_samples = train_test_split(
            samples,
            test_size=test_ratio,
            random_state=split_seed,
            stratify=labels,
        )

    train_val_labels = [sample["label"] for sample in train_val_samples]
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_ratio,
        random_state=split_seed,
        stratify=train_val_labels,
    )
    return train_samples, val_samples, test_samples


def build_model_config(args) -> dict:
    """Build model configuration from command-line arguments."""
    config = default_model_config()
    config.update(
        {
            "hidden_dim": args.hidden_dim,
            "context_dim": args.context_dim,
            "mil_hidden_dim": args.mil_hidden_dim,
            "classifier_hidden_dim": args.classifier_hidden_dim,
            "num_layers": args.gat_layers,
            "heads": args.attention_heads,
            "dropout": args.dropout,
            "encoder_type": "gat",
            "pool_type": "mean",
        }
    )
    return config
