"""Evaluate a trained SeACDroid checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seacdroid.data import collate_apk_batch, load_dataset
from seacdroid.metrics import binary_metrics
from seacdroid.models import GatedAttentionMIL, default_model_config

LOGGER = logging.getLogger(__name__)


def make_loader(samples: list[dict], batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_apk_batch,
        num_workers=num_workers,
    )


def evaluate(model: GatedAttentionMIL, loader: DataLoader, device: torch.device, collect_predictions: bool = False) -> dict:
    """Evaluate a model and optionally return per-sample predictions."""
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    rows: list[dict] = []

    with torch.no_grad():
        for batch_data in loader:
            if batch_data["empty_labels"] is not None:
                empty_label = batch_data["empty_labels"].tolist()
                labels.extend(empty_label)
                preds.extend([0] * len(empty_label))
                if collect_predictions:
                    for meta, label in zip(batch_data["empty_metadata"], empty_label):
                        rows.append(
                            {
                                "sha256": meta.get("sha256", ""),
                                "path": meta.get("path", ""),
                                "year": meta.get("year", ""),
                                "label": label,
                                "prediction": 0,
                                "prob_malware": 0.0,
                            }
                        )

            graph_batch = batch_data["batch"].to(device)
            graph_to_apk = batch_data["graph_to_apk"].to(device)
            if (graph_to_apk >= 0).sum() == 0:
                continue

            logits, unique_apks, _ = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, graph_to_apk)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            pred = logits.argmax(dim=1).cpu().tolist()
            label = batch_data["labels"][unique_apks.cpu()].tolist()

            labels.extend(label)
            preds.extend(pred)

            if collect_predictions:
                for local_idx, sample_idx in enumerate(unique_apks.cpu().tolist()):
                    meta = batch_data["metadata"][sample_idx]
                    rows.append(
                        {
                            "sha256": meta.get("sha256", ""),
                            "path": meta.get("path", ""),
                            "year": meta.get("year", ""),
                            "label": label[local_idx],
                            "prediction": pred[local_idx],
                            "prob_malware": prob[local_idx],
                        }
                    )

    metrics = binary_metrics(labels, preds)
    if collect_predictions:
        metrics["predictions"] = rows
    return metrics


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GatedAttentionMIL, dict]:
    """Load a SeACDroid checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config") or default_model_config()
    model = GatedAttentionMIL(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def write_prediction_csv(path: Path, rows: list[dict]) -> None:
    """Write per-sample predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sha256", "path", "year", "label", "prediction", "prob_malware"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SeACDroid")
    parser.add_argument("--data_dir", required=True, help="Feature root: data_dir/<dataset>/{benign,malware}/*.pkl")
    parser.add_argument("--checkpoint", default=None, help="Path to one trained .pth checkpoint")
    parser.add_argument("--checkpoint_dir", default=None, help="Directory containing checkpoints to evaluate")
    parser.add_argument("--checkpoint_glob", default="*.pth", help="Glob used with --checkpoint_dir")
    parser.add_argument("--test_years", nargs="+", required=True, help="Test years or dataset names")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_nodes", type=int, default=None, help="Override checkpoint max_nodes")
    parser.add_argument("--directed", action="store_true", help="Use directed graph edges instead of the default checkpoint setting")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--prediction_dir", default=None, help="Directory for per-sample CSV predictions")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    if args.checkpoint is None and args.checkpoint_dir is None:
        parser.error("one of --checkpoint or --checkpoint_dir is required")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_paths: list[Path] = []
    if args.checkpoint is not None:
        checkpoint_paths.append(Path(args.checkpoint))
    if args.checkpoint_dir is not None:
        checkpoint_paths.extend(sorted(Path(args.checkpoint_dir).glob(args.checkpoint_glob)))
    if not checkpoint_paths:
        parser.error("no checkpoint files found")

    LOGGER.info("Device=%s, checkpoints=%d", device, len(checkpoint_paths))

    results = []
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_model(checkpoint_path, device)
        max_nodes = checkpoint.get("max_nodes", 0) if args.max_nodes is None else args.max_nodes
        undirected = bool(checkpoint.get("undirected", True) and not args.directed)
        LOGGER.info("Checkpoint=%s, max_nodes=%s, undirected=%s", checkpoint_path.name, max_nodes, undirected)

        for year in args.test_years:
            samples = load_dataset(args.data_dir, [year], max_nodes=max_nodes, undirected=undirected)
            loader = make_loader(samples, args.batch_size, args.num_workers)
            collect_predictions = args.prediction_dir is not None
            metrics = evaluate(model, loader, device, collect_predictions=collect_predictions)

            predictions = metrics.pop("predictions", None)
            row = {
                "checkpoint": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "epoch": checkpoint.get("epoch"),
                "train_mode": checkpoint.get("train_mode", ""),
                "train_set": checkpoint.get("train_set", checkpoint.get("train_year", "")),
                "test_year": year,
                **metrics,
            }
            results.append(row)
            LOGGER.info(
                "%s -> %s | n=%d acc=%.4f f1=%.4f precision=%.4f recall=%.4f",
                checkpoint_path.name,
                year,
                row["n"],
                row["accuracy"],
                row["f1"],
                row["precision"],
                row["recall"],
            )

            if predictions is not None:
                write_prediction_csv(Path(args.prediction_dir) / f"{checkpoint_path.stem}_predictions_{year}.csv", predictions)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
