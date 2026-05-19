"""Dataset loading and batching utilities for SeACDroid."""

from __future__ import annotations

import gzip
import logging
import pickle
from pathlib import Path
from typing import Iterable

import torch
from torch_geometric.data import Batch, Data

LOGGER = logging.getLogger(__name__)

SEMANTIC_DIM = 384


def load_pickle(path: Path):
    """Load a pickle or gzip-compressed pickle file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def sha_from_feature_path(path: Path) -> str:
    """Return an uppercase SHA256 stem for .pkl or .pkl.gz feature files."""
    name = path.name
    if name.endswith(".pkl.gz"):
        name = name[:-7]
    elif name.endswith(".pkl"):
        name = name[:-4]
    return name.upper()


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Convert a PyG edge_index tensor to an undirected edge list."""
    if edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def feature_files(root: Path) -> list[Path]:
    """List supported feature files in a class directory."""
    if not root.exists():
        return []
    return sorted(list(root.glob("*.pkl")) + list(root.glob("*.pkl.gz")))


def graph_from_subgraph(subgraph: dict, max_nodes: int = 0, undirected: bool = True) -> Data | None:
    """Convert one serialized context subgraph into a PyG Data object."""
    node_features = subgraph.get("node_features")
    if node_features is None or len(node_features) == 0:
        return None

    x = torch.as_tensor(node_features, dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return None
    if max_nodes > 0 and x.shape[0] > max_nodes:
        return None

    edge_index = subgraph.get("edge_index")
    if edge_index is None:
        edge_index = torch.stack([torch.arange(x.shape[0]), torch.arange(x.shape[0])], dim=0)
    else:
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = torch.stack([torch.arange(x.shape[0]), torch.arange(x.shape[0])], dim=0)
        elif edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

    if undirected:
        edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index.contiguous())
    data.anchor = subgraph.get("anchor", "")
    data.entry_type = subgraph.get("entry_type", "Unknown")
    data.signatures = subgraph.get("signatures", [])
    return data


def load_feature_file(path: Path, label: int, max_nodes: int = 0, undirected: bool = True) -> dict | None:
    """Load one APK feature file into an APK bag."""
    try:
        payload = load_pickle(path)
    except Exception as exc:
        LOGGER.warning("Failed to load %s: %s", path, exc)
        return None

    subgraphs = payload.get("subgraphs", []) if isinstance(payload, dict) else []
    graphs = []
    for subgraph in subgraphs:
        graph = graph_from_subgraph(subgraph, max_nodes=max_nodes, undirected=undirected)
        if graph is not None:
            graphs.append(graph)

    return {
        "sha256": sha_from_feature_path(path),
        "path": str(path),
        "graphs": graphs,
        "label": int(label),
        "is_empty": len(graphs) == 0,
    }


def load_dataset(
    data_dir: str | Path,
    years: Iterable[str],
    max_nodes: int = 0,
    undirected: bool = True,
) -> list[dict]:
    """Load feature files arranged as data_dir/<dataset>/{benign,malware}/*.pkl."""
    data_root = Path(data_dir)
    samples: list[dict] = []

    for year in [str(y) for y in years]:
        for label, class_name in [(0, "benign"), (1, "malware")]:
            class_dir = data_root / year / class_name
            files = feature_files(class_dir)

            LOGGER.info("%s/%s: %d feature files", year, class_name, len(files))
            for path in files:
                sample = load_feature_file(path, label, max_nodes=max_nodes, undirected=undirected)
                if sample is not None:
                    sample["year"] = year
                    samples.append(sample)

    LOGGER.info("Loaded %d APK samples", len(samples))
    return samples


def collate_apk_batch(apk_batch: list[dict]) -> dict:
    """Collate a batch of APK bags into one PyG graph batch plus graph-to-APK ids."""
    graphs: list[Data] = []
    graph_to_apk: list[int] = []
    labels: list[int] = []
    metadata: list[dict] = []
    empty_labels: list[int] = []
    empty_metadata: list[dict] = []

    non_empty_idx = 0
    for apk in apk_batch:
        if apk.get("is_empty") or not apk.get("graphs"):
            empty_labels.append(int(apk["label"]))
            empty_metadata.append({k: apk.get(k) for k in ("sha256", "path", "year")})
            continue

        labels.append(int(apk["label"]))
        metadata.append({k: apk.get(k) for k in ("sha256", "path", "year")})
        for graph in apk["graphs"]:
            graphs.append(graph)
            graph_to_apk.append(non_empty_idx)
        non_empty_idx += 1

    if not graphs:
        graphs = [Data(x=torch.zeros(1, SEMANTIC_DIM), edge_index=torch.tensor([[0], [0]], dtype=torch.long))]
        graph_to_apk = [-1]

    return {
        "batch": Batch.from_data_list(graphs),
        "graph_to_apk": torch.tensor(graph_to_apk, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "metadata": metadata,
        "empty_labels": torch.tensor(empty_labels, dtype=torch.long) if empty_labels else None,
        "empty_metadata": empty_metadata,
    }
