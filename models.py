"""SeACDroid GAT encoder and gated-attention MIL classifier."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GATContextEncoder(nn.Module):
    """Encode API context subgraphs with a GAT and global mean pooling."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        context_dim: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        encoder_type: str = "gat",
        pool_type: str = "mean",
    ) -> None:
        super().__init__()
        if encoder_type != "gat":
            raise ValueError("SeACDroid supports encoder_type='gat'.")
        if pool_type != "mean":
            raise ValueError("SeACDroid supports pool_type='mean'.")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        self.dropout = dropout
        self.pool_type = pool_type
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.fc = nn.Linear(hidden_dim, context_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(h, edge_index))), p=self.dropout, training=self.training)
        pooled = global_mean_pool(h, batch)
        return self.fc(pooled)


class GatedAttentionMIL(nn.Module):
    """Aggregate context embeddings into an APK-level malware prediction."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        context_dim: int = 64,
        mil_hidden_dim: int = 128,
        classifier_hidden_dim: int = 32,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        encoder_type: str = "gat",
        pool_type: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = GATContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            encoder_type=encoder_type,
            pool_type=pool_type,
        )
        self.attn_V = nn.Sequential(nn.Linear(context_dim, mil_hidden_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(context_dim, mil_hidden_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(mil_hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        graph_to_apk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_embeddings = self.encoder(x, edge_index, batch)
        scores = self.attn_w(self.attn_V(context_embeddings) * self.attn_U(context_embeddings))

        unique_apks = torch.unique(graph_to_apk[graph_to_apk >= 0], sorted=True)
        apk_embeddings = []
        attention_weights = torch.zeros_like(scores)
        for apk_id in unique_apks:
            mask = graph_to_apk == apk_id
            weights = F.softmax(scores[mask], dim=0)
            attention_weights[mask] = weights
            apk_embeddings.append((weights * context_embeddings[mask]).sum(dim=0))

        if not apk_embeddings:
            return torch.empty(0, 2, device=x.device), unique_apks, attention_weights

        logits = self.classifier(torch.stack(apk_embeddings))
        return logits, unique_apks, attention_weights


AttentionMIL = GatedAttentionMIL


def default_model_config() -> dict:
    """Return the default SeACDroid model configuration."""
    return {
        "input_dim": 384,
        "hidden_dim": 128,
        "context_dim": 64,
        "mil_hidden_dim": 128,
        "classifier_hidden_dim": 32,
        "num_layers": 3,
        "heads": 4,
        "dropout": 0.3,
        "encoder_type": "gat",
        "pool_type": "mean",
    }
