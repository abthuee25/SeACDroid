"""Explain one APK from SeACDroid high-attention API contexts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch_geometric.data import Batch

from seacdroid.data import graph_from_subgraph, load_pickle
from seacdroid.evaluation.evaluate_detector import load_model
from seacdroid.llm.context import recover_contexts
from seacdroid.llm.manifest import infer_manifest_json, load_manifest
from seacdroid.llm.prompts import SYSTEM_PROMPT, build_call_patterns, build_explanation_prompt, build_prompt_preview


def feature_stem(path: Path) -> str:
    """Return the sample stem for .pkl or .pkl.gz."""
    name = path.name
    if name.endswith(".pkl.gz"):
        return name[:-7]
    if name.endswith(".pkl"):
        return name[:-4]
    return path.stem


def infer_static_ir_file(feature_file: Path, feature_root: str | None, static_ir_root: str | None) -> Path | None:
    """Infer the matching static-analysis text path from release directory layout."""
    if not feature_root or not static_ir_root:
        return None
    try:
        relative = feature_file.resolve().relative_to(Path(feature_root).resolve())
    except ValueError:
        return None
    return Path(static_ir_root) / relative.parent / f"{feature_stem(feature_file)}.txt"


def build_valid_graphs(raw_subgraphs: list[dict], max_nodes: int = 0, undirected: bool = True):
    """Convert raw serialized subgraphs into PyG graphs while keeping raw indices."""
    graphs = []
    raw_indices = []
    for raw_idx, subgraph in enumerate(raw_subgraphs):
        graph = graph_from_subgraph(subgraph, max_nodes=max_nodes, undirected=undirected)
        if graph is not None:
            graphs.append(graph)
            raw_indices.append(raw_idx)
    return graphs, raw_indices


def select_top_contexts(
    checkpoint: str | Path,
    feature_file: str | Path,
    top_k: int,
    device: torch.device,
    max_nodes_override: int | None = None,
    directed_override: bool = False,
) -> tuple[list[dict], list[float], dict]:
    """Run one APK through the detector and select top-attention raw subgraphs."""
    model, checkpoint_data = load_model(Path(checkpoint), device)
    model.eval()

    max_nodes = checkpoint_data.get("max_nodes", 0) if max_nodes_override is None else max_nodes_override
    undirected = bool(checkpoint_data.get("undirected", True) and not directed_override)

    payload = load_pickle(Path(feature_file))
    raw_subgraphs = payload.get("subgraphs", []) if isinstance(payload, dict) else []
    graphs, raw_indices = build_valid_graphs(raw_subgraphs, max_nodes=max_nodes, undirected=undirected)
    if not graphs:
        raise ValueError(f"No valid context subgraphs found in {feature_file}")

    batch = Batch.from_data_list(graphs).to(device)
    graph_to_apk = torch.zeros(len(graphs), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, attention = model(batch.x, batch.edge_index, batch.batch, graph_to_apk)
        probabilities = torch.softmax(logits, dim=1)[0]
        weights = attention.squeeze(-1).cpu().tolist()

    ranked_graph_indices = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)[:top_k]
    selected_raw_indices = [raw_indices[idx] for idx in ranked_graph_indices]
    selected_subgraphs = [raw_subgraphs[idx] for idx in selected_raw_indices]
    selected_weights = [float(weights[idx]) for idx in ranked_graph_indices]
    detector_output = {
        "prediction": int(torch.argmax(probabilities).cpu()),
        "prob_malware": float(probabilities[1].cpu()),
        "num_contexts": len(raw_subgraphs),
        "num_valid_contexts": len(graphs),
        "selected_indices": selected_raw_indices,
    }
    return selected_subgraphs, selected_weights, detector_output


def call_deepseek(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Call the DeepSeek OpenAI-compatible chat API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one APK with SeACDroid-selected API contexts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_file", required=True, help="Path to one APK feature .pkl or .pkl.gz file")
    parser.add_argument("--static_ir_file", default=None, help="Matching static-analysis text file")
    parser.add_argument("--feature_root", default=None, help="Feature root used to infer --static_ir_file")
    parser.add_argument("--static_ir_root", default=None, help="Static-analysis text root used to infer --static_ir_file")
    parser.add_argument("--manifest_json", default=None, help="Manifest metadata JSON")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output", default="explanation_report.json")
    parser.add_argument("--prompt_output", default=None)
    parser.add_argument("--dry_run", action="store_true", help="Generate prompt and report without calling the LLM")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api_key", default=None, help="API key. Prefer --api_key_env for shared runs.")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--directed", action="store_true", help="Use directed graph edges instead of the default checkpoint setting")
    args = parser.parse_args()

    feature_file = Path(args.feature_file)
    static_ir_file = Path(args.static_ir_file) if args.static_ir_file else infer_static_ir_file(feature_file, args.feature_root, args.static_ir_root)
    if static_ir_file is None or not static_ir_file.exists():
        raise FileNotFoundError("Static-analysis text file is required. Pass --static_ir_file or --feature_root with --static_ir_root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_subgraphs, selected_weights, detector_output = select_top_contexts(
        args.checkpoint,
        feature_file,
        args.top_k,
        device,
        max_nodes_override=args.max_nodes,
        directed_override=args.directed,
    )
    contexts = recover_contexts(static_ir_file, selected_subgraphs)
    manifest_json = Path(args.manifest_json) if args.manifest_json else infer_manifest_json(static_ir_file)
    manifest = load_manifest(manifest_json)
    prompt = build_explanation_prompt(manifest, contexts, selected_weights)
    prompt_preview = build_prompt_preview(prompt)

    if args.prompt_output:
        prompt_path = Path(args.prompt_output)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_preview, encoding="utf-8")

    llm_response = None
    if not args.dry_run:
        api_key = args.api_key or os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing DeepSeek API key. Set {args.api_key_env}, pass --api_key, or use --dry_run.")
        llm_response = call_deepseek(prompt, api_key, args.base_url, args.model, args.temperature, args.max_tokens)

    report = {
        "sample": {
            "feature_file": str(feature_file),
            "static_ir_file": str(static_ir_file),
            "manifest_json": str(manifest_json),
            "sha256": feature_stem(feature_file).upper(),
        },
        "detector": detector_output,
        "manifest": manifest,
        "call_patterns": build_call_patterns(contexts),
        "attention_weights": selected_weights,
        "system_prompt": SYSTEM_PROMPT,
        "llm_user_input": prompt,
        "llm_prompt": prompt_preview,
        "llm_response": llm_response,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
