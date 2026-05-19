"""Batch LLM explanation over SeACDroid feature files."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from seacdroid.data import load_pickle
from seacdroid.evaluation.evaluate_detector import load_model
from seacdroid.llm.context import recover_contexts
from seacdroid.llm.explain_sample import build_valid_graphs, call_deepseek, feature_stem
from seacdroid.llm.manifest import infer_manifest_json, load_manifest
from seacdroid.llm.prompts import SYSTEM_PROMPT, build_call_patterns, build_explanation_prompt


def parse_decision(text: str | None) -> str:
    """Parse the final LLM decision when present."""
    if not text:
        return "unknown"
    lowered = text.lower()
    if "decision: malware" in lowered or "decision=malware" in lowered or "label=1" in lowered:
        return "malware"
    if "decision: benign" in lowered or "decision=benign" in lowered or "label=0" in lowered:
        return "benign"
    if "malware" in lowered or "malicious" in lowered:
        return "malware"
    if "benign" in lowered:
        return "benign"
    return "unknown"


def collect_feature_files(feature_dir: Path, limit: int = 0) -> list[Path]:
    """Collect feature files from benign and malware class directories."""
    files: list[Path] = []
    for class_name in ("benign", "malware"):
        class_dir = feature_dir / class_name
        files.extend(sorted(class_dir.glob("*.pkl")))
        files.extend(sorted(class_dir.glob("*.pkl.gz")))
    files = sorted(files)
    if limit > 0:
        return files[:limit]
    return files


def infer_label(feature_file: Path) -> str:
    """Infer class label from the parent directory name."""
    parent = feature_file.parent.name.lower()
    if parent in {"benign", "malware"}:
        return parent
    return "unknown"


def matching_static_ir(feature_file: Path, feature_dir: Path, static_ir_dir: Path) -> Path:
    """Map feature_dir/.../<sha>.pkl to static_ir_dir/.../<sha>.txt."""
    relative = feature_file.resolve().relative_to(feature_dir.resolve())
    return static_ir_dir / relative.parent / f"{feature_stem(feature_file)}.txt"


def prepare_task(
    model,
    checkpoint_data: dict,
    feature_file: Path,
    feature_dir: Path,
    static_ir_dir: Path,
    top_k: int,
    device: torch.device,
    max_nodes_override: int | None,
    directed_override: bool,
    only_predicted_malware: bool,
) -> dict | None:
    """Run GNN attention selection and build one LLM prompt."""
    static_ir_file = matching_static_ir(feature_file, feature_dir, static_ir_dir)
    if not static_ir_file.exists():
        return {"feature_file": str(feature_file), "error": f"missing static-analysis text file: {static_ir_file}"}

    max_nodes = checkpoint_data.get("max_nodes", 0) if max_nodes_override is None else max_nodes_override
    undirected = bool(checkpoint_data.get("undirected", True) and not directed_override)
    payload = load_pickle(feature_file)
    raw_subgraphs = payload.get("subgraphs", []) if isinstance(payload, dict) else []
    graphs, raw_indices = build_valid_graphs(raw_subgraphs, max_nodes=max_nodes, undirected=undirected)
    if not graphs:
        return {"feature_file": str(feature_file), "error": "no valid context subgraphs"}

    batch = Batch.from_data_list(graphs).to(device)
    graph_to_apk = torch.zeros(len(graphs), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, attention = model(batch.x, batch.edge_index, batch.batch, graph_to_apk)
        probabilities = torch.softmax(logits, dim=1)[0]
        prediction = int(torch.argmax(probabilities).cpu())
        prob_malware = float(probabilities[1].cpu())
        weights = attention.squeeze(-1).cpu().tolist()

    if only_predicted_malware and prediction != 1:
        return None

    ranked_graph_indices = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)[:top_k]
    selected_raw_indices = [raw_indices[idx] for idx in ranked_graph_indices]
    selected_subgraphs = [raw_subgraphs[idx] for idx in selected_raw_indices]
    selected_weights = [float(weights[idx]) for idx in ranked_graph_indices]
    contexts = recover_contexts(static_ir_file, selected_subgraphs)

    sha256 = feature_stem(feature_file).upper()
    label = infer_label(feature_file)
    manifest_json = infer_manifest_json(static_ir_file)
    if manifest_json is None:
        expected = static_ir_file.with_suffix(".manifest.json")
        return {"feature_file": str(feature_file), "error": f"missing Manifest metadata sidecar: {expected}"}
    manifest = load_manifest(manifest_json)
    prompt = build_explanation_prompt(manifest, contexts, selected_weights)

    return {
        "sample": sha256,
        "feature_file": str(feature_file),
        "static_ir_file": str(static_ir_file),
        "manifest_json": str(manifest_json),
        "true_label": label,
        "detector_prediction": "malware" if prediction == 1 else "benign",
        "detector_prob_malware": prob_malware,
        "selected_indices": selected_raw_indices,
        "attention_weights": selected_weights,
        "manifest": manifest,
        "call_patterns": build_call_patterns(contexts),
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch explain APKs with SeACDroid-selected API contexts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True, help="Directory containing benign/ and malware/ feature files")
    parser.add_argument("--static_ir_dir", required=True, help="Directory containing matching benign/ and malware/ text files")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only_predicted_malware", action="store_true")
    parser.add_argument("--output", default="llm_batch_results.json")
    parser.add_argument("--dry_run", action="store_true", help="Generate prompts without calling the LLM")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api_key", default=None, help="API key. Prefer --api_key_env for shared runs.")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=800)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--directed", action="store_true", help="Use directed graph edges instead of the default checkpoint setting")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    static_ir_dir = Path(args.static_ir_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint_data = load_model(Path(args.checkpoint), device)
    model.eval()

    tasks = []
    errors = []
    for feature_file in tqdm(collect_feature_files(feature_dir, args.limit), desc="Preparing prompts"):
        task = prepare_task(
            model,
            checkpoint_data,
            feature_file,
            feature_dir,
            static_ir_dir,
            args.top_k,
            device,
            args.max_nodes,
            args.directed,
            args.only_predicted_malware,
        )
        if task is None:
            continue
        if "error" in task:
            errors.append(task)
        else:
            tasks.append(task)

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not args.dry_run and not api_key:
        raise RuntimeError(f"Missing DeepSeek API key. Set {args.api_key_env}, pass --api_key, or use --dry_run.")

    results = []
    if args.dry_run:
        results = [{**task, "llm_response": None, "llm_prediction": "unknown"} for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(call_deepseek, task["prompt"], api_key, args.base_url, args.model, args.temperature, args.max_tokens): task
                for task in tasks
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calling LLM"):
                task = futures[future]
                try:
                    response = future.result()
                    task["llm_response"] = response
                    task["llm_prediction"] = parse_decision(response)
                except Exception as exc:
                    task["llm_response"] = None
                    task["llm_prediction"] = "unknown"
                    task["error"] = str(exc)
                results.append(task)

    report = {
        "checkpoint": args.checkpoint,
        "system_prompt": SYSTEM_PROMPT,
        "feature_dir": str(feature_dir),
        "static_ir_dir": str(static_ir_dir),
        "top_k": args.top_k,
        "dry_run": args.dry_run,
        "num_prepared": len(tasks),
        "num_errors": len(errors),
        "errors": errors,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
