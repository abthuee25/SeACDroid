"""Recover readable API-call contexts from static-analysis text files."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

MEANINGLESS_APIS = {
    "Ljava/lang/Object;-><init>()V",
    "Ljava/lang/Object;->toString()Ljava/lang/String;",
    "Ljava/lang/Object;->equals(Ljava/lang/Object;)Z",
    "Ljava/lang/Object;->hashCode()I",
    "Ljava/lang/Object;->getClass()Ljava/lang/Class;",
}

_txt_parse_cache: dict[str, tuple[dict, dict, dict]] = {}
_official_packages_cache: dict[str, dict] = {}


def default_official_packages_file() -> Path:
    """Return the release-package official API package list."""
    return Path(__file__).resolve().parents[2] / "data" / "official_packages.txt"


def load_official_packages(pkg_file: str | Path | None = None) -> dict[str, list[str]]:
    """Load Android/Java package prefixes treated as official API namespaces."""
    path = Path(pkg_file) if pkg_file else default_official_packages_file()
    cache_key = str(path.resolve()) if path.exists() else str(path)
    if cache_key in _official_packages_cache:
        return _official_packages_cache[cache_key]

    packages: dict[str, list[str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                dalvik_pkg = "L" + line.replace(".", "/") + "/"
                top_level = dalvik_pkg.split("/")[0] + "/"
                packages.setdefault(top_level, []).append(dalvik_pkg)

    _official_packages_cache[cache_key] = packages
    return packages


def is_official_api(signature: str, official_packages: dict[str, list[str]]) -> bool:
    """Return whether a Dalvik signature belongs to an official API package."""
    if not signature:
        return False
    class_part = signature.split("->", 1)[0] if "->" in signature else signature
    if "/" not in class_part:
        return False
    top_level = class_part.split("/")[0] + "/"
    return any(class_part.startswith(pkg) for pkg in official_packages.get(top_level, []))


def is_meaningless_api(signature: str) -> bool:
    """Return whether a signature is ignored during context reconstruction."""
    return signature in MEANINGLESS_APIS


def parse_static_ir_with_lines(static_ir_file: str | Path, use_cache: bool = True) -> tuple[dict, dict, dict]:
    """Parse a static-analysis text file with method line ranges."""
    cache_key = str(Path(static_ir_file).resolve())
    if use_cache and cache_key in _txt_parse_cache:
        return _txt_parse_cache[cache_key]

    entries: dict[str, str] = {}
    method_bodies: dict[str, list[str]] = {}
    method_lines: dict[str, list[str]] = {}
    current_method = None
    current_apis: list[str] = []
    current_lines: list[str] = []

    with open(static_ir_file, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split("\t")
            record_type = parts[0]

            if record_type == "ENTRY" and len(parts) >= 3:
                entries[parts[1]] = parts[2]
            elif record_type == "METHOD_START" and len(parts) >= 2:
                current_method = parts[1]
                current_apis = []
                current_lines = [stripped]
            elif record_type == "API" and current_method and len(parts) >= 2:
                current_apis.append(parts[1])
                current_lines.append(stripped)
            elif record_type == "METHOD_END" and current_method:
                current_lines.append(stripped)
                method_bodies[current_method] = current_apis
                method_lines[current_method] = current_lines
                current_method = None
                current_apis = []
                current_lines = []

    result = (entries, method_bodies, method_lines)
    if use_cache:
        _txt_parse_cache[cache_key] = result
    return result


def build_reverse_call_graph(method_bodies: dict[str, list[str]]) -> dict[str, list[str]]:
    """Build callee -> callers edges among application-defined methods."""
    method_set = set(method_bodies)
    reverse = defaultdict(list)
    for method, calls in method_bodies.items():
        for callee in calls:
            if callee in method_set:
                reverse[callee].append(method)
    return reverse


def flatten_with_topology(
    method_signature: str,
    method_bodies: dict[str, list[str]],
    official_packages: dict[str, list[str]],
    visited: set[str] | None = None,
    max_depth: int = 5,
    current_penetration: int = 0,
) -> tuple[list[str], list[tuple[str, str]], dict[str, int]]:
    """Recursively flatten method calls into official API sequence and edges."""
    if visited is None:
        visited = set()

    if method_signature in visited or max_depth <= 0:
        return [], [], {}
    visited.add(method_signature)

    if method_signature not in method_bodies:
        if is_official_api(method_signature, official_packages):
            return [method_signature], [], {method_signature: current_penetration}
        return [], [], {}

    api_calls = method_bodies[method_signature]
    result_sequence: list[str] = []
    result_edges: list[tuple[str, str]] = []
    result_depths: dict[str, int] = {}
    last_official = None

    has_direct_official = any(
        is_official_api(api, official_packages) and not is_meaningless_api(api)
        for api in api_calls
    )

    for api in api_calls:
        if is_official_api(api, official_packages):
            if is_meaningless_api(api):
                continue

            result_sequence.append(api)
            if api not in result_depths or current_penetration < result_depths[api]:
                result_depths[api] = current_penetration
            if last_official:
                result_edges.append((last_official, api))
            last_official = api

        elif api in method_bodies:
            if not has_direct_official:
                continue

            sub_sequence, sub_edges, sub_depths = flatten_with_topology(
                api,
                method_bodies,
                official_packages,
                visited.copy(),
                max_depth - 1,
                current_penetration + 1,
            )
            if sub_sequence:
                if last_official:
                    result_edges.append((last_official, sub_sequence[0]))
                result_sequence.extend(sub_sequence)
                result_edges.extend(sub_edges)
                for node, depth in sub_depths.items():
                    if node not in result_depths or depth < result_depths[node]:
                        result_depths[node] = depth
                last_official = sub_sequence[-1]

    return result_sequence, result_edges, result_depths


def sequences_match(left: Iterable[str], right: Iterable[str], threshold: float = 0.7) -> bool:
    """Check whether two API-signature collections match by Jaccard overlap."""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return False
    union = left_set | right_set
    return bool(union) and len(left_set & right_set) / len(union) >= threshold


def reconstruct_subgraph_from_static_ir(
    static_ir_file: str | Path,
    anchor_api: str,
    target_signatures: list[str],
    official_packages: dict[str, list[str]],
) -> dict | None:
    """Re-run context extraction logic and match the serialized subgraph."""
    _, method_bodies, _ = parse_static_ir_with_lines(static_ir_file)
    if not method_bodies:
        return None

    reverse_call_graph = build_reverse_call_graph(method_bodies)
    target_set = set(target_signatures)
    direct_callers = [
        method
        for method, api_calls in method_bodies.items()
        if anchor_api in api_calls and "support/" not in method and "androidx/" not in method
    ]
    if not direct_callers:
        return None

    best_match = None
    best_score = 0.0

    for direct_caller in direct_callers:
        upstream_callers = reverse_call_graph.get(direct_caller, [None])
        upstream_callers = [
            upstream_caller
            for upstream_caller in upstream_callers
            if upstream_caller is None or ("support/" not in upstream_caller and "androidx/" not in upstream_caller)
        ]

        direct_sequence, _, _ = flatten_with_topology(
            direct_caller,
            method_bodies,
            official_packages,
            max_depth=999,
            current_penetration=0,
        )

        upstream_contexts = []
        combined_sequence = list(direct_sequence)
        for upstream_caller in upstream_callers:
            if not upstream_caller:
                continue
            upstream_sequence, _, _ = flatten_with_topology(
                upstream_caller,
                method_bodies,
                official_packages,
                max_depth=2,
                current_penetration=0,
            )
            if upstream_sequence:
                upstream_contexts.append({"upstream_caller": upstream_caller, "upstream_sequence": upstream_sequence})
                combined_sequence.extend(upstream_sequence)

        combined_unique = list(set(combined_sequence))
        if sequences_match(combined_unique, target_signatures, threshold=0.7):
            score = len(set(combined_unique) & target_set) / len(target_set) if target_set else 0.0
            if score > best_score:
                best_score = score
                best_match = {
                    "direct_caller": direct_caller,
                    "direct_sequence": direct_sequence,
                    "upstream_callers": upstream_contexts,
                    "combined_seq": combined_unique,
                    "match_score": score,
                }

    return best_match


def find_entry_points(
    method: str,
    method_bodies: dict[str, list[str]],
    entries: dict[str, str],
    max_depth: int = 10,
) -> list[tuple[str, str, int]]:
    """Trace all reachable upstream entry points for one method."""
    found: list[tuple[str, str, int]] = []
    visited = set()
    queue = deque([(method, 0)])

    while queue:
        current, depth = queue.popleft()
        if depth > max_depth or current in visited:
            continue
        visited.add(current)

        if current in entries:
            found.append((current, entries[current], depth))
            continue

        for caller, api_calls in method_bodies.items():
            if current in api_calls and caller not in visited:
                queue.append((caller, depth + 1))

    return found


def recover_context(static_ir_file: str | Path, subgraph: dict, pkg_file: str | Path | None = None) -> dict:
    """Recover readable method-level context for one selected subgraph."""
    anchor = subgraph.get("anchor_node") or subgraph.get("anchor") or ""
    signatures = list(subgraph.get("node_signatures") or subgraph.get("signatures") or [])
    if not anchor or not signatures:
        return {
            "anchor": anchor,
            "signatures": signatures,
            "api_sequence": signatures,
            "entry_points": [],
            "match_score": 0.0,
        }

    official_packages = load_official_packages(pkg_file)
    match = reconstruct_subgraph_from_static_ir(static_ir_file, anchor, signatures, official_packages)
    if not match:
        return {
            "anchor": anchor,
            "signatures": signatures,
            "api_sequence": signatures,
            "entry_points": [],
            "match_score": 0.0,
        }

    entries, method_bodies, method_lines = parse_static_ir_with_lines(static_ir_file)
    direct_caller = match["direct_caller"]
    upstream_callers = []
    for upstream_context in match["upstream_callers"]:
        upstream_caller = upstream_context["upstream_caller"]
        if upstream_caller in method_lines:
            upstream_callers.append(
                {
                    "method": upstream_caller,
                    "entry_type": entries.get(upstream_caller),
                    "lines": method_lines[upstream_caller],
                }
            )

    return {
        "anchor": anchor,
        "signatures": signatures,
        "nodes": signatures,
        "api_sequence": match["combined_seq"],
        "direct_caller": {
            "method": direct_caller,
            "entry_type": entries.get(direct_caller),
            "lines": method_lines.get(direct_caller, []),
        },
        "upstream_callers": upstream_callers,
        "combined_apis": match["combined_seq"],
        "entry_points": find_entry_points(direct_caller, method_bodies, entries, max_depth=10),
        "match_score": match["match_score"],
        "summary": {
            "num_upstream_callers": len(upstream_callers),
            "total_apis": len(match["combined_seq"]),
            "match_score": match["match_score"],
        },
    }


def recover_contexts(
    static_ir_file: str | Path,
    subgraphs: list[dict],
    pkg_file: str | Path | None = None,
) -> list[dict]:
    """Recover readable contexts for multiple selected subgraphs."""
    return [recover_context(static_ir_file, subgraph, pkg_file=pkg_file) for subgraph in subgraphs]
