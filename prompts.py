"""Prompt formatting for SeACDroid LLM-based explanations."""

from __future__ import annotations

from typing import Iterable


SYSTEM_PROMPT = (
    "You are an Android application security analyst. "
    "Analyze the app behavior from API call patterns and answer concisely."
)

NOISE_APIS = {
    "String.equals()",
    "String.startsWith()",
    "String.endsWith()",
    "String.contains()",
    "String.substring()",
    "String.length()",
    "String.toLowerCase()",
    "String.toUpperCase()",
    "String.trim()",
    "String.getBytes()",
    "String.valueOf()",
    "StringBuilder.<init>()",
    "StringBuilder.append()",
    "StringBuilder.toString()",
    "StringBuffer.<init>()",
    "StringBuffer.append()",
    "StringBuffer.toString()",
    "Log.d()",
    "Log.i()",
    "Log.v()",
    "Log.w()",
    "Integer.valueOf()",
    "Integer.intValue()",
    "Long.valueOf()",
    "Long.longValue()",
    "Boolean.valueOf()",
    "Boolean.booleanValue()",
    "Double.valueOf()",
    "Double.doubleValue()",
    "Double.parseDouble()",
    "Object.<init>()",
    "Object.toString()",
    "Object.equals()",
    "Object.hashCode()",
}


def simplify_api_signature(signature: str) -> str:
    """Convert a Dalvik-style method signature into Class.method()."""
    if "->" not in signature:
        return signature
    try:
        class_part, method_part = signature.split("->", 1)
        class_name = class_part.split("/")[-1].replace(";", "")
        method_name = method_part.split("(", 1)[0]
        return f"{class_name}.{method_name}()"
    except ValueError:
        return signature


def format_manifest(manifest: dict) -> str:
    """Format Manifest metadata for the user prompt."""
    lines = ["Application metadata:"]
    lines.append(f"  Package: {manifest.get('package', 'unknown')}")

    lines.append("  Permissions:")
    permissions = manifest.get("permissions", [])
    if permissions:
        for permission in permissions:
            lines.append(f"    - {permission}")
    else:
        lines.append("    - None")

    lines.append("  Components:")
    component_rows = []
    for key, title in [
        ("services", "Service"),
        ("receivers", "Receiver"),
        ("activities", "Activity"),
        ("providers", "Provider"),
    ]:
        for value in manifest.get(key, []) or []:
            component_rows.append(f"    {title}: {value}")

    if component_rows:
        lines.extend(component_rows)
    else:
        lines.append("    None")

    return "\n".join(lines)


def format_entry_points(context: dict) -> str:
    """Format entry points for the user prompt."""
    value = format_entry_point_value(context)
    return f"Entry point: {value}"


def format_entry_point_value(context: dict) -> str:
    """Return the display value for one context's entry points."""
    entry_points = context.get("entry_points", [])
    if not entry_points:
        return "N/A"
    values = []
    for entry in entry_points:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            values.append(f"{simplify_api_signature(str(entry[0]))} [{entry[1]}]")
        else:
            values.append(str(entry))
    return "; ".join(values)


def collect_api_calls(lines: Iterable[str]) -> list[str]:
    """Collect API calls from static-analysis text lines for prompt display."""
    api_calls = []
    for line in lines:
        if line.startswith("API\t"):
            signature = line.split("\t", 1)[1]
            simplified = simplify_api_signature(str(signature))
            if simplified not in NOISE_APIS:
                api_calls.append(simplified)
    return api_calls


def fallback_call_chain(context: dict) -> list[str]:
    """Collect API calls when method-level context is unavailable."""
    api_calls = []
    for signature in context.get("api_sequence") or context.get("signatures") or []:
        simplified = simplify_api_signature(str(signature))
        if simplified not in NOISE_APIS:
            api_calls.append(simplified)
    return api_calls or ["N/A"]


def build_local_calling_context(context: dict) -> dict:
    """Build the local calling context shown to the LLM."""
    upstream_callers = []
    for upstream in context.get("upstream_callers", []):
        upstream_callers.append(
            {
                "method": simplify_api_signature(str(upstream.get("method", "N/A"))),
                "api_calls": collect_api_calls(upstream.get("lines", [])) or ["N/A"],
            }
        )

    direct = context.get("direct_caller") or {}
    direct_method = direct.get("method")
    if direct_method:
        direct_caller = {
            "method": simplify_api_signature(str(direct_method)),
            "api_calls": collect_api_calls(direct.get("lines", [])) or fallback_call_chain(context),
        }
    else:
        direct_caller = {
            "method": "N/A",
            "api_calls": fallback_call_chain(context),
        }

    return {"upstream_callers": upstream_callers, "direct_caller": direct_caller}


def format_context(context: dict, rank: int, attention_weight: float | None = None) -> str:
    """Format one selected context using the explanation prompt structure."""
    lines = [f"=== Call pattern {rank} ==="]
    lines.append(format_entry_points(context))
    lines.append("Local calling context:")

    local_context = build_local_calling_context(context)
    for idx, upstream in enumerate(local_context["upstream_callers"], 1):
        lines.append(f"Upstream caller {idx}: {upstream['method']}")
        for api in upstream["api_calls"]:
            lines.append(f"-> {api}")

    direct_caller = local_context["direct_caller"]
    lines.append(f"Direct caller: {direct_caller['method']}")
    for api in direct_caller["api_calls"]:
        lines.append(f"-> {api}")

    return "\n".join(lines)


def build_call_patterns(contexts: Iterable[dict]) -> list[dict]:
    """Build the report-facing call-pattern structure shown to the LLM."""
    patterns = []
    for idx, context in enumerate(contexts, 1):
        patterns.append(
            {
                "rank": idx,
                "entry_point": format_entry_point_value(context),
                "local_calling_context": build_local_calling_context(context),
            }
        )
    return patterns


def build_explanation_prompt(
    manifest: dict,
    contexts: Iterable[dict],
    attention_weights: Iterable[float] | None = None,
) -> str:
    """Build the user prompt for LLM-based explanation."""
    lines = [format_manifest(manifest), ""]
    lines.append("Call patterns ranked by importance:")
    for idx, context in enumerate(contexts, 1):
        lines.append(format_context(context, idx))
        lines.append("")

    lines.append("Analysis request:")
    lines.append("1. Function summary")
    lines.append("2. Suspicious behavior")
    lines.append("3. Risk level and reason")
    lines.append("4. Overall assessment")
    return "\n".join(lines)


def build_prompt_preview(user_input: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Render system and user messages for prompt-output inspection."""
    return f"[System prompt]\n{system_prompt}\n\n[User input]\n{user_input}"
