#!/usr/bin/env python3
"""
Extract static-analysis artifacts from an APK using Androguard.

Static-analysis text format:
    # ENTRY POINTS
    ENTRY	<method_sig>	<entry_type>
    ...
    # METHOD BODIES
    METHOD_START	<app_method_sig>
    API	<api_sig>
    API	<api_sig>
    ...
    METHOD_END

The extractor also writes a Manifest metadata sidecar JSON next to the text
file by default, using the same stem and a .manifest.json suffix.
"""

import os
import sys
import time
import argparse
import zipfile
import logging
import json
from pathlib import Path

# Suppress Androguard logs
logging.getLogger("androguard").setLevel(logging.ERROR)
try:
    from loguru import logger
    logger.disable("androguard")
except ImportError:
    pass

from androguard.misc import AnalyzeAPK


# ============================================================================
# Constants
# ============================================================================

# FlowDroid callback interfaces (loaded from file)
CALLBACK_INTERFACES = set()

# Android component base classes
COMPONENT_CLASSES = {
    "Landroid/app/Activity;",
    "Landroid/app/Service;",
    "Landroid/content/BroadcastReceiver;",
    "Landroid/content/ContentProvider;",
    "Landroid/app/Application;",
    "Landroid/app/Fragment;",
    "Landroid/support/v4/app/Fragment;",
    "Landroidx/fragment/app/Fragment;",
    "Landroid/os/AsyncTask;",
    "Landroid/webkit/WebViewClient;",
    "Landroid/webkit/WebChromeClient;",
}

# Lifecycle methods
LIFECYCLE_METHODS = {
    "onCreate", "onStart", "onResume", "onPause", "onStop", "onDestroy", "onRestart",
    "onActivityResult", "onNewIntent", "onSaveInstanceState", "onRestoreInstanceState",
    "onStartCommand", "onBind", "onUnbind", "onRebind", "onReceive",
    "query", "insert", "update", "delete", "getType", "call",
    "onAttach", "onCreateView", "onViewCreated", "onDestroyView", "onDetach",
    "doInBackground", "onPreExecute", "onPostExecute", "onProgressUpdate",
    "shouldOverrideUrlLoading", "onPageStarted", "onPageFinished", "onReceivedError",
}

# Callback method patterns
CALLBACK_METHOD_PATTERNS = {"run", "call", "handleMessage", "onClick", "onTouch", "onLongClick"}

# Exclude support/androidx from entry points
EXCLUDE_PREFIXES = ("Landroid/support/", "Landroidx/")


# ============================================================================
# Helper Functions
# ============================================================================

def load_callbacks(filepath: str) -> int:
    """Load callback interfaces from file."""
    global CALLBACK_INTERFACES
    if not os.path.exists(filepath):
        return 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                dalvik_name = 'L' + line.replace('.', '/') + ';'
                CALLBACK_INTERFACES.add(dalvik_name)
    
    return len(CALLBACK_INTERFACES)


def format_signature(method) -> str:
    """Format method signature in Dalvik format."""
    class_name = method.get_class_name()
    method_name = method.get_name()
    descriptor = method.get_descriptor()
    return f"{class_name}->{method_name}{descriptor}"


def get_superclasses(class_analysis, dx, visited=None):
    """Get all superclasses of a class."""
    if visited is None:
        visited = set()
    result = []
    
    try:
        class_def = class_analysis.get_vm_class()
        if class_def is None:
            return result
        
        superclass_name = class_def.get_superclassname()
        if superclass_name and superclass_name not in visited and superclass_name != 'Ljava/lang/Object;':
            visited.add(superclass_name)
            result.append(superclass_name)
            super_analysis = dx.get_class_analysis(superclass_name)
            if super_analysis:
                result.extend(get_superclasses(super_analysis, dx, visited))
    except:
        pass
    
    return result


def get_interfaces(class_analysis, dx, visited=None):
    """Get all interfaces implemented by a class."""
    if visited is None:
        visited = set()
    result = []
    
    try:
        class_def = class_analysis.get_vm_class()
        if class_def is None:
            return result
        
        interfaces = class_def.get_interfaces()
        if interfaces:
            for iface in interfaces:
                if iface not in visited:
                    visited.add(iface)
                    result.append(iface)
                    iface_analysis = dx.get_class_analysis(iface)
                    if iface_analysis:
                        result.extend(get_interfaces(iface_analysis, dx, visited))
        
        superclass_name = class_def.get_superclassname()
        if superclass_name and superclass_name != 'Ljava/lang/Object;':
            super_analysis = dx.get_class_analysis(superclass_name)
            if super_analysis:
                result.extend(get_interfaces(super_analysis, dx, visited))
    except:
        pass
    
    return result


def implements_callback_interface(class_analysis, dx):
    """Check if class implements a callback interface."""
    try:
        class_name = class_analysis.name
        if class_name in CALLBACK_INTERFACES:
            return class_name
        
        interfaces = get_interfaces(class_analysis, dx)
        for iface in interfaces:
            if iface in CALLBACK_INTERFACES:
                return iface
        
        superclasses = get_superclasses(class_analysis, dx)
        for superclass in superclasses:
            if superclass in CALLBACK_INTERFACES:
                return superclass
    except:
        pass
    
    return None


def is_android_component(class_analysis, dx):
    """Check if class is an Android component."""
    try:
        class_name = class_analysis.name
        if class_name in COMPONENT_CLASSES:
            return class_name
        
        superclasses = get_superclasses(class_analysis, dx)
        for superclass in superclasses:
            if superclass in COMPONENT_CLASSES:
                return superclass
    except:
        pass
    
    return None


def is_entry_point(method, class_analysis, dx):
    """
    Determine if method is an entry point.
    
    Returns:
        Entry type string or None
    """
    method_name = method.get_name()
    class_name = method.get_class_name()
    
    # Exclude support/androidx
    for prefix in EXCLUDE_PREFIXES:
        if class_name.startswith(prefix):
            return None
    
    # Static initializer
    if method_name == "<clinit>":
        return "StaticInit"
    
    # Runnable.run()
    if method_name == "run":
        interfaces = get_interfaces(class_analysis, dx)
        if "Ljava/lang/Runnable;" in interfaces:
            return "Runnable"
    
    # Callable.call()
    if method_name == "call":
        interfaces = get_interfaces(class_analysis, dx)
        if "Ljava/util/concurrent/Callable;" in interfaces:
            return "Callable"
    
    # Callback interface methods
    callback_interface = implements_callback_interface(class_analysis, dx)
    if callback_interface:
        if method_name.startswith("on") or method_name in CALLBACK_METHOD_PATTERNS:
            return "Callback"
    
    # Android component lifecycle methods
    component_class = is_android_component(class_analysis, dx)
    if component_class and method_name in LIFECYCLE_METHODS:
        return "Lifecycle"
    
    return None


def extract_manifest_metadata(apk) -> dict:
    """Extract Manifest-derived metadata from an Androguard APK object."""
    def sorted_values(getter_name: str) -> list[str]:
        try:
            values = getattr(apk, getter_name)() or []
        except Exception:
            return []
        return sorted(set(str(value) for value in values if value))

    try:
        package = apk.get_package() or "unknown"
    except Exception:
        package = "unknown"

    try:
        target_sdk = apk.get_target_sdk_version()
    except Exception:
        target_sdk = None

    return {
        "package": package,
        "permissions": sorted_values("get_permissions"),
        "services": sorted_values("get_services"),
        "receivers": sorted_values("get_receivers"),
        "activities": sorted_values("get_activities"),
        "providers": sorted_values("get_providers"),
        "target_sdk": target_sdk,
    }


def default_manifest_output(output_file: str) -> str:
    """Return the default Manifest sidecar path for one static-analysis text file."""
    return str(Path(output_file).with_suffix(".manifest.json"))


# ============================================================================
# Main Extraction
# ============================================================================

def extract_call_graph(apk_path: str, output_file: str, callbacks_file: str = None, manifest_output_file: str = None) -> dict:
    """
    Extract call graph from APK.
    
    Args:
        apk_path: Path to APK file
        output_file: Output txt file path
        callbacks_file: Path to AndroidCallbacks.txt
        manifest_output_file: Optional Manifest metadata JSON path
    
    Returns:
        Result dict with status and statistics
    """
    if not os.path.exists(apk_path):
        return {"status": "not_found"}
    
    if not zipfile.is_zipfile(apk_path):
        return {"status": "invalid_apk"}
    
    # Load callbacks if not already loaded
    if callbacks_file and len(CALLBACK_INTERFACES) == 0:
        load_callbacks(callbacks_file)
    
    try:
        # Parse APK
        try:
            a, d, dx = AnalyzeAPK(apk_path)
        except Exception as parse_err:
            return {
                "status": "failed",
                "error": f"APK parse error: {type(parse_err).__name__}: {str(parse_err)[:100]}"
            }
        
        manifest = extract_manifest_metadata(a)
        entries = []
        method_bodies = []  # [(app_method, [api1, api2, ...])]
        
        for method_analysis in dx.get_methods():
            method = method_analysis.get_method()
            
            # Skip external methods
            if method_analysis.is_external():
                continue
            
            src_sig = format_signature(method)
            class_name = method.get_class_name()
            
            # Check if entry point
            class_analysis = dx.get_class_analysis(class_name)
            if class_analysis:
                entry_reason = is_entry_point(method, class_analysis, dx)
                if entry_reason:
                    entries.append((src_sig, entry_reason))
            
            # Extract API call sequence (preserve order, no dedup)
            api_sequence = []
            try:
                xrefs = list(method_analysis.get_xref_to())
                xrefs_sorted = sorted(xrefs, key=lambda x: x[2] if x[2] is not None else 0)
                
                for _, callee, offset in xrefs_sorted:
                    callee_method = callee.get_method()
                    callee_sig = format_signature(callee_method)
                    api_sequence.append(callee_sig)
            except:
                pass
            
            if api_sequence:
                method_bodies.append((src_sig, api_sequence))
        
        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            # Entry points
            f.write("# ENTRY POINTS\n")
            for sig, reason in entries:
                f.write(f"ENTRY\t{sig}\t{reason}\n")
            
            # Method bodies
            f.write("# METHOD BODIES\n")
            for app_method, api_seq in method_bodies:
                f.write(f"METHOD_START\t{app_method}\n")
                for api in api_seq:
                    f.write(f"API\t{api}\n")
                f.write("METHOD_END\n")

        manifest_path = manifest_output_file or default_manifest_output(output_file)
        Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "manifest_output": manifest_path,
            "package": manifest["package"],
            "permissions": len(manifest["permissions"]),
            "components": (
                len(manifest["activities"])
                + len(manifest["services"])
                + len(manifest["receivers"])
                + len(manifest["providers"])
            ),
            "entries": len(entries),
            "methods": len(method_bodies),
            "total_apis": sum(len(seq) for _, seq in method_bodies)
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description='Extract call graph from APK')
    parser.add_argument('apk_path', help='Path to APK file')
    parser.add_argument('output_file', help='Output static-analysis text file path')
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_callbacks = os.path.join(package_root, 'data', 'callbacks', 'AndroidCallbacks.txt')
    parser.add_argument('-c', '--callbacks', default=default_callbacks,
                        help='Path to AndroidCallbacks.txt')
    parser.add_argument('--manifest_output', default=None,
                        help='Output Manifest metadata JSON path. Defaults to <output>.manifest.json')
    args = parser.parse_args()
    
    print(f"APK: {args.apk_path}")
    print(f"Output: {args.output_file}")
    
    if args.callbacks:
        n = load_callbacks(args.callbacks)
        print(f"Loaded {n} callbacks")
    
    start = time.time()
    result = extract_call_graph(args.apk_path, args.output_file, args.callbacks, args.manifest_output)
    elapsed = time.time() - start
    
    print(f"\nResult: {result}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
