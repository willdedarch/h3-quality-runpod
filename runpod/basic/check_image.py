"""Verify native node registration and graph schemas in the built image, without a GPU."""
import importlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
COMFY = Path(os.environ.get("H3_BASIC_COMFY", "/opt/ComfyUI"))
sys.path.insert(0, str(COMFY))
sys.argv = ["check_image", "--cpu", "--disable-all-custom-nodes", "--disable-partner-nodes"]
import main as comfy_main
import nodes
from comfy_api.latest._io import ComfyNode, get_finalized_class_inputs


def check_graphs():
    used = set()
    for path in (PACKAGE / "workflows").glob("*.json"):
        graph = json.loads(path.read_text())
        for node in graph["nodes"]:
            kind = node["type"]
            used.add(kind)
            if kind not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError(f"Missing native node: {kind} in {path.name}")
            cls = nodes.NODE_CLASS_MAPPINGS[kind]
            schema = cls.INPUT_TYPES()
            if issubclass(cls, ComfyNode):
                # Use the same expansion as native V3 execution, including
                # official Math Expression inputs such as values.a/values.b.
                live_inputs = {field["name"]: None for field in node.get("inputs", [])}
                schema = get_finalized_class_inputs(schema, live_inputs)[0]
            known = set(schema.get("required", {})) | set(schema.get("optional", {}))
            for field in node.get("inputs", []):
                if field["name"] not in known:
                    raise RuntimeError(f"Unknown input: {kind}.{field['name']}")
    for module in ("torch", "huggingface_hub", "hf_xet", "av"):
        importlib.import_module(module)
    print(json.dumps({"native_nodes": sorted(used), "workflows": 5, "node_registration": "passed", "gpu_inference": "not_run"}))


if __name__ == "__main__":
    # Initialize the real server before extension registration, as main.py does.
    # This prepares native nodes and assets without binding a port or loading weights.
    loop, prompt_server, _ = comfy_main.start_comfyui()
    try:
        check_graphs()
    finally:
        prompt_server.asset_manager.shutdown()
        loop.close()
