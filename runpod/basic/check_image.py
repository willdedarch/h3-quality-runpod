"""Verify native node registration and graph schemas in the built image, without a GPU."""
import asyncio
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
COMFY = Path(os.environ.get("H3_BASIC_COMFY", "/opt/ComfyUI"))
sys.path.insert(0, str(COMFY))
sys.argv = ["check_image", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import nodes


async def main():
    parameters = inspect.signature(nodes.init_extra_nodes).parameters
    kwargs = {k: False for k in ("init_custom_nodes", "init_api_nodes") if k in parameters}
    result = nodes.init_extra_nodes(**kwargs)
    if inspect.isawaitable(result):
        await result
    used = set()
    for path in (PACKAGE / "workflows").glob("*.json"):
        graph = json.loads(path.read_text())
        for node in graph["nodes"]:
            kind = node["type"]
            used.add(kind)
            if kind not in nodes.NODE_CLASS_MAPPINGS:
                raise RuntimeError(f"Missing native node: {kind} in {path.name}")
            schema = nodes.NODE_CLASS_MAPPINGS[kind].INPUT_TYPES()
            known = set(schema.get("required", {})) | set(schema.get("optional", {}))
            for field in node.get("inputs", []):
                if field["name"] not in known:
                    raise RuntimeError(f"Unknown input: {kind}.{field['name']}")
    for module in ("torch", "huggingface_hub", "hf_xet", "av"):
        importlib.import_module(module)
    print(json.dumps({"native_nodes": sorted(used), "workflows": 5, "node_registration": "passed", "gpu_inference": "not_run"}))


if __name__ == "__main__":
    asyncio.run(main())
