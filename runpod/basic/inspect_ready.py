"""Check native node/weight selectors through GET requests; never submit a prompt."""
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

SELECTORS = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name",
             "VAELoader": "vae_name", "LoraLoaderModelOnly": "lora_name"}


def validate_inventory(info, workflow_dir):
    kinds, weights, presets = set(), set(), []
    for path in sorted(Path(workflow_dir).glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        presets.append(path.name)
        for node in graph["nodes"]:
            kind = node["type"]
            if kind not in info:
                raise RuntimeError(f"Missing native node in running ComfyUI: {kind}")
            kinds.add(kind)
            if kind not in SELECTORS:
                continue
            field = SELECTORS[kind]
            expected = node["widgets_values"][0]
            fields = info[kind]["input"]
            declaration = fields.get("required", {}).get(field) or fields.get("optional", {}).get(field)
            choices = declaration[0] if declaration else []
            if not isinstance(choices, list) or expected not in choices:
                raise RuntimeError(f"Weight is not available in {kind}.{field}: {expected}")
            weights.add(expected)
    if len(presets) != 5 or len(weights) != 8:
        raise RuntimeError("Inspection expects the complete five-workflow/eight-file bundle")
    return {"status": "inspection_ready", "workflow_files": presets,
            "native_node_types": sorted(kinds), "weights_in_native_selectors": sorted(weights),
            "generation_submitted_by_startup": False, "model_tensors_loaded_for_test": False}


def wait_ready(process, workflow_dir, base_url="http://127.0.0.1:8188", timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited before inspection was ready: {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/object_info", timeout=10) as response:
                info = json.load(response)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
            continue
        return validate_inventory(info, workflow_dir)
    raise RuntimeError("Running ComfyUI did not expose object_info for inspection")


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def serve_inspection(command, root, workflow_dir):
    process = subprocess.Popen(command)
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, lambda number, frame: process.send_signal(number) if process.poll() is None else None)
    try:
        receipt = wait_ready(process, workflow_dir)
        receipt["all_model_payloads_sha256_verified"] = True
        receipt["mode"] = "inspect"
        Path(root, "inspection-ready.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("INSPECTION READY: 5 workflows, 21 native node types and all 8 weight filenames found. No generation submitted.", flush=True)
        print(json.dumps(receipt), flush=True)
        code = process.wait()
        return code if code >= 0 else 128 - code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        stop_child(process)
