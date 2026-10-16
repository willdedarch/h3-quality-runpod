"""CPU HTTP smoke test with synthetic filenames; no learned weights or inference."""
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import bootstrap
from inspect_ready import wait_ready, stop_child


def main():
    with tempfile.TemporaryDirectory(prefix="h3-inspect-ci-") as directory:
        root = Path(directory)
        for name in ("models", "input", "output", "temp", "user"):
            (root / name).mkdir()
        installed_workflows = bootstrap.materialize(root)
        manifest = json.loads((bootstrap.PACKAGE / "models.lock.json").read_text())
        for item in manifest["models"]:
            path = root / item["destination"]
            path.parent.mkdir(parents=True, exist_ok=True)
            # Only test native filename discovery. These are not model weights.
            path.touch()
        command = bootstrap.comfy_command(root) + ["--cpu", "--disable-all-custom-nodes", "--disable-partner-nodes"]
        process = subprocess.Popen(command, cwd=bootstrap.COMFY)
        try:
            result = wait_ready(process, installed_workflows)
            with urllib.request.urlopen("http://127.0.0.1:8188/", timeout=10) as response:
                if response.status != 200 or b"<html" not in response.read().lower():
                    raise RuntimeError("ComfyUI frontend HTML did not load")
            with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=10) as response:
                queue = json.load(response)
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("The smoke test must not submit generation")
            result.update({"test": "real_comfyui_http_cpu", "frontend_html": "passed",
                           "queue_empty": True, "synthetic_model_filenames_only": True,
                           "learned_weights_downloaded": False, "gpu_inference": "not_run"})
            print(json.dumps(result), flush=True)
        finally:
            stop_child(process)


if __name__ == "__main__":
    main()
