"""Ephemeral native H3 runtime: pinned downloads, five presets, no Studio dependency."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("H3_BASIC_ROOT", "/workspace/h3-basic"))
COMFY = Path(os.environ.get("H3_BASIC_COMFY", "/opt/ComfyUI"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["size_bytes"]:
        return False
    return digest(path) == item["sha256_published"]


def download_one(item, root, token, download_fn):
    """local_dir + atomic rename stores one full copy on the temporary disk."""
    target = root / item["destination"]
    if target.exists():
        if not verify(target, item):
            raise RuntimeError(f"Existing model differs from the locked file: {target.name}")
        return target
    stage = root / ".downloads" / item["repository"].replace("/", "--")
    last_error = None
    for attempt in range(3):
        try:
            print(f"Downloading {Path(item['file']).name}", flush=True)
            path = Path(download_fn(repo_id=item["repository"], filename=item["file"],
                                    revision=item["revision"], local_dir=stage, token=token))
            if not verify(path, item):
                raise RuntimeError(f"Downloaded model failed size/SHA-256: {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
            print(f"Verified {target.name}", flush=True)
            return target
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Model download failed: {target.name}") from last_error


def materialize(root):
    target = root / "user/default/workflows/H3_QUALITY"
    target.mkdir(parents=True, exist_ok=True)
    for source in (PACKAGE / "workflows").glob("*.json"):
        shutil.copyfile(source, target / source.name)
    originals = root / "official-originals"
    originals.mkdir(exist_ok=True)
    for source in (PACKAGE / "upstream").glob("*.json"):
        shutil.copyfile(source, originals / source.name)
    return target


def main():
    # Configure Hub before importing it. Tokens come only from runtime secrets.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no model download started")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 88 * 1024**3:
        raise RuntimeError("This package targets the complete 96 GB GPU, not a smaller MIG partition")
    probe = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del probe
    print(f"GPU: {device.name}; torch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    for folder in ("models", "input", "output", "temp", "user"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    materialize(ROOT)
    manifest = json.loads((PACKAGE / "models.lock.json").read_text())
    items = manifest["models"]
    # This checks the actual filesystem; no persistent mount is required.
    remaining = sum(x["size_bytes"] for x in items if not (ROOT / x["destination"]).is_file())
    reserve = 8 * 1024**3
    if shutil.disk_usage(ROOT).free < remaining + reserve:
        raise RuntimeError("Insufficient free temporary disk for the locked model set and output reserve; no alternative quantization will be substituted")
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
    token = os.environ.get("HF_TOKEN", "").strip() or None
    # Validate every pinned URL before starting the expensive payload transfers.
    for item in items:
        meta = get_hf_file_metadata(hf_hub_url(item["repository"], item["file"], revision=item["revision"]), token=token)
        if meta.size != item["size_bytes"]:
            raise RuntimeError(f"Pinned remote size differs: {item['file']}")
        if meta.etag and meta.etag.strip('"') != item["sha256_published"]:
            raise RuntimeError(f"Pinned remote SHA-256 differs: {item['file']}")
    workers = int(os.environ.get("H3_BASIC_DOWNLOAD_WORKERS", "2"))
    if not 1 <= workers <= 4:
        raise ValueError("H3_BASIC_DOWNLOAD_WORKERS must be between 1 and 4")
    receipts = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, item, ROOT, token, hf_hub_download) for item in items]
        for item, future in zip(items, futures):
            path = future.result()
            receipts.append({"file": path.name, "sha256": item["sha256_published"], "verified": True})
    (ROOT / "model-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    # Full-package readiness means all five presets have their files, including Singularity.
    print("All 8 model files verified. Starting ComfyUI with the 5 quality workflows.", flush=True)
    command = [sys.executable, str(COMFY / "main.py"), "--listen", "0.0.0.0", "--port", "8188",
        "--models-directory", str(ROOT / "models"), "--input-directory", str(ROOT / "input"),
        "--output-directory", str(ROOT / "output"), "--temp-directory", str(ROOT / "temp"),
        "--user-directory", str(ROOT / "user"), "--use-pytorch-cross-attention"]
    os.chdir(COMFY)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
