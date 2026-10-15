"""Create a portable source/workflow ZIP; never include model weights or secrets."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
archive = ROOT.parent / "h3-quality-bf16-singularity-2026-10-15.zip"
allowed = {".json", ".py", ".txt"}
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or any(p.startswith(".test-") for p in path.parts):
            continue
        if path.name == "prepare_manifest.py":
            continue
        if path.suffix not in allowed and path.name not in {"Dockerfile", "Dockerfile.dockerignore", "LICENSE.workflow_templates"}:
            continue
        output.write(path, "runpod/basic/" + path.relative_to(ROOT).as_posix())
print(archive)
