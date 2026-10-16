# Build from the project root; independent of the existing Studio application.
# CUDA 13 runtime layer used by the selected Jalberty H3 template family.
# Linux/amd64 manifest resolved from the registry before publication.
FROM ls250824/comfyui-runtime3:06102026@sha256:8d002537bee4151af32dae9b37a759f1d81490c81b45f3bcb5e23ae8e0d0c9b5

ENV PYTHONUNBUFFERED=1 H3_BASIC_ROOT=/workspace/h3-basic \
    H3_BASIC_COMFY=/opt/ComfyUI HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DOWNLOAD_TIMEOUT=120

COPY runpod/basic/ /opt/h3-basic/
RUN python -c "import importlib.metadata as m; from pathlib import Path; Path('/opt/h3-basic/torch.constraints.txt').write_text(''.join(p+'=='+m.version(p)+'\n' for p in ('torch','torchvision','torchaudio')))" \
    && git clone --depth 1 --branch v0.40.0 https://github.com/Comfy-Org/ComfyUI.git /opt/ComfyUI \
    && python -m pip install --no-cache-dir -c /opt/h3-basic/torch.constraints.txt -r /opt/ComfyUI/requirements.txt 'click>=8,<8.4,!=8.3.0' \
    && python /opt/h3-basic/build_workflows.py \
    && python -m pip check \
    && python -m pip freeze > /opt/h3-basic/installed-packages.txt \
    && python /opt/h3-basic/check_image.py \
    && cd /opt/ComfyUI && python main.py --cpu --quick-test-for-ci \
    && python /opt/h3-basic/smoke_inspect.py

EXPOSE 8188
WORKDIR /opt/ComfyUI
ENTRYPOINT ["python", "/opt/h3-basic/bootstrap.py"]
CMD []
