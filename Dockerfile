ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG DIFFSYNTH_COMMIT=fb337fb
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ROOT=/app \
    DIFFSYNTH_ROOT=/opt/DiffSynth-Studio \
    RUNTIME_ROOT=/runpod-volume/privacy-identity-lora \
    MODEL_CACHE_ROOT=/runpod-volume/models/identity-lora \
    HF_HOME=/runpod-volume/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none https://github.com/modelscope/DiffSynth-Studio.git /opt/DiffSynth-Studio \
    && git -C /opt/DiffSynth-Studio checkout --detach "${DIFFSYNTH_COMMIT}" \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e /opt/DiffSynth-Studio

COPY scripts/patch_diffsynth_runner.py /tmp/patch_diffsynth_runner.py
RUN python /tmp/patch_diffsynth_runner.py --root /opt/DiffSynth-Studio --expected-commit "${DIFFSYNTH_COMMIT}" \
    && grep -Fq "PRIVACY_WAN_DIT_EXACT_STEP_PATCH_V1" /opt/DiffSynth-Studio/diffsynth/diffusion/runner.py

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade --upgrade-strategy only-if-needed -r /app/requirements.txt \
    && python -m pip check

RUN python -c "from importlib.metadata import version; actual=version('runpod'); expected='1.11.0'; assert actual == expected, f'runpod sdk mismatch: expected {expected}, got {actual}'; print(f'RUNPOD_SDK_BUILD_OK={actual}')"

COPY handler.py /app/handler.py
COPY identity_worker /app/identity_worker

RUN mkdir -p /runpod-volume/privacy-identity-lora /runpod-volume/models/identity-lora /runpod-volume/huggingface \
    && python -m compileall -q /app \
    && python -m identity_worker.runtime_preflight --diffsynth-root /opt/DiffSynth-Studio

CMD ["python", "-u", "/app/handler.py"]
