ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG DIFFSYNTH_COMMIT=fb337fb
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 APP_ROOT=/app DIFFSYNTH_ROOT=/opt/DiffSynth-Studio RUNTIME_ROOT=/runpod-volume/privacy-identity-lora MODEL_CACHE_ROOT=/runpod-volume/models/identity-lora HF_HOME=/runpod-volume/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/modelscope/DiffSynth-Studio.git /opt/DiffSynth-Studio && git -C /opt/DiffSynth-Studio checkout --detach "${DIFFSYNTH_COMMIT}" && python -m pip install --upgrade pip setuptools wheel && python -m pip install -e /opt/DiffSynth-Studio
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt
COPY handler.py /app/handler.py
COPY identity_worker /app/identity_worker
RUN mkdir -p /runpod-volume/privacy-identity-lora /runpod-volume/models/identity-lora /runpod-volume/huggingface && python -m compileall -q /app
CMD ["python","-u","/app/handler.py"]
