# The Chunkr-compatible Parse API on your own GPU (AWS, GCP, Azure, on-prem).
# Same CUDA image, model and pinned versions as the Modal deployment (src/deltaparse/gpu.py).
#
#   docker build -t deltaparse .
#   docker run --gpus all -p 8000:8000 -e DELTAPARSE_API_KEY=<a long random string> \
#       -v deltaparse-data:/data -v deltaparse-vllm-cache:/root/.cache/vllm deltaparse
#
# Needs an NVIDIA Hopper GPU (H100 tested; the vLLM profile uses FlashAttention 3) and the
# NVIDIA Container Toolkit. See docs/docker.md.

# CUDA devel image: FlashInfer JIT-compiles its sampling kernels with nvcc at warm-up
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    DELTAPARSE_MODEL_DIR=/model \
    DELTAPARSE_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

# The model (weights plus its nemotron_parse package), then vLLM. These layers are the big ones.
RUN pip install huggingface-hub==1.24.0 \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Nemotron-3.5-Parse', local_dir='/model')"
RUN cd /model && pip install '.[vllm,client]' 'vllm==0.26.0' 'transformers==5.14.1' && (pip check || true)
RUN pip install pymupdf==1.27.2.3 'fastapi>=0.110' 'uvicorn>=0.30' 'httpx>=0.27'

# Our package last, so code changes rebuild in seconds.
COPY pyproject.toml README.md LICENSE NOTICE /src/
COPY src /src/src
RUN pip install --no-deps /src && rm -rf /src

EXPOSE 8000
VOLUME ["/data"]
# /ready turns healthy once vLLM has loaded the model (about 2-3 minutes after start).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(httpx.get('http://127.0.0.1:8000/ready', timeout=4).status_code != 200)"
CMD ["deltaparse", "serve", "--host", "0.0.0.0", "--port", "8000"]
