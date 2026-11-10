# Run the Chunkr-compatible API with Docker

Use this to run Parse in your own cloud or data center (AWS, GCP, Azure, on-prem) instead of Modal. The image serves the same API as `deltaparse deploy`, so Chunkr code only changes its base URL and key ([migration guide](migrate-from-chunkr.md)).

## What you need

| | |
|---|---|
| GPU | One NVIDIA Hopper GPU: H100 (tested) or H200. The serving profile uses FlashAttention 3, which needs Hopper. |
| GPU memory | The weights take 1.2 GiB. vLLM uses the rest of the card as KV cache (71 GiB on an H100). |
| Host | Linux with the NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), so `docker run --gpus all` works |
| CPU and RAM | 8 CPUs and 32 GB RAM, as on Modal (pages are rendered on the CPUs) |
| Disk | Room for a large image: the CUDA devel base, PyTorch and vLLM, and the 2.7 GB model (estimate 20-25 GB; not yet measured with `docker images`) |

## Build

```bash
git clone https://github.com/DeltaXY-AI/parse.git
cd parse
docker build -t deltaparse .
```

The model is downloaded at build time, so the container needs no Hugging Face access when it runs. There is no published image yet: build it yourself.

## Run

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"     # pick an API key

docker run -d --name deltaparse --gpus all -p 8000:8000 \
  -e DELTAPARSE_API_KEY=<the key you just printed> \
  -v deltaparse-data:/data \
  -v deltaparse-vllm-cache:/root/.cache/vllm \
  deltaparse
```

The API answers at once. Tasks sent before the model has loaded wait in `Starting` and run when it is ready. To see when it is ready:

```bash
curl http://localhost:8000/ready        # {"ready":false} (503), then {"ready":true} (200)
docker logs -f deltaparse               # "vLLM ready in ...s"
```

Then point Chunkr code at `http://<host>:8000` (client 0.3.x, `CHUNKR_URL`) or `http://<host>:8000/api/v1` (client 0.1.0aN, `CHUNKR_BASE_URL`), with `CHUNKR_API_KEY=<your key>`. To check it with Chunkr's own client:

```bash
python examples/chunkr_client_check.py --url http://localhost:8000 --key <your key> examples/sample.pdf
```

Put HTTPS in front of the container (a load balancer or reverse proxy) before you expose it beyond your network. The API reads `X-Forwarded-Proto`, so task URLs come back as `https://`.

## Settings

| Variable | Default | |
|---|---|---|
| `DELTAPARSE_API_KEY` | required | Clients send it in the `Authorization` header. The container exits if it is not set. |
| `DELTAPARSE_DEFAULT_EXPIRES_IN` | not set (keep tasks) | Seconds to keep a task when the request does not set `expires_in`. Expired tasks are removed hourly. |
| `DELTAPARSE_DATA_DIR` | `/data` | Where tasks are stored, one JSON file each. Mount a volume to keep them across restarts. |

Mount `/root/.cache/vllm` as a volume too, so restarts reuse vLLM's compiled kernels.

## Measured

Built from this Dockerfile and run on one H100 (through Modal, on 2026-11-10), with `examples/docker_gpu_check.py`:

| | |
|---|---|
| Build | 215 s on Modal's builders (the model and vLLM layers are most of it) |
| Start to ready | 128-132 s |
| `sample.pdf` and `scanned_sample.pdf` | 2.0 s and 2.1 s per one-page task, 6 segments each (SectionHeader, Text, Table) |

Long documents were not run through the image yet. The vLLM settings are the same as the Modal deployment, which measured 35 s for 160 pages and 45 s for 252 pages through the API.

## How it differs from the Modal deployment

| | Docker | Modal (`deltaparse deploy`) |
|---|---|---|
| GPU | always on while the container runs | scales to zero after 5 idle minutes |
| Task store | JSON files in `/data` | a Modal Dict |
| API keys | one key (`DELTAPARSE_API_KEY`) | one key, plus per-customer keys (`deltaparse keys`) |
| Restarts | tasks that were running are marked `Failed` ("the server restarted"); finished tasks are kept | |

The container runs one server process: `vllm serve` on `127.0.0.1:8001`, and the API on port 8000. Run one container per GPU. The task store is local to each container, so put several behind a load balancer only with sticky routing by task.

## Without Docker

On a machine that already has the GPU stack, the same server runs with:

```bash
DELTAPARSE_API_KEY=<key> DELTAPARSE_MODEL_DIR=<model folder> deltaparse serve --port 8000
```

`<model folder>` is a download of `nvidia/Nemotron-3.5-Parse` with its package installed (`pip install '.[vllm,client]' vllm==0.26.0 transformers==5.14.1` inside it). The Dockerfile shows each step.

## Check the image on a GPU you do not have locally

`examples/docker_gpu_check.py` builds this Dockerfile on Modal, starts `deltaparse serve` inside it on an H100, and parses the sample PDFs through the API:

```bash
modal run examples/docker_gpu_check.py
```
