# Migrate from Chunkr to Parse (self-hosted)

Chunkr's hosted API shuts down on **November 15, 2026**. Your Chunkr API key will not work anywhere else, and your documents and task outputs are not migrated. Download any outputs you need before then. Source: [Chunkr's FAQ](https://docs.chunkr.ai/pages/migrate-to-extend/faq).

Parse gives you a Chunkr-compatible API on **your own** Modal account. It runs NVIDIA Nemotron 3.5 Parse on an H100 and scales to zero when idle. To run it in your own cloud or data center instead, use the Docker image ([below](#run-it-on-your-own-gpu-with-docker)). Your code keeps using Chunkr's own Python client. Only the base URL and the key change.

## 1. Deploy (about 5 minutes)

You need Python 3.10+ and a [Modal](https://modal.com) account.

```bash
git clone https://github.com/DeltaXY-AI/parse.git
cd parse
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[gpu]"
modal setup                       # log in to Modal, once

# pick an API key for your server and store it as a Modal secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
modal secret create deltaparse-api-key DELTAPARSE_API_KEY=<the key you just printed>

deltaparse deploy
```

The last command prints your API URL, in this form:

```
└── ✓ Created Function web
    └── https://<your-workspace>--deltaparse.us-east.modal.run
```

### Run it on your own GPU with Docker

If you would rather not use Modal, build the image and run it on a machine with an H100 (or H200) and the NVIDIA Container Toolkit:

```bash
docker build -t deltaparse .
docker run -d --name deltaparse --gpus all -p 8000:8000   -e DELTAPARSE_API_KEY=<your key> -v deltaparse-data:/data deltaparse
curl http://localhost:8000/ready        # {"ready":true} after about 2 minutes
```

Your API URL is then `http://<host>:8000`. Use it in step 2 in place of the Modal URL. [docs/docker.md](docker.md) covers requirements, settings and HTTPS.

## 2. Point your Chunkr code at it

Set two environment variables. Your code does not change.

**Chunkr client 0.3.x** (`chunkr.upload(...)`):

```bash
CHUNKR_URL=https://<your-workspace>--deltaparse.us-east.modal.run
CHUNKR_API_KEY=<your key>
```

```python
from chunkr_ai import Chunkr

chunkr = Chunkr()  # reads CHUNKR_URL and CHUNKR_API_KEY
task = chunkr.upload("examples/sample.pdf")
print(task.status, len(task.output.chunks), "chunks")
print(task.markdown()[:300])
```

**Chunkr client 0.1.0aN** (`client.tasks.parse.create(...)`). Note the `/api/v1` at the end of the URL:

```bash
CHUNKR_BASE_URL=https://<your-workspace>--deltaparse.us-east.modal.run/api/v1
CHUNKR_API_KEY=<your key>
```

```python
import base64
import time

from chunkr_ai import Chunkr

client = Chunkr()  # reads CHUNKR_BASE_URL and CHUNKR_API_KEY
data = base64.b64encode(open("examples/sample.pdf", "rb").read()).decode()
task = client.tasks.parse.create(file=data, file_name="sample.pdf")
while not task.completed:
    time.sleep(1)
    task = client.tasks.parse.get(task.task_id)
print(task.status, len(task.output.chunks), "chunks")
print(task.output.chunks[0].segments[0].content)
```

To check a deployment with either client:

```bash
python examples/chunkr_client_check.py --url <your API URL> --key <your key> examples/sample.pdf
```

## 3. What to expect

| | Measured through the API (sample PDFs and 160- and 252-page 10-K filings) |
|---|---|
| First deploy | about 12 minutes on a fresh checkout while Modal builds the images (703 s measured); later deploys are faster |
| First call after idle | about 2.5 minutes while the H100 and vLLM start (135 s measured) |
| Later calls | 8 s for a one-page scan; 35 s for 160 pages; 45 s for 252 pages |
| Throughput | 4.6-5.7 pages/s for one document through the API; the batch `deltaparse gpu` command reaches 12.2 pages/s |
| Cost | about $0.22-0.28 per 1,000 pages, plus about $0.17 per cold start and $0.38 for the 5 idle minutes before shutdown |
| Idle cost | none after those 5 minutes: the GPU shuts down 5 minutes after the last request |

Costs use Modal's list prices: H100 at $0.001097/s, plus 8 CPU cores and 32 GiB of memory, which comes to about $0.00127/s (about $4.58/hour) while the container runs.

## Field mapping

Parse returns the same task and output shape as Chunkr. Fields that Parse cannot fill are present but empty, so Chunkr's clients still accept the response.

| Chunkr field | Parse value |
|---|---|
| `task_id`, `status`, `message`, `created_at`, `started_at`, `finished_at`, `expires_at`, `task_url` | same meaning |
| `completed`, `task_type`, `file_info`, `version_info` (0.1.0aN client) | same meaning; `task_type` is always `Parse` |
| `output.chunks[].chunk_length` | number of words |
| `output.chunks[].content`, `embed` | the segments' content joined |
| `segment.segment_type` | from Nemotron's layout classes: `Title`, `SectionHeader`, `Text`, `ListItem`, `Table`, `Picture` (also for charts), `Caption`, `Formula`, `Footnote`, `PageHeader`, `PageFooter`, or `Page` with `segmentation_strategy: Page` |
| `segment.bbox`, `page_width`, `page_height` | pixels at 144 dpi, top-left origin, like Chunkr |
| `segment.content` | 0.3.x client: plain text. 0.1.0aN client: Markdown, or HTML if `segment_processing.<Type>.format` is `Html`. Tables are HTML. |
| `segment.text` | plain text |
| `segment.markdown`, `segment.html` | Nemotron's markdown, and the same rendered as HTML |
| `segment.segment_length` | number of words |
| `chunk_processing.target_length` | words per chunk (default 4096; 0 = one segment per chunk). Headings start a new chunk. |
| `expires_in` | supported: the task is deleted after that many seconds |

## Known gaps

| Not supported | What happens |
|---|---|
| Word, Excel, PowerPoint input | 400 error. Send PDFs or images (PNG, JPEG, TIFF, GIF, BMP, WebP). |
| Files API (`ch://files/...`), listing tasks, extract tasks, webhooks | 400 or 404. Send the file as base64, a data URL, or an http(s) URL. |
| Updating a task (`PATCH .../parse`, `task.update()`) | 501. Create a new task instead. |
| Page images (`output.pages`), segment crops (`segment.image`), `pdf_url` | `null` |
| LLM options (`segment_processing.<Type>.strategy: LLM`, `description`, `llm_processing`, `extended_context`) | ignored: all content comes from Nemotron |
| Word-level OCR boxes (`segment.ocr`) and layout `confidence` | empty list and `null` |
| Tokenizers other than words, `ocr_strategy`, `high_resolution`, `base64_urls` | ignored |
| Chunkr client 0.3.x on Python 3.14 | the client itself fails inside `nest_asyncio`, against any server. Use Python 3.12 or 3.13 with that client. |

Outputs are kept in a Modal Dict named `deltaparse-tasks` in your account until the task is deleted or `expires_in` passes. Uploaded files are not stored.
