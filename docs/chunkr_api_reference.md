# Chunkr API reference (captured 2026-11-10)

Chunkr's hosted API stays live through 2026-11-14 and shuts down on 2026-11-15. This file records the parts of its API that the Parse Chunkr-compatible server (`deltaparse.server`) implements, so the facts survive the shutdown.

Sources:
- OpenAPI spec v2.67.0: https://api.chunkr.ai/docs/openapi.json
- API reference: https://docs.chunkr.ai/api-references/tasks/create-parse-task.md and https://docs.chunkr.ai/api-references/tasks/get-task.md
- Outputs (bbox units): https://docs.chunkr.ai/pages/features/parse/outputs.md
- Shutdown FAQ: https://docs.chunkr.ai/pages/migrate-to-extend/faq
- Python SDK source, read directly: `chunkr-ai` 0.3.7 (v1 client) and `chunkr-ai` 0.1.0a15 (v2 client), https://pypi.org/project/chunkr-ai/

## Two API generations

Chunkr users run one of two Python clients, and they call different paths. A compatible server has to serve both.

| | v1 client (`chunkr-ai` 0.3.x, `from chunkr_ai import Chunkr`) | v2 client (`chunkr-ai` 0.1.0aN, `from chunkr_ai import Chunkr` with `client.tasks.parse`) |
|---|---|---|
| Default base URL | `https://api.chunkr.ai` (env `CHUNKR_URL`) | `https://api.chunkr.ai/api/v1/` (env `CHUNKR_BASE_URL`) |
| Create | `POST {url}/task/parse` | `POST {base}/tasks/parse` |
| Get | `GET {url}/task/{task_id}?include_chunks=&base64_urls=` | `GET {base}/tasks/{task_id}/parse` or `GET {base}/tasks/{task_id}` |
| Poll | `GET {task_url}` every 0.5 s until status is not `Starting`/`Processing` | none built in: the caller calls `get` until `completed` is true |
| Cancel | `GET {url}/task/{task_id}/cancel` | `GET {base}/tasks/{task_id}/cancel` |
| Delete | `DELETE {url}/task/{task_id}` | `DELETE {base}/tasks/{task_id}` |
| Update | `PATCH {url}/task/{task_id}/parse` | not in the v2 spec |
| Other | none | `GET /tasks` (list), `/files` (Files API), `/tasks/extract`, `GET /health` (returns a string) |

## Auth

`Authorization: <api key>`: the bare key, with no `Bearer` prefix. Both SDKs send it this way (v1 `HeadersMixin._headers`, v2 `Chunkr.auth_headers`), and the spec declares an `apiKey` scheme in the `Authorization` header. Both clients read the key from `CHUNKR_API_KEY`.

## Create task request (JSON only)

| Field | Notes |
|---|---|
| `file` (required) | `http(s)://` URL to fetch, `data:<mime>;base64,...`, raw base64, or `ch://files/{id}` (v2 Files API) |
| `file_name` | optional; generated if missing |
| `expires_in` | seconds until the task is deleted |
| `chunk_processing` | `{target_length (words, default 4096; 0 = one segment per chunk), tokenizer {"Enum": "Word"} default}` |
| `segmentation_strategy` | `LayoutAnalysis` (default) or `Page` |
| `segment_processing` | per segment type: `{format: Html or Markdown, strategy: LLM, Auto or Ignore, crop_image, description, extended_context}` |
| `ocr_strategy` | `All` (default) or `Auto` |
| `error_handling` | `Fail` (default) or `Continue` |
| `pipeline`, `llm_processing`, `high_resolution` | deprecated or vendor-specific |

The v1 client sends `{"file": <base64 or URL>, "file_name": ..., **config}`: local files become raw base64, and URLs are passed through.

## Task response

Status values: `Starting`, `Processing`, `Succeeded`, `Failed`, `Cancelled`.

v1 (`TaskResponse`, required fields in bold): **`task_id`**, **`status`**, **`created_at`**, **`configuration`** (object; may be empty), `started_at`, `finished_at`, `expires_at`, `message`, `output`, `task_url` (absolute URL the v1 client polls with GET).

v2 (`ParseGetResponse` and `ParseCreateResponse`) adds the required fields **`completed`** (true when status is `Succeeded`, `Failed` or `Cancelled`), **`file_info`** `{url (required), name, mime_type, page_count}`, **`message`** (string), **`task_type`** (`Parse` or `Extract`) and **`version_info`** `{server_version, client_version}`, where `client_version` is `"Legacy"`, `"Unspecified"`, `{"ManualSdk": ...}` or `{"GeneratedSdk": ...}`.

## Output (`output`)

```
output:  {chunks: [Chunk], file_name, mime_type, page_count, pages: [Page] or null, pdf_url}
Chunk:   {chunk_id, chunk_length (required), segments (required), content, embed}
Segment: {segment_id, segment_type, bbox, page_number, page_width, page_height (all required),
          content, text, html, markdown, llm, image, ocr, confidence, segment_length, embed, description}
BoundingBox: {left, top, width, height}
Page:    {image (required), page_number, page_width, page_height, dpi}
```

- `segment_type`: `Title`, `SectionHeader`, `Text`, `ListItem`, `Table`, `Picture`, `Caption`, `Formula`, `Footnote`, `PageHeader`, `PageFooter`, `Page`; v2 adds `FormRegion`, `GraphicalItem`, `Legend`, `LineNumber`, `PageNumber`, `Unknown`.
- bbox: "pixel (px) values in the page coordinate space", where `dpi` "describes the pixel resolution of the rendered page image". The docs' example uses `page_width` 1224 and `page_height` 1584 at `dpi` 144 (US Letter at 144 dpi). Parse uses 144 dpi too.
- `content`: in v1 the plain text of the segment, with `html` and `markdown` in separate fields. In v2 "either HTML or Markdown, depending on format chosen". `text` is the plain text.
- `chunk_length`: "total number of tokens in the `embed` field", counted with the tokenizer (words by default).
- `segment_length`: length of the segment in tokens.
- v1 helpers: `task.markdown()` joins `segment.markdown` with blank lines, `task.html()` joins `segment.html`, and `task.content()` joins `segment.content`.

Response examples: the docs have no complete, valid task JSON (the outputs page uses `// ...` placeholders). The fragments they do show:

```json
{"segment_type": "Text", "bbox": {"left": 100, "top": 250, "width": 500, "height": 50},
 "page_number": 1, "page_height": 1584, "page_width": 1224}
{"dpi": 144, "image": "https://chunkr.ai/page_1.jpg", "page_number": 1, "page_height": 1584, "page_width": 1224}
```

The tests check Parse's responses against both SDKs' own pydantic models instead (`tests/test_server.py` and `examples/chunkr_client_check.py`).

## Rules from the spec

- Delete: the task must be `Succeeded` or `Failed`.
- Cancel: the task must be `Starting`; otherwise the server returns 400.
- Limits (hosted Chunkr): 10 files per second, 1-hour task timeout, 2,000-page soft limit.

## Shutdown facts (FAQ)

- Live through 2026-11-14, shut down on 2026-11-15.
- Chunkr API keys do not work on Extend, the successor platform. Documents and task outputs are not migrated, so users must download them before the shutdown.
- The open-source repo (https://github.com/lumina-ai-inc/chunkr) stays up, separate from the hosted API.
