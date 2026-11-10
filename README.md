# Parse

The best OCR for everyone. Own your own intelligence.

Parse turns PDFs and scans into markdown with layout boxes. It uses the best open document model available, which today is [NVIDIA Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse). You can call a hosted API, or run the same model on your own GPU.

## Backends

| Backend | Script | When to use | Measured |
|---|---|---|---|
| OpenDocRouter (hosted) | `parse_odr.py` | No GPU, small or irregular jobs | 208 pages in 109 s, $1.09 per 1k pages, 0 failures |
| Your own GPU (Modal + vLLM, H100) | `parse_vllm.py` | Bulk runs, private documents | 12.2 pages/s, about $0.15-0.35 per 1k pages, about 2.5 min startup |

Both backends produce the same files per PDF:
- `<name>.md`: the combined markdown.
- `<name>.json`: the pages and blocks with boxes.
- `<name>.html`: a side-by-side viewer showing the page image with boxes next to the parsed output.

## Usage

```bash
# Hosted: needs OPEN_DOC_ROUTER_API_KEY in the environment or in .env
python parse_odr.py "<pdf or folder>" --out-dir out_odr [--model nvidia/nemotron-3.5-parse]

# Own GPU on Modal (needs the Modal CLI)
modal run parse_vllm.py --path "<pdf or folder>" --out-dir out_vllm [--mode single|double|triple]

# Rebuild a viewer from a saved JSON file and its source PDF
python viewer.py out/doc.json path/to/doc.pdf

# Rank a folder of PDFs by how hard they are to parse (scans, broken text layers, tables)
python find_tough.py "<folder>" report.csv
```

## Files

| File | Purpose |
|---|---|
| `parse_odr.py` | OpenDocRouter backend |
| `parse_vllm.py` | Modal + vLLM backend: blank-page skipping, fail-fast startup, batched H100 serving |
| `parse_modal.py` | Earlier Transformers version (slow; kept for reference) |
| `viewer.py` | Self-contained HTML viewer |
| `find_tough.py` | Finds difficult PDFs in a folder |
| `make_sample.py`, `sample.pdf` | Small test document |
| `bench_gpu.py` | GPU comparison stub |

## Status

Early and pre-1.0. Expect breaking changes. Bugs and requests go in [GitHub Issues](https://github.com/DeltaXY-AI/parse/issues).

## License

Parse's code is licensed under the [Apache License 2.0](LICENSE). Models have their own licenses: Nemotron 3.5 Parse is under OpenMDW-1.1. See [NOTICE](NOTICE). To contribute, see [CONTRIBUTING.md](CONTRIBUTING.md).
