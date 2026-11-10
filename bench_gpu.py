"""Benchmark the same page on different GPUs (cold + warm call).
Usage: modal run bench_gpu.py --path some.pdf --page 1 --gpus L4,H100
"""
from pathlib import Path

from parse_modal import Parser, app, render_pages


@app.local_entrypoint()
def bench(path: str, page: int = 0, gpus: str = "L4,H100"):
    png = render_pages(Path(path))[page]["png"]
    for gpu in gpus.split(","):
        p = Parser.with_options(gpu=gpu)()
        for run in ("cold", "warm"):
            r = p.parse.remote(png)
            print(f"{gpu:6} {run}: {r['tokens']} tokens, {r['seconds']}s, "
                  f"{r['tokens'] / r['seconds']:.0f} tok/s, {len(r['blocks'])} blocks")
