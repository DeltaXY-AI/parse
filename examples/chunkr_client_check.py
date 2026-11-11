"""Check that Chunkr's own Python client works against a Parse server, unchanged.

Works with either Chunkr client generation:
    pip install "chunkr-ai==0.3.7"        # v1 client
    pip install "chunkr-ai==0.1.0a15"     # v2 client

    python examples/chunkr_client_check.py --url https://<you>--deltaparse.modal.run --key <key> [file.pdf]

Only the base URL and key differ from talking to Chunkr itself.
"""
import argparse
import time
from pathlib import Path


def sdk_generation() -> str:
    import chunkr_ai

    return "v2" if hasattr(chunkr_ai, "resources") or hasattr(chunkr_ai, "AsyncChunkr") else "v1"


def run_v1(url: str, key: str, pdf: Path, http_client=None) -> dict:
    from chunkr_ai import Chunkr

    chunkr = Chunkr(url=url, api_key=key)
    if http_client is not None:  # tests: talk to an in-process server
        chunkr._client = http_client
    task = chunkr.upload(str(pdf))  # create + poll until finished, exactly as Chunkr users do
    result = {"status": str(task.status.value), "markdown": task.markdown(), "content": task.content(),
              "segments": [s.segment_type.value for c in task.output.chunks for s in c.segments],
              "task_id": task.task_id}
    chunkr.delete_task(task.task_id)
    return result


def run_v2(url: str, key: str, pdf: Path, http_client=None) -> dict:
    import base64

    from chunkr_ai import Chunkr

    kwargs = {"http_client": http_client} if http_client is not None else {}
    client = Chunkr(base_url=url, api_key=key, **kwargs)
    task = client.tasks.parse.create(file=base64.b64encode(pdf.read_bytes()).decode(), file_name=pdf.name)
    for _ in range(1200):  # up to 10 min: the first call starts the GPU
        if task.completed:
            break
        time.sleep(0.5)
        task = client.tasks.parse.get(task.task_id)
    segs = [s for c in task.output.chunks for s in c.segments] if task.output else []
    result = {"status": task.status, "markdown": "\n\n".join(s.content or "" for s in segs),
              "segments": [s.segment_type for s in segs], "task_id": task.task_id}
    client.tasks.delete(task.task_id)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", nargs="?", type=Path, default=Path(__file__).with_name("sample.pdf"))
    ap.add_argument("--url", required=True, help="the Parse server URL")
    ap.add_argument("--key", required=True, help="the server's API key")
    a = ap.parse_args()
    gen = sdk_generation()
    t = time.time()
    res = (run_v2 if gen == "v2" else run_v1)(a.url.rstrip("/"), a.key, a.pdf)
    print(f"chunkr-ai {gen} client: {res['status']} in {time.time() - t:.1f}s, "
          f"{len(res['segments'])} segments {sorted(set(res['segments']))}")
    print(f"task {res['task_id']} (hosted: `deltaparse timings {res['task_id']}` for each step)")
    print("--- markdown (first 800 chars) ---")
    print(res["markdown"][:800])


if __name__ == "__main__":
    main()
