"""Build the repo's Dockerfile and run its own command on a real H100 (via Modal), then parse
PDFs through the Chunkr-compatible API inside that container.

For checking the Docker image when you have no local GPU. With a GPU machine, use
`docker run` instead (docs/docker.md).

    modal run examples/docker_gpu_check.py                    # examples/sample.pdf + scanned_sample.pdf
    modal run examples/docker_gpu_check.py --pdfs a.pdf,b.pdf

Modal builds the image from the Dockerfile (the first build downloads the model and vLLM,
about 10-15 minutes) and ignores CMD, so this starts the same `deltaparse serve` command itself.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
image = modal.Image.from_dockerfile(ROOT / "Dockerfile", context_dir=ROOT)
app = modal.App("deltaparse-docker-check", image=image)


@app.function(gpu="H100", cpu=8, memory=32768, timeout=3600)
def check(pdfs: dict[str, bytes]) -> dict:
    import base64
    import os
    import subprocess
    import time

    import httpx

    key = "docker-check-key"
    t0 = time.time()
    server = subprocess.Popen(["deltaparse", "serve", "--host", "0.0.0.0", "--port", "8000"],
                              env={**os.environ, "DELTAPARSE_API_KEY": key, "DELTAPARSE_DATA_DIR": "/tmp/data"})
    base, auth = "http://127.0.0.1:8000", {"Authorization": key}
    report = {"tasks": {}}
    try:
        while True:  # the API answers first; /ready waits for vLLM
            if server.poll() is not None:
                raise RuntimeError(f"deltaparse serve exited with {server.returncode}")
            try:
                r = httpx.get(f"{base}/ready", timeout=3)
                if r.status_code == 200:
                    break
                if r.status_code == 500:
                    raise RuntimeError(f"vLLM failed: {r.text}")
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 1200:
                raise RuntimeError("not ready after 20 minutes")
            time.sleep(3)
        report["ready_seconds"] = round(time.time() - t0)
        report["health"] = httpx.get(f"{base}/health").text

        for name, data in pdfs.items():
            t = time.time()
            word = "tasks" if name.startswith("scanned") else "task"  # one per client generation
            r = httpx.post(f"{base}/api/v1/{word}/parse", headers=auth, timeout=120,
                           json={"file": base64.b64encode(data).decode(), "file_name": name})
            r.raise_for_status()
            tid = r.json()["task_id"]
            while True:
                rec = httpx.get(f"{base}/api/v1/{word}/{tid}", headers=auth, timeout=60).json()
                if rec["status"] in ("Succeeded", "Failed", "Cancelled"):
                    break
                time.sleep(1)
            segs = [s for c in (rec.get("output") or {}).get("chunks", []) for s in c["segments"]]
            report["tasks"][name] = {
                "status": rec["status"], "message": rec["message"], "seconds": round(time.time() - t, 1),
                "pages": (rec.get("output") or {}).get("page_count"), "segments": len(segs),
                "types": sorted({s["segment_type"] for s in segs}),
                "first_text": next((s["content"][:120] for s in segs if s.get("content")), ""),
            }
        no_key = httpx.post(f"{base}/api/v1/task/parse", json={"file": "x"}).status_code
        report["without_key_status"] = no_key
    finally:
        server.terminate()
    return report


@app.local_entrypoint()
def main(pdfs: str = ""):
    import json

    paths = [Path(p) for p in pdfs.split(",") if p] or [ROOT / "examples/sample.pdf",
                                                        ROOT / "examples/scanned_sample.pdf"]
    report = check.remote({p.name: p.read_bytes() for p in paths})
    print(json.dumps(report, indent=2))
    ok = report["tasks"] and all(t["status"] == "Succeeded" and t["segments"] for t in report["tasks"].values())
    print("DOCKER IMAGE CHECK:", "PASS" if ok and report["without_key_status"] == 401 else "FAIL")
