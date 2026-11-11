"""Time a Parse API call the way users far apart see it: Chunkr's own client, run from several regions.

    modal run examples/region_check.py --url https://<you>--deltaparse.modal.run --key <key>
    modal run examples/region_check.py --url ... --key ... --runs 3 --pdf examples/scanned_sample.pdf

Each region runs a small CPU container with chunkr-ai 0.3.7 (v1 client) and 0.1.0a15 (v2 client)
and calls the API `runs` times in a row; the first call may include a GPU cold start, the rest
are warm. Prints seconds per call, end to end, as the client sees them.
"""
import modal

REGIONS = ["us-east", "us-west", "eu-west", "ap-south"]
CHECK = "/root/chunkr_client_check.py"

app = modal.App("deltaparse-region-check")
images = {
    gen: modal.Image.debian_slim(python_version="3.12").pip_install(pin)
    .add_local_file(__file__.replace("region_check.py", "chunkr_client_check.py"), CHECK)
    .add_local_file(__file__, "/root/region_check.py")  # the serialized functions refer to this module
    for gen, pin in (("v1", "chunkr-ai==0.3.7"), ("v2", "chunkr-ai==0.1.0a15"))
}


def _run(url: str, key: str, pdf: bytes, runs: int) -> list[float]:
    import importlib.util
    import os
    import tempfile
    import time
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("check", CHECK)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    path = Path(tempfile.mkdtemp()) / "doc.pdf"
    path.write_bytes(pdf)
    fn = check.run_v2 if check.sdk_generation() == "v2" else check.run_v1
    out = []
    for _ in range(runs):
        t = time.time()
        res = fn(url, key, path)
        out.append(round(time.time() - t, 1) if res["status"] == "Succeeded" else -1.0)
    return out + [os.environ.get("MODAL_REGION", "?")]


checkers = {}
for region in REGIONS:
    for gen, image in images.items():
        checkers[(region, gen)] = app.function(image=image, region=region, timeout=1800,
                                               name=f"check_{gen}_{region.replace('-', '_')}",
                                               serialized=True)(_run)


@app.local_entrypoint()
def main(url: str, key: str, pdf: str = "examples/scanned_sample.pdf", runs: int = 3,
         regions: str = ",".join(REGIONS)):
    from pathlib import Path

    data = Path(pdf).read_bytes()
    wanted = [r for r in regions.split(",") if r]
    print("region\tclient\tran in\tseconds per call (first may be cold)")
    for region in wanted:  # one region at a time, so the calls do not queue behind each other
        for gen in images:
            *secs, ran_in = checkers[(region, gen)].remote(url.rstrip("/"), key, data, runs)
            print(f"{region}\t{gen}\t{ran_in}\t" + " ".join(f"{s:.1f}" for s in secs), flush=True)
