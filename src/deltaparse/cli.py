"""One command for everything: `deltaparse <command> ...`. Run `deltaparse -h` for the list."""
import argparse
import subprocess
import sys
from pathlib import Path


def _odr(a: argparse.Namespace) -> None:
    from deltaparse import odr

    odr.run(a)


def _gpu(a: argparse.Namespace) -> int:
    # The GPU backend is a Modal app, so it runs through `modal run` (needs the gpu extra).
    cmd = [sys.executable, "-m", "modal", "run", "-m", "deltaparse.gpu",
           "--path", a.path, "--out-dir", a.out_dir, "--mode", a.mode, "--limit", str(a.limit)]
    print("running:", " ".join(cmd[2:]))
    return subprocess.call(cmd)


def _deploy(a: argparse.Namespace) -> int:
    # Chunkr-compatible API on your Modal account (needs the gpu extra and the API key secret).
    cmd = [sys.executable, "-m", "modal", "deploy", "-m", "deltaparse.api"]
    print("running:", " ".join(cmd[2:]))
    return subprocess.call(cmd)


def _view(a: argparse.Namespace) -> None:
    from deltaparse.viewer import rebuild

    print(f"wrote {rebuild(a.json, a.pdf)}")


def _sample(a: argparse.Namespace) -> None:
    from deltaparse.sample import make_sample

    print(f"wrote {make_sample(a.out)}")


def build_parser() -> argparse.ArgumentParser:
    from deltaparse import odr

    ap = argparse.ArgumentParser(
        prog="deltaparse",
        description="Turn PDFs and scans into markdown with layout boxes (Nemotron 3.5 Parse).")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    p = sub.add_parser("odr", help="parse with the hosted OpenDocRouter API (no GPU needed)")
    odr.add_args(p)
    p.set_defaults(func=_odr)

    p = sub.add_parser("gpu", help="parse on your own H100 via Modal + vLLM")
    p.add_argument("path", help="a PDF or a folder of PDFs")
    p.add_argument("--out-dir", default="out_vllm")
    p.add_argument("--mode", default="single", choices=["single", "double", "triple"])
    p.add_argument("--limit", type=int, default=0, help="parse at most N PDFs (0 = all)")
    p.set_defaults(func=_gpu)

    p = sub.add_parser("deploy", help="deploy the Chunkr-compatible API to your Modal account")
    p.set_defaults(func=_deploy)

    p = sub.add_parser("view", help="rebuild the HTML viewer from a saved JSON file")
    p.add_argument("json", type=Path, help="<name>.json written by a backend")
    p.add_argument("pdf", type=Path, help="the source PDF")
    p.set_defaults(func=_view)

    p = sub.add_parser("sample", help="write a one-page sample PDF to try things on")
    p.add_argument("out", nargs="?", type=Path, default=Path("sample.pdf"))
    p.set_defaults(func=_sample)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    sys.exit(main())
