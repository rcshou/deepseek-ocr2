import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 PDF OCR with selectable backend; defaults to vLLM."
    )
    parser.add_argument(
        "--quality-mode",
        choices=("vllm", "transformers"),
        default="vllm",
        help="Backend runner to use. Defaults to vllm.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected runner.")
    return parser


def resolve_runner(mode: str) -> Path:
    repo_root = Path(__file__).resolve().parent
    if mode == "transformers":
        return repo_root / "run_pdf_ocr2_transformers.py"
    return repo_root / "run_pdf_ocr2_vllm.py"


def main() -> int:
    parsed = build_argument_parser().parse_args()
    forwarded_args = list(parsed.args)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]

    runner_path = resolve_runner(parsed.quality_mode)
    command = [sys.executable, str(runner_path), *forwarded_args]
    env = os.environ.copy()
    env.setdefault("DEEPSEEK_OCR_QUALITY_MODE", parsed.quality_mode)
    completed = subprocess.run(command, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())