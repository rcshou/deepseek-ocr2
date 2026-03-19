import argparse
import io
import logging
import os
import re
import time
import warnings
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


FLASH_ATTENTION_WARNING_PATTERNS = [
    r"You are attempting to use Flash Attention 2\.0 without specifying a torch dtype.*",
    r"You are attempting to use Flash Attention 2\.0 with a model not initialized on GPU.*",
]


class RegexMessageFilter(logging.Filter):
    def __init__(self, patterns: list[str]):
        super().__init__()
        self.patterns = [re.compile(pattern) for pattern in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self.patterns)


def configure_warning_filters() -> None:
    for pattern in FLASH_ATTENTION_WARNING_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern)

    log_filter = RegexMessageFilter(FLASH_ATTENTION_WARNING_PATTERNS)
    logging.getLogger().addFilter(log_filter)
    logging.getLogger("transformers").addFilter(log_filter)
    logging.getLogger("transformers.modeling_utils").addFilter(log_filter)


def configure_cuda_environment() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    targets_include = Path(conda_prefix) / "targets" / "x86_64-linux" / "include"
    lib64_path = Path(conda_prefix) / "lib64"

    os.environ.setdefault("CUDA_HOME", conda_prefix)
    os.environ["PATH"] = f"{Path(conda_prefix) / 'bin'}:{os.environ.get('PATH', '')}"
    os.environ["LD_LIBRARY_PATH"] = f"{lib64_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ["CPATH"] = f"{targets_include}:{os.environ.get('CPATH', '')}"


def log(start_time: float, message: str) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def patch_generate(model, tokenizer) -> None:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    model.config.eos_token_id = eos_token_id
    model.config.pad_token_id = pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = eos_token_id
        model.generation_config.pad_token_id = pad_token_id

    original_generate = model.generate

    def generate_with_defaults(*args, **kwargs):
        input_ids = args[0] if args else kwargs.get("input_ids")
        if input_ids is not None and kwargs.get("attention_mask") is None:
            kwargs["attention_mask"] = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        if kwargs.get("pad_token_id") is None:
            kwargs["pad_token_id"] = pad_token_id
        if kwargs.get("eos_token_id") is None:
            kwargs["eos_token_id"] = eos_token_id
        if kwargs.get("do_sample") is not True and kwargs.get("temperature") == 0.0:
            kwargs.pop("temperature")
        return original_generate(*args, **kwargs)

    model.generate = generate_with_defaults


def render_page(page, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 on a PDF from local cache.")
    parser.add_argument("pdf_path", help="Path to the input PDF.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for OCR outputs.")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-OCR-2", help="Model identifier or local path.")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>Convert the document to markdown. ", help="Prompt to use for OCR.")
    parser.add_argument("--dpi", type=int, default=144, help="Rasterization DPI for PDF pages.")
    parser.add_argument("--base-size", type=int, default=1024, help="Base image size for the model.")
    parser.add_argument("--image-size", type=int, default=768, help="Patch image size for the model.")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to process.")
    parser.add_argument("--page-offset", type=int, default=0, help="0-based page offset to start from.")
    parser.add_argument("--allow-network", action="store_true", help="Allow network access instead of forcing local cache only.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    start_time = time.time()

    configure_warning_filters()
    configure_cuda_environment()

    pdf_path = Path(args.pdf_path).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    pages_dir = output_dir / "pages"
    page_outputs_dir = output_dir / "page_outputs"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_outputs_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    local_files_only = not args.allow_network
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    log(start_time, f"PDF path: {pdf_path}")
    log(start_time, f"Torch: {torch.__version__}, CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
    log(start_time, "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    log(start_time, "Tokenizer ready")

    log(start_time, "Loading model")
    model = AutoModel.from_pretrained(
        args.model_name,
        _attn_implementation="flash_attention_2",
        torch_dtype=dtype,
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=local_files_only,
    )
    log(start_time, "Model loaded")

    if torch.cuda.is_available():
        log(start_time, "Moving model to CUDA")
        model = model.eval().cuda().to(torch.bfloat16)
        log(start_time, "Model ready on GPU")
    else:
        model = model.eval()

    patch_generate(model, tokenizer)

    pdf = fitz.open(pdf_path)
    total_pages = pdf.page_count
    page_start = max(args.page_offset, 0)
    page_end = total_pages if args.max_pages is None else min(total_pages, page_start + args.max_pages)
    log(start_time, f"Opened PDF with {total_pages} pages; processing pages {page_start + 1}..{page_end}")

    combined_pages = []
    for page_index in range(page_start, page_end):
        page_num = page_index + 1
        log(start_time, f"Rendering page {page_num}/{total_pages}")
        image = render_page(pdf[page_index], args.dpi)
        page_image_path = pages_dir / f"page_{page_num:03d}.png"
        image.save(page_image_path)

        page_output_dir = page_outputs_dir / f"page_{page_num:03d}"
        page_output_dir.mkdir(parents=True, exist_ok=True)

        log(start_time, f"Infer page {page_num}/{total_pages}")
        model.infer(
            tokenizer,
            prompt=args.prompt,
            image_file=str(page_image_path),
            output_path=str(page_output_dir),
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=True,
            save_results=True,
        )

        result_path = page_output_dir / "result.mmd"
        if not result_path.exists():
            raise RuntimeError(f"Expected OCR output was not written: {result_path}")

        page_text = result_path.read_text(encoding="utf-8")
        combined_pages.append(page_text)
        log(start_time, f"Completed page {page_num}/{total_pages}; chars={len(page_text)}")

    pdf.close()

    combined_text = "\n\n<--- Page Split --->\n\n".join(combined_pages)
    combined_mmd = output_dir / f"{pdf_path.stem}_combined.mmd"
    combined_md = output_dir / f"{pdf_path.stem}_combined.md"
    combined_mmd.write_text(combined_text, encoding="utf-8")
    combined_md.write_text(combined_text, encoding="utf-8")
    log(start_time, f"Wrote combined outputs to {combined_mmd} and {combined_md}")
    log(start_time, "Run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())