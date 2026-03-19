import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import fitz
import torch
from PIL import Image


def render_page(page, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DeepSeek-OCR-2 vLLM path on a PDF subset.")
    parser.add_argument("pdf_path")
    parser.add_argument("--page-offset", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--output-json", required=True)
    return parser


def clean_output(text: str) -> str:
    text = text.replace("<｜end▁of▁sentence｜>", "")
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    image_index = 0
    for full_match, label, _coords in matches:
        if label == "image":
            text = text.replace(full_match, f"![](images/{image_index}.jpg)\n")
            image_index += 1
        else:
            text = text.replace(full_match, "")
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = text.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")
    return text.strip()


def main() -> int:
    args = build_parser().parse_args()
    start = time.time()

    repo_root = Path(__file__).resolve().parent
    vllm_dir = repo_root / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm"
    sys.path.insert(0, str(vllm_dir))

    if torch.version.cuda == "11.8":
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from config import MODEL_PATH  # noqa: WPS433
    from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # noqa: WPS433
    from process.image_process import DeepseekOCR2Processor  # noqa: WPS433
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: WPS433

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

    pdf_path = Path(args.pdf_path)
    doc = fitz.open(pdf_path)
    page_start = max(args.page_offset, 0)
    page_end = min(doc.page_count, page_start + args.max_pages)
    images = [render_page(doc[index], args.dpi) for index in range(page_start, page_end)]
    doc.close()

    load_started = time.time()
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max(1, len(images)),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
    )
    load_elapsed = time.time() - load_started

    processor = DeepseekOCR2Processor()
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    preprocess_started = time.time()
    batch_inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=True)},
        }
        for image in images
    ]
    preprocess_elapsed = time.time() - preprocess_started

    logits_processors = [
        NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})
    ]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    generate_started = time.time()
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    generate_elapsed = time.time() - generate_started

    cleaned_lengths = []
    for output in outputs:
        text = output.outputs[0].text
        cleaned_lengths.append(len(clean_output(text)))

    summary = {
        "pages": list(range(page_start + 1, page_end + 1)),
        "total_elapsed_s": time.time() - start,
        "load_elapsed_s": load_elapsed,
        "preprocess_elapsed_s": preprocess_elapsed,
        "generate_elapsed_s": generate_elapsed,
        "cleaned_lengths": cleaned_lengths,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())