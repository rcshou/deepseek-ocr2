import argparse
import io
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz
import img2pdf
import torch
from PIL import Image, ImageDraw, ImageFont


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 vLLM PDF OCR from the local repo.")
    parser.add_argument("pdf_path", help="Path to the input PDF.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for OCR outputs.")
    parser.add_argument("--page-offset", type=int, default=0, help="0-based page offset to start from.")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to process.")
    parser.add_argument("--dpi", type=int, default=144, help="Rasterization DPI for PDF pages.")
    parser.add_argument("--max-concurrency", type=int, default=None, help="Maximum number of concurrent page requests.")
    parser.add_argument("--num-workers", type=int, default=None, help="Image preprocessing worker count.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of GPU memory for vLLM.")
    parser.add_argument("--allow-network", action="store_true", help="Allow network access instead of forcing local cache only.")
    return parser


def log(start_time: float, message: str) -> None:
    elapsed = time.time() - start_time
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def render_page(page, dpi: int) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def pil_to_pdf_img2pdf(pil_images: list[Image.Image], output_path: Path) -> None:
    if not pil_images:
        return

    image_bytes_list = []
    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        image_bytes_list.append(img_buffer.getvalue())

    pdf_bytes = img2pdf.convert(image_bytes_list)
    output_path.write_bytes(pdf_bytes)


def re_match(text: str):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    matches_image = []
    matches_other = []
    for match in matches:
        if "<|ref|>image<|/ref|>" in match[0]:
            matches_image.append(match[0])
        else:
            matches_other.append(match[0])
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text, image_width: int, image_height: int):
    del image_width, image_height
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception:
        return None
    return label_type, cor_list


def draw_bounding_boxes(image: Image.Image, refs, images_output_dir: Path, page_index: int) -> Image.Image:
    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    image_counter = 0
    for ref in refs:
        result = extract_coordinates_and_label(ref, image_width, image_height)
        if not result:
            continue
        label_type, points_list = result
        color = (
            60 + (page_index * 17) % 140,
            60 + (page_index * 29) % 140,
            80 + (page_index * 43) % 160,
        )
        color_a = color + (20,)
        for points in points_list:
            x1, y1, x2, y2 = points
            x1 = int(x1 / 999 * image_width)
            y1 = int(y1 / 999 * image_height)
            x2 = int(x2 / 999 * image_width)
            y2 = int(y2 / 999 * image_height)

            if label_type == "image":
                try:
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped.save(images_output_dir / f"{page_index}_{image_counter}.jpg")
                    image_counter += 1
                except Exception:
                    pass

            try:
                width = 4 if label_type == "title" else 2
                draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
                draw_overlay.rectangle([x1, y1, x2, y2], fill=color_a)
                text_bbox = draw.textbbox((0, 0), label_type, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                text_x = x1
                text_y = max(0, y1 - 15)
                draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=(255, 255, 255, 30))
                draw.text((text_x, text_y), label_type, font=font, fill=color)
            except Exception:
                pass

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def clean_content(content: str, page_index: int):
    if "<｜end▁of▁sentence｜>" in content:
        content = content.replace("<｜end▁of▁sentence｜>", "")

    matches_ref, matches_images, matches_other = re_match(content)
    cleaned = content
    for idx, match_image in enumerate(matches_images):
        cleaned = cleaned.replace(match_image, f"![](images/{page_index}_{idx}.jpg)\n")

    for match_other in matches_other:
        cleaned = cleaned.replace(match_other, "")

    cleaned = cleaned.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    cleaned = cleaned.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n").strip()
    return cleaned, matches_ref


def main() -> int:
    args = build_argument_parser().parse_args()
    start_time = time.time()

    repo_root = Path(__file__).resolve().parent
    vllm_dir = repo_root / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm"
    sys.path.insert(0, str(vllm_dir))

    if torch.version.cuda == "11.8":
        os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda-11.8/bin/ptxas")
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    from config import MODEL_PATH, CROP_MODE, TOKENIZER  # noqa: WPS433
    from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # noqa: WPS433
    from process.image_process import DeepseekOCR2Processor  # noqa: WPS433
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # noqa: WPS433
    from vllm import LLM, SamplingParams  # noqa: WPS433
    from vllm.model_executor.models.registry import ModelRegistry  # noqa: WPS433

    logging.getLogger("vllm").setLevel(logging.WARNING)
    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

    pdf_path = Path(args.pdf_path).expanduser()
    output_dir = Path(args.output_dir).expanduser().resolve()
    pages_dir = output_dir / "pages"
    page_outputs_dir = output_dir / "page_outputs"
    images_dir = output_dir / "images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_outputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    local_files_only = not args.allow_network
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    log(start_time, f"PDF path: {pdf_path}")
    log(start_time, f"Torch: {torch.__version__}, CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    page_start = max(args.page_offset, 0)
    page_end = total_pages if args.max_pages is None else min(total_pages, page_start + args.max_pages)
    selected_pages = list(range(page_start, page_end))
    log(start_time, f"Opened PDF with {total_pages} pages; processing pages {page_start + 1}..{page_end}")

    images = []
    render_started = time.time()
    for page_index in selected_pages:
        page_num = page_index + 1
        log(start_time, f"Rendering page {page_num}/{total_pages}")
        image = render_page(doc[page_index], args.dpi)
        page_image_path = pages_dir / f"page_{page_num:03d}.png"
        image.save(page_image_path)
        images.append((page_num, image, page_image_path))
    doc.close()
    log(start_time, f"Rendered {len(images)} pages in {time.time() - render_started:.1f}s")

    load_started = time.time()
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=args.max_concurrency or max(1, len(images)),
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_mm_preprocessor_cache=True,
    )
    log(start_time, f"vLLM engine ready in {time.time() - load_started:.1f}s")

    processor = DeepseekOCR2Processor(tokenizer=TOKENIZER)
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
    num_workers = args.num_workers or min(32, max(1, len(images)))
    preprocess_started = time.time()

    def process_single_image(image_tuple):
        page_num, image, _path = image_tuple
        return page_num, {
            "prompt": prompt,
            "multi_modal_data": {
                "image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE),
            },
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(executor.map(process_single_image, images))
    batch_inputs.sort(key=lambda item: item[0])
    request_payloads = [item[1] for item in batch_inputs]
    log(start_time, f"Prepared {len(request_payloads)} page requests in {time.time() - preprocess_started:.1f}s")

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
    outputs_list = llm.generate(request_payloads, sampling_params=sampling_params)
    log(start_time, f"Generated {len(outputs_list)} page outputs in {time.time() - generate_started:.1f}s")

    combined_cleaned = []
    combined_detailed = []
    layout_images = []
    for (page_num, image, _page_image_path), output in zip(images, outputs_list):
        page_output_dir = page_outputs_dir / f"page_{page_num:03d}"
        page_output_dir.mkdir(parents=True, exist_ok=True)

        raw_text = output.outputs[0].text
        raw_text = raw_text.replace("<｜end▁of▁sentence｜>", "")
        (page_output_dir / "result_det.mmd").write_text(raw_text, encoding="utf-8")

        cleaned_text, matches_ref = clean_content(raw_text, page_num - 1)
        (page_output_dir / "result.mmd").write_text(cleaned_text, encoding="utf-8")

        layout_image = draw_bounding_boxes(image, matches_ref, images_dir, page_num - 1)
        layout_image_path = page_output_dir / "result_with_boxes.jpg"
        layout_image.save(layout_image_path)
        layout_images.append(layout_image)

        combined_detailed.append(raw_text)
        combined_cleaned.append(cleaned_text)
        log(start_time, f"Completed page {page_num}/{total_pages}; chars={len(cleaned_text)}")

    combined_separator = "\n\n<--- Page Split --->\n\n"
    combined_mmd = output_dir / f"{pdf_path.stem}_combined.mmd"
    combined_md = output_dir / f"{pdf_path.stem}_combined.md"
    combined_det = output_dir / f"{pdf_path.stem}_combined_det.mmd"
    layouts_pdf = output_dir / f"{pdf_path.stem}_layouts.pdf"

    combined_mmd.write_text(combined_separator.join(combined_cleaned), encoding="utf-8")
    combined_md.write_text(combined_separator.join(combined_cleaned), encoding="utf-8")
    combined_det.write_text(combined_separator.join(combined_detailed), encoding="utf-8")
    pil_to_pdf_img2pdf(layout_images, layouts_pdf)
    log(start_time, f"Wrote combined outputs to {combined_mmd} and {combined_md}")
    log(start_time, "Run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())