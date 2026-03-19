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
    parser.add_argument("input_path", help="Path to an input PDF or a directory containing PDFs.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for OCR outputs.")
    parser.add_argument("--page-offset", type=int, default=0, help="0-based page offset to start from.")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum number of pages to process.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of PDFs to process when input_path is a directory.")
    parser.add_argument(
        "--pages-per-batch",
        type=int,
        default=None,
        help="Maximum pages to submit to vLLM at once for each PDF. Defaults to 4 in directory mode.",
    )
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


def discover_pdf_inputs(input_path: Path, max_files: int | None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pdf_paths = sorted(
        [path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: path.name.lower(),
    )
    if max_files is not None:
        pdf_paths = pdf_paths[:max_files]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in directory: {input_path}")
    return pdf_paths


def selected_page_count(pdf_path: Path, page_offset: int, max_pages: int | None) -> int:
    with fitz.open(pdf_path) as document:
        total_pages = document.page_count
    page_start = max(page_offset, 0)
    page_end = total_pages if max_pages is None else min(total_pages, page_start + max_pages)
    return max(0, page_end - page_start)


def build_output_dirs(pdf_paths: list[Path], base_output_dir: Path) -> dict[Path, Path]:
    if len(pdf_paths) == 1:
        return {pdf_paths[0]: base_output_dir}

    output_dirs: dict[Path, Path] = {}
    used_names: dict[str, int] = {}
    for pdf_path in pdf_paths:
        output_name = pdf_path.stem
        duplicate_count = used_names.get(output_name, 0)
        used_names[output_name] = duplicate_count + 1
        if duplicate_count:
            output_name = f"{output_name}_{duplicate_count + 1}"
        output_dirs[pdf_path] = base_output_dir / output_name
    return output_dirs


def resolve_pages_per_batch(pdf_paths: list[Path], requested: int | None) -> int | None:
    if requested is not None:
        return max(1, requested)
    if len(pdf_paths) > 1:
        return 4
    return None


def resolve_vllm_support_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm",
        repo_root / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm",
    ]
    for candidate in candidates:
        if (candidate / "config.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate DeepSeek-OCR-2 vLLM support modules under the nested upstream clone."
    )


def build_request_payloads(images, processor, crop_mode, prompt: str, num_workers: int):
    def process_single_image(image_tuple):
        page_num, image, _path = image_tuple
        return page_num, {
            "prompt": prompt,
            "multi_modal_data": {
                "image": processor.tokenize_with_images(images=[image], bos=True, eos=True, cropping=crop_mode),
            },
        }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(executor.map(process_single_image, images))
    batch_inputs.sort(key=lambda item: item[0])
    return [item[1] for item in batch_inputs]


def process_pdf(pdf_path: Path, output_dir: Path, runtime: dict, args, start_time: float) -> None:
    pages_dir = output_dir / "pages"
    page_outputs_dir = output_dir / "page_outputs"
    images_dir = output_dir / "images"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_outputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    log(start_time, f"[{pdf_path.name}] PDF path: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    page_start = max(args.page_offset, 0)
    page_end = total_pages if args.max_pages is None else min(total_pages, page_start + args.max_pages)
    selected_pages = list(range(page_start, page_end))
    log(start_time, f"[{pdf_path.name}] Opened PDF with {total_pages} pages; processing pages {page_start + 1}..{page_end}")

    images = []
    render_started = time.time()
    for page_index in selected_pages:
        page_num = page_index + 1
        log(start_time, f"[{pdf_path.name}] Rendering page {page_num}/{total_pages}")
        image = render_page(doc[page_index], args.dpi)
        page_image_path = pages_dir / f"page_{page_num:03d}.png"
        image.save(page_image_path)
        images.append((page_num, image, page_image_path))
    doc.close()
    log(start_time, f"[{pdf_path.name}] Rendered {len(images)} pages in {time.time() - render_started:.1f}s")

    combined_cleaned = []
    combined_detailed = []
    layout_images = []
    page_batch_size = runtime["pages_per_batch"] or len(images)
    total_batches = (len(images) + page_batch_size - 1) // page_batch_size
    num_workers = args.num_workers or min(32, max(1, page_batch_size))

    for batch_start in range(0, len(images), page_batch_size):
        batch_images = images[batch_start: batch_start + page_batch_size]
        batch_number = (batch_start // page_batch_size) + 1

        preprocess_started = time.time()
        request_payloads = build_request_payloads(
            batch_images,
            runtime["processor"],
            runtime["crop_mode"],
            runtime["prompt"],
            num_workers,
        )
        log(
            start_time,
            f"[{pdf_path.name}] Prepared chunk {batch_number}/{total_batches} with {len(request_payloads)} page request(s) in {time.time() - preprocess_started:.1f}s",
        )

        generate_started = time.time()
        outputs_list = runtime["llm"].generate(request_payloads, sampling_params=runtime["sampling_params"])
        log(
            start_time,
            f"[{pdf_path.name}] Generated chunk {batch_number}/{total_batches} in {time.time() - generate_started:.1f}s",
        )

        for (page_num, image, _page_image_path), output in zip(batch_images, outputs_list):
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
            log(start_time, f"[{pdf_path.name}] Completed page {page_num}/{total_pages}; chars={len(cleaned_text)}")

    combined_separator = "\n\n<--- Page Split --->\n\n"
    combined_mmd = output_dir / f"{pdf_path.stem}_combined.mmd"
    combined_md = output_dir / f"{pdf_path.stem}_combined.md"
    combined_det = output_dir / f"{pdf_path.stem}_combined_det.mmd"
    layouts_pdf = output_dir / f"{pdf_path.stem}_layouts.pdf"

    combined_mmd.write_text(combined_separator.join(combined_cleaned), encoding="utf-8")
    combined_md.write_text(combined_separator.join(combined_cleaned), encoding="utf-8")
    combined_det.write_text(combined_separator.join(combined_detailed), encoding="utf-8")
    pil_to_pdf_img2pdf(layout_images, layouts_pdf)
    log(start_time, f"[{pdf_path.name}] Wrote combined outputs to {combined_mmd} and {combined_md}")
    log(start_time, f"[{pdf_path.name}] Run complete")


def main() -> int:
    args = build_argument_parser().parse_args()
    start_time = time.time()

    input_path = Path(args.input_path).expanduser()
    pdf_paths = discover_pdf_inputs(input_path, args.max_files)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dirs = build_output_dirs(pdf_paths, output_dir)
    pages_per_batch = resolve_pages_per_batch(pdf_paths, args.pages_per_batch)

    max_num_seqs = args.max_concurrency
    if max_num_seqs is None:
        if pages_per_batch is not None:
            max_num_seqs = pages_per_batch
        else:
            max_num_seqs = max(selected_page_count(pdf_path, args.page_offset, args.max_pages) for pdf_path in pdf_paths)
        max_num_seqs = max(1, max_num_seqs)

    repo_root = Path(__file__).resolve().parent
    vllm_dir = resolve_vllm_support_dir(repo_root)
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

    local_files_only = not args.allow_network
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    log(start_time, f"Discovered {len(pdf_paths)} PDF(s) from {input_path}")
    if pages_per_batch is not None:
        log(start_time, f"Using chunked vLLM page batches of {pages_per_batch}")
    log(start_time, f"Torch: {torch.__version__}, CUDA available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")

    load_started = time.time()
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        swap_space=0,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_mm_preprocessor_cache=True,
    )
    log(start_time, f"vLLM engine ready in {time.time() - load_started:.1f}s")

    processor = DeepseekOCR2Processor(tokenizer=TOKENIZER)
    prompt = "<image>\n<|grounding|>Convert the document to markdown."

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

    runtime = {
        "llm": llm,
        "processor": processor,
        "sampling_params": sampling_params,
        "prompt": prompt,
        "crop_mode": CROP_MODE,
        "pages_per_batch": pages_per_batch,
    }

    for pdf_path in pdf_paths:
        process_pdf(pdf_path, output_dirs[pdf_path], runtime, args, start_time)

    log(start_time, f"Batch complete; processed {len(pdf_paths)} PDF(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())