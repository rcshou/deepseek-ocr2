# Archive

This folder contains code kept for reference but not used by the active workspace implementation.

Archived items:

- `benchmarks/benchmark_vllm_subset.py`: one-off benchmark helper used during backend comparison.
- `legacy/DeepSeek-OCR2-hf/run_dpsk_ocr2.py`: upstream Transformers example entry script.
- `legacy/DeepSeek-OCR2-vllm/run_dpsk_ocr2_image.py`: upstream vLLM image example entry script.
- `legacy/DeepSeek-OCR2-vllm/run_dpsk_ocr2_pdf.py`: upstream vLLM PDF example entry script.
- `legacy/DeepSeek-OCR2-vllm/run_dpsk_ocr2_eval_batch.py`: upstream vLLM batch-eval example entry script.

Active workspace entrypoints live at the repo root:

- `run_pdf_ocr2.py`: main wrapper, defaults to `--quality-mode vllm`
- `run_pdf_ocr2_vllm.py`: active vLLM PDF runner
- `run_pdf_ocr2_transformers.py`: active Transformers PDF runner