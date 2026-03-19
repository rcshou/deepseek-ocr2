# DeepSeek-OCR-2 Local Wrapper Workspace

This root folder is your custom workspace and should be its own git repository.

The upstream DeepSeek-OCR-2 code now lives in a nested clone at `DeepSeek-OCR2-master/` and should be treated as pull-only reference code. Do not make local edits there. Update it with:

```bash
git -C DeepSeek-OCR2-master pull --ff-only
```

## Layout

- `DeepSeek-OCR2-master/`: upstream clone from `deepseek-ai/DeepSeek-OCR-2`
- `run_pdf_ocr2.py`: main wrapper, defaults to `--quality-mode vllm`
- `run_pdf_ocr2_vllm.py`: active vLLM backend for multi-page throughput
- `run_pdf_ocr2_transformers.py`: active Transformers fallback backend
- `WSL2_SETUP.md`: local environment setup notes
- `archive/`: archived helper scripts and legacy experiments
- `outputs/`: generated OCR results, intentionally not tracked by git

## Runner Usage

Default backend:

```bash
python run_pdf_ocr2.py "/path/to/file.pdf" --output-dir ./outputs/run_vllm
```

Persistent batch mode with one vLLM model load for the whole folder:

```bash
python run_pdf_ocr2.py "/path/to/pdf_folder" --output-dir ./outputs/batch_vllm
```

In directory mode, the vLLM backend loads once, then processes each PDF in the same Python process and writes one subfolder per document under the output directory.

By default, directory mode submits pages in chunks of 4 per vLLM request batch so long documents show progress more steadily. Override that with:

```bash
python run_pdf_ocr2.py "/path/to/pdf_folder" --output-dir ./outputs/batch_vllm --pages-per-batch 6
```

Transformers fallback:

```bash
python run_pdf_ocr2.py --quality-mode transformers "/path/to/file.pdf" --output-dir ./outputs/run_tf
```

Direct backend entrypoints:

```bash
python run_pdf_ocr2_vllm.py "/path/to/file.pdf" --output-dir ./outputs/run_vllm_direct
python run_pdf_ocr2_transformers.py "/path/to/file.pdf" --output-dir ./outputs/run_tf_direct
```

## Dependency Boundary

The vLLM wrapper imports support modules from the nested upstream clone, currently under:

- `DeepSeek-OCR2-master/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/config.py`
- `DeepSeek-OCR2-master/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepseek_ocr2.py`
- `DeepSeek-OCR2-master/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/process/`
- `DeepSeek-OCR2-master/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/deepencoderv2/`

If upstream changes break these imports, update the wrappers in this root repo rather than patching the nested clone.

## Archive

See [archive/README.md](archive/README.md) for archived scripts and benchmark helpers that are no longer part of the active workflow.
