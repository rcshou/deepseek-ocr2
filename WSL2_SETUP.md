# WSL2 Setup Guide

This guide recreates the current DeepSeek-OCR-2 workspace inside WSL2 Ubuntu so you can continue there with the VS Code agent.

Repository boundary:

- This root folder is your custom repo.
- `DeepSeek-OCR2-master/` is a nested upstream clone.
- Pull updates only inside the nested clone with `git -C DeepSeek-OCR2-master pull --ff-only`.

## Target outcome

- Ubuntu running under WSL2
- NVIDIA GPU visible inside WSL2
- This project available under Linux
- Local Python virtual environment in WSL2
- CUDA-enabled PyTorch installed
- Project open in VS Code using the WSL extension

## 1. Install WSL2 and Ubuntu

Open an elevated PowerShell on Windows and run:

```powershell
wsl --install -d Ubuntu
```

If WSL is already installed, update it:

```powershell
wsl --update
```

Restart Windows if prompted.

Verify WSL status:

```powershell
wsl --status
```

You want:

- Default version: 2
- A recent WSL release

## 2. Start Ubuntu and create your Linux user

Launch Ubuntu from the Start menu or run:

```powershell
wsl -d Ubuntu
```

Complete the first-run Linux username and password setup.

## 3. Confirm GPU access inside WSL2

Inside Ubuntu, run:

```bash
nvidia-smi
```

If it works, WSL GPU passthrough is active.

If it does not work:

- Update the Windows NVIDIA driver first
- Run `wsl --update` again
- Reboot Windows

## 4. Install Linux basics

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential git curl wget python3 python3-venv python3-pip python3-dev pkg-config libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1
```

Optional but useful:

```bash
sudo apt install -y unzip htop
```

## 5. Choose where to keep the repo in WSL

Best practice is to keep active Linux projects in the Linux filesystem, not directly under `/mnt/c`, for fewer file-watching and path issues.

Create a Linux workspace:

```bash
mkdir -p ~/projects
cd ~/projects
```

## 6. Copy the current Windows workspace into WSL

From Ubuntu:

```bash
cp -r "/mnt/c/Users/reg/test_code_win/deepseek-ocr2" ~/projects/
cd ~/projects/deepseek-ocr2
```

This gives you a Linux-local copy.

If you prefer a fresh git-based copy later, use git instead. For now, copy is the fastest path.

## 7. Remove the Windows virtual environment copy

Do not reuse the `.venv` copied from Windows.

Inside Ubuntu:

```bash
rm -rf .venv
```

## 8. Create a new Linux virtual environment

Inside Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 9. Install CUDA-enabled PyTorch

Use the official Linux CUDA wheel index. Start with CUDA 12.8 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Verify:

```bash
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})"
```

You want `cuda: True`.

## 10. Install the Python dependencies for this project

Inside the activated venv:

```bash
pip install transformers tokenizers einops addict easydict pillow huggingface_hub accelerate safetensors PyMuPDF matplotlib
```

If you want the Unsloth runtime path available too:

```bash
pip install --upgrade unsloth unsloth_zoo
```

## 11. Optional: install flash-attn

This is one of the main reasons to use WSL2 instead of native Windows.

Try:

```bash
pip install flash-attn --no-build-isolation
```

If that fails, keep going without it first. The project can still be tested without it, but flash attention is the more likely path for this model family.

## 12. Add a Hugging Face token

This avoids throttling and remote-code download issues.

Inside Ubuntu:

```bash
export HF_TOKEN="your_token_here"
```

To persist it:

```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## 13. Open the project in VS Code inside WSL

From the project folder in Ubuntu:

```bash
code .
```

Requirements:

- VS Code installed on Windows
- Remote Development or WSL extension installed in VS Code

In the VS Code window, confirm the bottom-left corner shows `WSL: Ubuntu`.

That means the agent will run against the Linux environment, not Windows.

## 14. First checks inside the WSL VS Code terminal

Run:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python run_pdf_ocr2.py --help
```

The workspace wrapper defaults to the vLLM backend. To use the older Transformers path explicitly:

```bash
python run_pdf_ocr2.py --quality-mode transformers --help
```

## 15. Test the same PDF from WSL

Your Windows file is available from WSL under `/mnt/c/...`.

Example:

```bash
python run_pdf_ocr2.py "/mnt/c/Users/reg/OneDrive - SOCAR Exploration/Documents/Geobrain_data/raw_data/Articles/004_015_OGP20170400325-RU.pdf" --output-dir ./outputs --max-pages 1
```

Once that works, run the full document:

```bash
python run_pdf_ocr2.py "/mnt/c/Users/reg/OneDrive - SOCAR Exploration/Documents/Geobrain_data/raw_data/Articles/004_015_OGP20170400325-RU.pdf" --output-dir ./outputs
```

If you want to compare backends directly:

```bash
python run_pdf_ocr2.py --quality-mode vllm "/mnt/c/Users/reg/OneDrive - SOCAR Exploration/Documents/Geobrain_data/raw_data/Articles/004_015_OGP20170400325-RU.pdf" --output-dir ./outputs/vllm
python run_pdf_ocr2.py --quality-mode transformers "/mnt/c/Users/reg/OneDrive - SOCAR Exploration/Documents/Geobrain_data/raw_data/Articles/004_015_OGP20170400325-RU.pdf" --output-dir ./outputs/transformers
```

## 16. Recommended agent workflow in WSL

Use this order:

1. Open Ubuntu project with `code .`
2. Let the agent inspect the workspace
3. Verify Python environment from the WSL terminal
4. Run one-page test first
5. Run the full PDF only after the one-page test succeeds

## 17. If you want the cleanest possible restart

Inside WSL, from the project root:

```bash
rm -rf .venv outputs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install transformers tokenizers einops addict easydict pillow huggingface_hub accelerate safetensors PyMuPDF matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python run_pdf_ocr2.py --help
```

## Notes for this workspace

- The current Windows workspace was useful for dependency discovery, but the model runtime is still hitting compatibility issues.
- WSL2 is the more realistic target for this OCR 2 stack.
- If the official model still needs version alignment in WSL2, do that there rather than spending more time on native Windows patches.