# nanograd

## Setup (first time)

```bash
python -m venv .venv
# PowerShell:
.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate

pip install -e ".[dev]"
```

## Commands (run inside activated venv)

- **Test**: `python -m pytest`
- **Lint**: `python -m ruff check .`
- **Format**: `python -m ruff format .`

## Stack

Python 3.12, NumPy ≥2.0 (runtime only). PyTorch ≥2.0 for tests only — never import torch in `src/`.
