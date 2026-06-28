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

> **Note**: ruff's native binary is blocked by Windows Application Control on this machine (`[WinError 4551]`). Lint/format commands will fail locally — they pass in CI.

## Stack

Python 3.12, PyTorch ≥2.0 (runtime + tests). NumPy available transitively but not a direct dependency.
