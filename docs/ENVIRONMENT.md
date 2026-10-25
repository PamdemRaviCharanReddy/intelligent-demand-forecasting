# Environment Report

> Auto-generated on 2026-10-25T10:55 IST. Re-check at the start of each phase.

## Operating System
- **OS:** Windows NT 10.0.26300.0 (Windows 11)
- **Shell:** PowerShell 5.x (note: does not support `&&` chaining; use `;`)
- **Architecture:** x86_64 (AMD64)

## Hardware
- **CPU:** AMD Ryzen (Family 25, Model 68) — 8 logical processors
- **RAM:** ~12 GB (11.7 GB reported)
- **GPU:** NVIDIA GeForce RTX 3050 A Laptop GPU — 4 GB VRAM
- **Disk:** C: drive — 357 GB free of ~475 GB total

## Python
- **System Python:** 3.14.5 and 3.14.7 (two installs — too new for most ML libraries)
- **Project Python:** 3.12.15 — installed via `uv` at `C:\Users\PAMDEM RAVI CHARAN\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe`
- **Package manager:** `uv` 0.13.5 at `C:\Users\PAMDEM RAVI CHARAN\AppData\Local\Python\pythoncore-3.14-64\Scripts\uv.exe`
- **pip:** Available (26.2.1)

## Git
- **Git:** PortableGit 2.56.0.3 at `C:\Users\PAMDEM RAVI CHARAN\PortableGit\cmd\git.exe`
- **Not on system PATH** — must be added per-session: `$env:PATH = "C:\Users\PAMDEM RAVI CHARAN\PortableGit\cmd;$env:PATH"`
- **GitHub CLI (`gh`):** NOT INSTALLED — cannot create PRs, issues, or milestones from CLI. Logged in NEEDS_HUMAN.md.

## Tool Availability
| Tool | Status | Path/Notes |
|------|--------|------------|
| `git` | ✅ Portable | Must add to PATH each session |
| `gh` | ❌ Not installed | Log PR/issue commands in NEEDS_HUMAN.md |
| `python3.12` | ✅ via uv | Use `uv run` or venv |
| `uv` | ✅ | Package/venv manager |
| `pip` | ✅ | Available |
| `node` | ✅ | `C:\Program Files\nodejs\node.exe` |
| `make` | ❌ | Use `python scripts/dev.py` instead |
| `pandoc` | ❌ | PDF/DOCX export limited; use HTML |
| `winget` | ✅ | v1.29.380 — can install tools but needs admin for some |
| `Kaggle CLI` | ❌ | No credentials found |

## Kaggle Credentials
- **~/.kaggle/kaggle.json:** NOT FOUND
- **KAGGLE_USERNAME / KAGGLE_KEY:** NOT SET
- **Action:** Proceed with synthetic data; document manual steps in `data/README.md` and `NEEDS_HUMAN.md`

## Constraints and Adaptations
1. **Python 3.14 → 3.12:** ML libraries (TensorFlow, Prophet, XGBoost, PyTorch) lack 3.14 wheels. Using 3.12.15 via uv.
2. **No `make`:** Using `python scripts/dev.py` as cross-platform task runner.
3. **No `gh`:** Git operations local only; PR/issue commands documented for manual execution.
4. **No `pandoc`:** Reports generated as HTML and Markdown; DOCX/PDF export deferred.
5. **GPU available:** RTX 3050 4GB — sufficient for LSTM on modest sequences. PyTorch can use CUDA.
6. **OneDrive path:** Repo is inside OneDrive Desktop — watch for sync conflicts on large files.
