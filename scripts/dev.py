"""Cross-platform task runner for IDFS (replaces Makefile).

Usage:
    python scripts/dev.py check      # lint + type-check + tests
    python scripts/dev.py lint       # ruff check + format check
    python scripts/dev.py format     # ruff format (auto-fix)
    python scripts/dev.py test       # pytest
    python scripts/dev.py typecheck  # mypy
    python scripts/dev.py generate   # generate synthetic data
    python scripts/dev.py eda        # run ingestion and generate EDA figures
    python scripts/dev.py benchmark  # run Phase 2 model benchmark
    python scripts/dev.py all        # format + check
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], check: bool = True) -> int:
    """Run a command and return its exit code."""
    print(f"\n{'='*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"\n[FAILED]: {' '.join(cmd)}")
    else:
        print(f"\n[PASSED]: {' '.join(cmd)}")
    return result.returncode


def lint() -> int:
    """Run ruff linter and format check."""
    code = run([sys.executable, "-m", "ruff", "check", "src/", "tests/"])
    code |= run([sys.executable, "-m", "ruff", "format", "--check", "src/", "tests/"])
    return code


def fmt() -> int:
    """Auto-format with ruff."""
    code = run([sys.executable, "-m", "ruff", "format", "src/", "tests/"])
    code |= run([sys.executable, "-m", "ruff", "check", "--fix", "src/", "tests/"])
    return code


def typecheck() -> int:
    """Run mypy type checker."""
    return run([sys.executable, "-m", "mypy", "src/idfs/"])


def test() -> int:
    """Run pytest."""
    return run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def check() -> int:
    """Run all checks: lint, typecheck, test."""
    codes = [lint(), typecheck(), test()]
    return 1 if any(c != 0 for c in codes) else 0


def generate() -> int:
    """Generate synthetic data."""
    return run([sys.executable, "-m", "idfs.cli", "generate"])


def eda() -> int:
    """Run data ingestion and generate EDA figures."""
    return run([sys.executable, "scripts/run_eda.py"])


def benchmark() -> int:
    """Run Phase 2 model benchmark."""
    return run([sys.executable, "scripts/run_benchmark.py"])


def main() -> None:
    """Entry point."""
    commands = {
        "check": check,
        "lint": lint,
        "format": fmt,
        "test": test,
        "typecheck": typecheck,
        "generate": generate,
        "eda": eda,
        "benchmark": benchmark,
        "all": lambda: fmt() or check(),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python scripts/dev.py <command>")
        print(f"Commands: {', '.join(commands)}")
        sys.exit(1)

    sys.exit(commands[sys.argv[1]]())


if __name__ == "__main__":
    main()
