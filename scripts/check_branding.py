#!/usr/bin/env python3
"""Fail CI when upstream branding leaks outside intentional attribution."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_NAME = "Spark" + " Image Lab"
UPSTREAM_SLUG = "spark" + "-image-lab"
LEGACY_MODEL_MARKER = "spark" + "-model"
TOKENS = (UPSTREAM_NAME, UPSTREAM_SLUG, LEGACY_MODEL_MARKER)


def allowed(path: str, line: str) -> bool:
    if path == "NOTICE":
        return line.strip() == f"Based on {UPSTREAM_NAME}:"
    if path == "README.md":
        return f"https://github.com/joeynyc/{UPSTREAM_SLUG}" in line
    if path == "docs/upstream.md":
        return True
    return False


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [entry.decode() for entry in result.stdout.split(b"\0") if entry]


def main() -> int:
    failures = []
    for relative in tracked_files():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if any(token in line for token in TOKENS) and not allowed(relative, line):
                failures.append(f"{relative}:{number}: {line.strip()}")

    if failures:
        print("Unexpected upstream branding references:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("Upstream branding references are limited to the allowlist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
