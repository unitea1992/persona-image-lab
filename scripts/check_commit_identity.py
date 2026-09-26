#!/usr/bin/env python3
"""Prevent public commits from exposing the maintainer's private email."""

import subprocess
import sys


PUBLIC_NAMES = {"unitea", "unitea1992"}
PUBLIC_EMAIL = "127289237+unitea1992@users.noreply.github.com"


def git_value(format_string: str) -> str:
    return subprocess.run(
        ["git", "show", "-s", f"--format={format_string}", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def main() -> int:
    author_name = git_value("%an")
    author_email = git_value("%ae")
    if author_name.casefold() in PUBLIC_NAMES and author_email != PUBLIC_EMAIL:
        print(
            "Maintainer-authored public commits must use the GitHub noreply address.",
            file=sys.stderr,
        )
        return 1
    print("Commit identity is safe for the public repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
