"""Shared launcher for sub-index-specific calculator scripts."""

from __future__ import annotations

import sys

from calculate_subindices import main


def run_subindex(command: str) -> None:
    sys.argv = [sys.argv[0], command, *sys.argv[1:]]
    main()
