"""Shared launcher for year-specific NAVER crawling scripts."""

from __future__ import annotations

import sys

from crawling_naver_10years import main


DEFAULT_ARGS = [
    "--topk", "10",
    "--per_article", "30",
    "--sleep", "0.02",
    "--workers", "8",
    "--max_list_pages", "5",
    "--same_day_only",
    "--sort", "FAVORITE",
    "--out_dir", "data/NAVER_by_year",
]


def run_year(year: int) -> None:
    year_args = [
        "--start", f"{year}0101",
        "--end", f"{year}1231",
        *DEFAULT_ARGS,
    ]
    sys.argv = [sys.argv[0], *year_args, *sys.argv[1:]]
    main()
