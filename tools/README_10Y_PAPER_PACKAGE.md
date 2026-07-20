# K-FGI 10-Year Paper Package

This folder contains the non-duplicated 10-year materials needed for the paper
and model reproduction.

## Main Dataset

- `01_final_dataset/KFG_final_10y.csv`: model-ready daily dataset.
- `01_final_dataset/KFG_final_10y_raw.csv`: merged dataset before model-ready row filtering.
- `01_final_dataset/KFG_final_10y_missing_report.csv`: missing-value audit.

`KFG_final_10y.csv` covers 2015-01-16 to 2025-12-29 and contains no missing
values in the final modeling columns.

## Subindex Data

`02_subindices` contains each subindex's 10-year outputs and build scripts.
Files with `2013_2025` include warm-up observations for rolling-window
calculations. Files with `2015_2025` are the analysis-period subindex values.

## Sentiment Data

`03_sentiment` contains the daily sentiment feature file and score summaries
from the 10-year Naver comment sentiment pipeline.

## Scripts

`04_scripts` contains the Python scripts used for collection, subindex
construction, final merging, filtering, sentiment aggregation, and modeling.

## Notes

- Legacy 2022-2025-only outputs and dashboard duplicates are not copied here.
- Large cache files, virtual environments, `.git`, `.DS_Store`, and bytecode
  caches are excluded.
- KOSPI200 is used as the price series for the 10-year final dataset. For
  compatibility with the existing modeling code, `kospi_close` is kept as an
  alias of `kospi200_close`.
