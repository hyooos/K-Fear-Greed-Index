# GitHub Release Notes

## Final Paper Version

This repository has been organized around the final downside-defense model:

```text
lean K-FGI + smoothed sentiment + EGARCH volatility + 1.0x exposure cap
```

## What Changed

- The main README now describes K-FGI as a downside-risk-management framework, not a return-prediction model.
- The final feature set was reduced to 10 features.
- Raw sentiment features were removed from the final model in favor of `sent_composite_ma10`.
- The final strategy uses a no-leverage `1.0x` exposure cap.
- Feature importance was recalculated using market-down-day defense success as the target.
- Paper-ready statistical tests, conditional defense tables, PCA tables, and literature positioning summaries were consolidated under `paper_outputs/`.

## Main Files for Review

- `README.md`
- `docs/FINAL_MODEL_REPRODUCIBILITY.md`
- `docs/DATA_AVAILABILITY.md`
- `paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md`
- `paper_outputs/tables/final_master_statistical_tests.csv`
- `paper_outputs/tables/final_master_conditional_downside_defense.csv`
- `paper_outputs/tables/final_master_feature_role_map.csv`
- `paper_outputs/tables/final_master_elasticnet_defense_feature_importance.csv`
- `paper_outputs/tables/final_master_rf_defense_permutation_importance.csv`
- `paper_outputs/tables/final_conservative_kfgi_sensitivity.csv`

## Files Intentionally Excluded

- Raw NAVER comments
- Raw crawling archives
- Manual KRX downloads
- Local `data/` folders
- Conda/venv directories
- temporary logs and caches

## Suggested GitHub Description

Korean Fear & Greed Index using KRX market indicators, smoothed NAVER Finance comment sentiment, and EGARCH volatility for KOSPI200 downside risk management.

