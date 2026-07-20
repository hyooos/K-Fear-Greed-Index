# Final Model Reproducibility Guide

This guide documents the final GitHub-ready model specification used in the paper outputs.

## 1. Environment

Use the conda environment named `kfgi`.

```bash
conda env create -f environment.yml
conda activate kfgi
```

If the environment already exists:

```bash
conda activate kfgi
pip install -r requirements.txt
```

## 2. Final Input

The final master summary is generated from:

```text
paper_outputs/robustness/data/priority_ab_kfgi_timeseries_robust.csv
```

This file is a compact derived dataset used for paper reproduction. Raw NAVER comment data and KRX manual-download files are intentionally excluded from GitHub.

## 3. Final Feature Set

```text
sub_index2
sub_index3
sub_index4
sub_index5
sub_index6
sub_index7
sent_composite_ma10
egarch_vol
vol_regime_high
vol_ratio
```

Rationale:

- `sub_index2-7`: Korean adaptation of CNN-style market sentiment components.
- `sent_composite_ma10`: 10-day smoothed comment sentiment composite. Raw sentiment features were excluded from the final model because smoothing improved stability and interpretability.
- `egarch_vol`, `vol_regime_high`, `vol_ratio`: volatility state and exposure-control features.
- `sub_index1`: excluded from final K-FGI to avoid duplicating momentum already captured by market/trend features.

## 4. Final Position Rule

The final strategy caps exposure at `1.0x`.

This is a research-design choice, not just a performance choice: the paper is about downside risk management, so the final rule avoids leveraged exposure.

## 5. Reproduce Final Paper Summary

```bash
python tools/create_final_paper_master_md.py
```

Outputs:

```text
paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md
paper_outputs/tables/final_master_statistical_tests.csv
paper_outputs/tables/final_master_conditional_downside_defense.csv
paper_outputs/tables/final_master_feature_role_map.csv
paper_outputs/tables/final_master_elasticnet_defense_feature_importance.csv
paper_outputs/tables/final_master_rf_defense_permutation_importance.csv
paper_outputs/tables/final_master_pca_explained_variance.csv
paper_outputs/tables/final_master_pca_loadings.csv
```

## 6. Reproduce Supporting Experiments

Final conservative sensitivity:

```bash
python tools/run_final_conservative_sensitivity.py
```

Smoothed sentiment feature comparison:

```bash
python tools/run_smoothed_feature_experiments.py
```

Position-cap comparison:

```bash
python tools/run_conservative_kfgi_variants.py
```

Feature-drop ablation:

```bash
python tools/run_feature_drop_ablation.py
```

Event-regime interpretation experiments:

```bash
python tools/run_event_regime_overlay_experiments.py
```

## 7. Statistical Tests Included

The final master summary includes:

- one-sample t-test
- normal z-test
- Newey-West HAC t-test
- Wilcoxon signed-rank test
- bootstrap confidence interval
- F-test variance ratio
- Levene median test
- Welch t-test
- Spearman correlation
- one-way ANOVA
- Kruskal-Wallis test
- PCA/factor-structure check
- Elastic Net logistic feature importance
- Random Forest permutation importance

The feature-importance target is not return prediction. It is:

```text
market-down-day defense success
= 1 if strategy return > Buy & Hold return on a market-down day
```

## 8. Expected Final Numbers

| Metric | Expected value |
| --- | ---: |
| Effective observations | 2,445 |
| Final feature count | 10 |
| Buy & Hold MDD | -41.2% |
| Final K-FGI MDD | -26.6% |
| MDD improvement | +14.6%p |
| Market-down-day defense | 43.1 bp/day |
| Downside defense t-stat | 19.66 |
| Downside defense p-value | <0.001 |
| Downside defense hit rate | 68.1% |
| Annualized volatility reduction | 40.3% |

