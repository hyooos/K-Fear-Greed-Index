#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
PAPER="$PROJECT_ROOT/논문용"
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$PROJECT_ROOT/논문용_archive_before_restructure_$TS"
TMP="$PROJECT_ROOT/논문용_restructured_tmp_$TS"

copy_dir_contents() {
  local src="$1"
  local dst="$2"
  if [ -d "$src" ]; then
    mkdir -p "$dst"
    cp -R "$src"/. "$dst"/
  fi
}

copy_file_if_exists() {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
  fi
}

mkdir -p "$TMP"

mkdir -p \
  "$TMP/00_docs" \
  "$TMP/01_final_dataset" \
  "$TMP/02_KFGI_subindex/data" \
  "$TMP/02_KFGI_subindex/scripts" \
  "$TMP/03_Naver_crawling/scripts" \
  "$TMP/04_Filtering_final/scripts" \
  "$TMP/05_Sentiment_analysis/data" \
  "$TMP/05_Sentiment_analysis/scripts" \
  "$TMP/06_Merge_to_final_csv" \
  "$TMP/07_KFGI_weight" \
  "$TMP/08_Modeling/scripts" \
  "$TMP/08_Modeling/experiment_outputs" \
  "$TMP/09_Dashboard" \
  "$TMP/10_Paper_outputs/figures" \
  "$TMP/10_Paper_outputs/tables" \
  "$TMP/10_Paper_outputs/data" \
  "$TMP/11_References" \
  "$TMP/12_Reproducibility_tools"

# 00. Documentation
copy_dir_contents "$PAPER/00_docs" "$TMP/00_docs"
copy_dir_contents "$CODE_ROOT/docs" "$TMP/00_docs"
copy_file_if_exists "$CODE_ROOT/README.md" "$TMP/00_docs/GITHUB_README.md"

# 01. Final modeling dataset
copy_dir_contents "$PAPER/01_final_dataset" "$TMP/01_final_dataset"

# 02. K-FGI subindices: data + reconstruction scripts
copy_dir_contents "$PAPER/02_subindices" "$TMP/02_KFGI_subindex/data"
copy_dir_contents "$CODE_ROOT/1_KFGI_subindex" "$TMP/02_KFGI_subindex/scripts"

# 03. NAVER crawling scripts
copy_dir_contents "$PAPER/2_Naver_crawling" "$TMP/03_Naver_crawling/scripts"
copy_dir_contents "$PAPER/04_scripts/crawling" "$TMP/03_Naver_crawling/scripts"

# 04. Filtering scripts
copy_dir_contents "$PAPER/3_Filtering_final" "$TMP/04_Filtering_final/scripts"
copy_dir_contents "$PAPER/04_scripts/filtering" "$TMP/04_Filtering_final/scripts"

# 05. Sentiment data + scripts
copy_dir_contents "$PAPER/03_sentiment" "$TMP/05_Sentiment_analysis/data"
copy_file_if_exists "$PAPER/04_scripts/sentiment.py" "$TMP/05_Sentiment_analysis/scripts/sentiment.py"
copy_file_if_exists "$PAPER/04_scripts/sent_feature.py" "$TMP/05_Sentiment_analysis/scripts/sent_feature.py"
copy_file_if_exists "$PAPER/04_scripts/sent_viz.py" "$TMP/05_Sentiment_analysis/scripts/sent_viz.py"
copy_dir_contents "$PAPER/4_Sentiment_analysis" "$TMP/05_Sentiment_analysis"

# 06. Merge to final CSV
copy_dir_contents "$PAPER/5_Merge_to_final_csv" "$TMP/06_Merge_to_final_csv"
copy_dir_contents "$PAPER/04_scripts/merge" "$TMP/06_Merge_to_final_csv/scripts"

# 07. K-FGI weight and sensitivity
copy_dir_contents "$PAPER/6_KFGI_weight" "$TMP/07_KFGI_weight"

# 08. Modeling and 10-year experiment outputs
copy_dir_contents "$PAPER/7_Modeling" "$TMP/08_Modeling/scripts"
copy_dir_contents "$PAPER/04_scripts/modeling" "$TMP/08_Modeling/scripts"
copy_dir_contents "$PAPER/06_10y_experiments" "$TMP/08_Modeling/experiment_outputs"

# 09. Dashboard
copy_dir_contents "$PAPER/8_Dashboard" "$TMP/09_Dashboard"

# 10. Paper-ready outputs
copy_dir_contents "$PAPER/figure 모음/paper_figures" "$TMP/10_Paper_outputs/figures"
copy_dir_contents "$PAPER/06_10y_experiments/tables" "$TMP/10_Paper_outputs/tables"
copy_file_if_exists "$PAPER/06_10y_experiments/kfgi_10y_timeseries.csv" "$TMP/10_Paper_outputs/data/kfgi_10y_timeseries.csv"
copy_file_if_exists "$PAPER/06_10y_experiments/strategy_returns_10y.csv" "$TMP/10_Paper_outputs/data/strategy_returns_10y.csv"
copy_file_if_exists "$PAPER/06_10y_experiments/run_metadata.json" "$TMP/10_Paper_outputs/data/run_metadata.json"
copy_file_if_exists "$PAPER/figure 모음/paper_figures/FIGURE_GUIDE.md" "$TMP/10_Paper_outputs/FIGURE_GUIDE.md"
copy_file_if_exists "$PAPER/figure 모음/paper_figures/FIGURE_INDEX.md" "$TMP/10_Paper_outputs/FIGURE_INDEX.md"

# 11. References and prior literature
copy_dir_contents "$PAPER/선행연구" "$TMP/11_References"

# 12. Reproducibility tools
copy_dir_contents "$CODE_ROOT/tools" "$TMP/12_Reproducibility_tools"

cat > "$TMP/README.md" <<'EOF'
# K-FGI 10-Year Paper Package

이 폴더는 2015~2025년 10개년 K-FGI 연구를 논문 작성/제출 기준으로 정리한 최종 폴더입니다.

## Folder Flow

```text
00_docs                    연구 설명, 재현 가이드, 문헌 포지셔닝
01_final_dataset           모델 입력용 최종 10개년 CSV
02_KFGI_subindex           K-FGI 하위지표 원자료 및 재구성 스크립트
03_Naver_crawling          NAVER 금융 기사/댓글 수집 스크립트
04_Filtering_final         정치/독성/비관련 댓글 필터링 스크립트
05_Sentiment_analysis      감성 점수 및 일별 감성 피처
06_Merge_to_final_csv      subindex, 감성, 시장 데이터 병합 단계
07_KFGI_weight             K-FGI 가중치 및 민감도 분석
08_Modeling                EGARCH, K-FGI 전략, 검증 결과
09_Dashboard               Streamlit 대시보드
10_Paper_outputs           논문용 figure, table, 요약 데이터
11_References              선행연구 및 참고문헌 자료
12_Reproducibility_tools   전체 재현/정리용 보조 스크립트
```

## Main Result

K-FGI는 수익률 예측 모델이라기보다 NAVER 금융 댓글 감성과 EGARCH 변동성을 결합한 하방 위험 관리형 투자심리 지표입니다.

- 분석 기간: 2015-01-16 ~ 2025-12-29
- 최종 실험 기간: 2015-10-08 ~ 2025-12-22
- 시장 기준: KOSPI200
- 최종 K-FGI 구성: sub_index2~sub_index7 + 감성 피처 + EGARCH 변동성 피처
- sub_index1: 원자료에는 보존, 최종 K-FGI/modeling에서는 제외

## Start Here

1. 연구 전체 개요: `00_docs/EXPERIMENT_SUMMARY_10Y.md`
2. 논문 figure 설명: `10_Paper_outputs/FIGURE_GUIDE.md`
3. 최종 성능표: `10_Paper_outputs/tables/performance_summary.csv`
4. 최종 모델 데이터: `01_final_dataset/KFG_final_10y.csv`
5. 대시보드: `09_Dashboard/streamlit_app.py`

## Dashboard

```bash
conda activate kfgi
streamlit run 09_Dashboard/streamlit_app.py
```

## Notes

원자료, 중간 산출물, 예전 중복 폴더는 삭제하지 않고 `논문용_archive_before_restructure_*` 폴더에 보관했습니다.
EOF

cat > "$TMP/00_docs/FOLDER_STRUCTURE.md" <<'EOF'
# Folder Structure

이 폴더는 논문 작성 순서와 재현 순서를 동일하게 맞춘 구조입니다.

| Folder | Purpose |
|---|---|
| `00_docs` | 연구 설명, 재현 가이드, 문헌 포지셔닝 |
| `01_final_dataset` | 모델에 투입되는 최종 10개년 CSV |
| `02_KFGI_subindex` | K-FGI 하위지표 데이터와 계산 코드 |
| `03_Naver_crawling` | NAVER 금융 댓글 수집 코드 |
| `04_Filtering_final` | 정치/독성/비관련 댓글 필터링 코드 |
| `05_Sentiment_analysis` | 감성 점수 및 감성 피처 생성 |
| `06_Merge_to_final_csv` | 최종 모델 CSV 병합 과정 |
| `07_KFGI_weight` | K-FGI 가중치와 민감도 분석 |
| `08_Modeling` | EGARCH, K-FGI 전략, 성능 검증 |
| `09_Dashboard` | 결과 확인용 Streamlit 대시보드 |
| `10_Paper_outputs` | 논문용 figure/table/data |
| `11_References` | 선행연구 및 참고문헌 |
| `12_Reproducibility_tools` | 전체 재현/정리용 보조 스크립트 |

중복을 피하기 위해 기존 `02_subindices`와 `1_KFGI_subindex`처럼 같은 역할을 하던 폴더는 하나의 canonical folder로 통합했습니다.
EOF

find "$TMP" -name ".DS_Store" -delete
find "$TMP" -name "__pycache__" -type d -prune -exec rm -rf {} +

mkdir -p "$ARCHIVE"
find "$PAPER" -mindepth 1 -maxdepth 1 -exec mv {} "$ARCHIVE"/ \;
find "$TMP" -mindepth 1 -maxdepth 1 -exec mv {} "$PAPER"/ \;
rmdir "$TMP"

echo "paper=$PAPER"
echo "archive=$ARCHIVE"
