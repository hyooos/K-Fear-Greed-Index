# GitHub Upload Checklist

## 1. 업로드 전 확인

```bash
git status --short
git check-ignore -v data
git check-ignore -v 3_Filtering_final/recollect_final_filtered_model_toxicity
git check-ignore -v 4_Sentiment_analysis/recollect_sentiment_scores_model_toxicity
git check-ignore -v paper_outputs/robustness/cache
```

`data/`, `recollect_*`, `__pycache__/`, `.DS_Store`, `paper_outputs/robustness/cache/`가 GitHub에 올라가지 않으면 정상입니다.

## 2. 포함하면 좋은 파일

- `README.md`
- `environment.yml`
- `requirements.txt`
- `.gitattributes`
- `1_KFGI_subindex/` ~ `8_Dashboard/` 코드
- `tools/`
- `docs/`
- `paper_outputs/figures/`
- `paper_outputs/tables/`
- `paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md`
- `paper_outputs/FINAL_CONSERVATIVE_KFGI_SENSITIVITY.md`

## 3. 제외해야 하는 파일

- NAVER 댓글 원문
- KRX 원본 다운로드 CSV 전체
- `data/`
- `recollect_*`
- `__pycache__/`
- `.DS_Store`
- `.venv/`
- 로컬 전용 중간 CSV
- `paper_outputs/robustness/cache/`

## 4. 권장 업로드 순서

```bash
git add README.md environment.yml requirements.txt .gitignore .gitattributes
git add 1_KFGI_subindex 2_Naver_crawling 3_Filtering_final 4_Sentiment_analysis
git add 5_Merge_to_final_csv 6_KFGI_weight 7_Modeling 8_Dashboard tools docs paper_outputs
git status --short
```

`git status`에서 `data/`나 `recollect_*`가 올라오지 않는지 꼭 확인합니다.

## 5. 커밋 예시

```bash
git commit -m "Organize final K-FGI downside defense package"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

이미 remote가 있다면 `git remote add origin`은 생략합니다.
