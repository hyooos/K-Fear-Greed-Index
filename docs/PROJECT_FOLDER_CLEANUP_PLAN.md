# 기계학습 폴더 정리 계획

기준일: 2026-07-19

## 결론

최종적으로 남길 핵심 폴더는 아래 3개이다.

```text
기계학습/
├── kfgi_최종/      # GitHub/reproducibility/code package
├── 논문용/         # 제출용 논문 패키지, figure, 표, 본문 참고자료
└── data/           # 원자료/수동 다운로드/재수집 방어용 local data archive
```

삭제 또는 장기 아카이브 후보는 아래 2개이다.

```text
kfgi/                                      # 이전 3개년/실험용 legacy folder
논문용_archive_before_restructure_20260718_161202/  # 논문용 리팩토링 전 백업
```

단, 바로 삭제하기보다 외장/압축 아카이브로 한 번 보관한 뒤 최종 논문과 GitHub 업로드가 끝나면 삭제하는 것을 권장한다.

## 폴더별 판단

| 폴더 | 현재 역할 | 판단 | 이유 |
| --- | --- | --- | --- |
| `kfgi_최종/` | 최종 코드, GitHub용 README/docs/tools/paper_outputs | 유지 | 최종 모델과 GitHub 업로드 기준 폴더 |
| `논문용/` | 제출용 논문 구조, figure, paper outputs, 참고문헌 | 유지 | 논문 작성/제출에 직접 필요 |
| `data/` | subindex 원자료, KFG_final_10y, KRX 다운로드 자료 | 유지하되 GitHub 제외 | 재현/검증 방어용 원자료 |
| `kfgi/` | 이전 실험, 3개년 모델, old figures, raw comments 일부 | 아카이브 후 삭제 후보 | 최종 모델은 `kfgi_최종`으로 이전됨 |
| `논문용_archive_before_restructure_20260718_161202/` | 논문용 리팩토링 전 백업 | 아카이브 후 삭제 후보 | 현재 `논문용`에 최종 구조가 있음 |
| `requiremnets.txt` | 오타 난 requirements 파일 | 삭제 후보 | `kfgi_최종/requirements.txt`와 `environment.yml` 사용 |
| `.DS_Store` | macOS 자동 파일 | 삭제 | 불필요 |

## 매우 중요한 GitHub 주의사항

`kfgi_최종`의 작업 파일 자체는 크지 않지만, 현재 `.git` 폴더가 약 11GB이다. 이는 과거 git history에 대용량 파일이 들어갔던 흔적으로 보인다.

따라서 GitHub 업로드는 현재 `.git` history를 그대로 push하지 말고, 아래 둘 중 하나로 진행하는 것을 권장한다.

### 권장안 A: 깨끗한 GitHub 업로드 폴더 새로 만들기

```bash
cd /Users/hyowon/Desktop/uni/3-1/기계학습
mkdir kfgi_github_clean
rsync -av --exclude .git --exclude __pycache__ --exclude .DS_Store --exclude data --exclude 'paper_outputs/robustness/cache' kfgi_최종/ kfgi_github_clean/
cd kfgi_github_clean
git init
git add .
git commit -m "Organize final K-FGI downside defense package"
```

이 방식이 가장 안전하다. 기존 `.git` 11GB를 가져가지 않기 때문이다.

### 대안 B: 기존 repo history 정리

`git filter-repo` 또는 BFG Repo-Cleaner로 history를 청소해야 한다. 실수 가능성이 있으므로 본 프로젝트에서는 권장안 A가 더 안전하다.

## 삭제 전 체크리스트

아래 파일이 존재하면 최종 모델/논문용 핵심 산출물은 살아있는 상태다.

```text
kfgi_최종/README.md
kfgi_최종/docs/FINAL_MODEL_REPRODUCIBILITY.md
kfgi_최종/paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md
kfgi_최종/paper_outputs/tables/final_master_statistical_tests.csv
kfgi_최종/paper_outputs/tables/final_master_conditional_downside_defense.csv
kfgi_최종/paper_outputs/tables/final_master_feature_role_map.csv
논문용/10_Paper_outputs/FINAL_PAPER_MASTER_SUMMARY.md
논문용/01_final_dataset/KFG_final_10y.csv
data/KFG_final_10y.csv
```

## 삭제/아카이브 추천 순서

1. `kfgi_최종`에서 최종 스크립트 실행 확인

```bash
cd /Users/hyowon/Desktop/uni/3-1/기계학습/kfgi_최종
conda activate kfgi
python tools/create_final_paper_master_md.py
```

2. 깨끗한 GitHub 업로드 폴더 생성

```bash
cd /Users/hyowon/Desktop/uni/3-1/기계학습
mkdir kfgi_github_clean
rsync -av --exclude .git --exclude __pycache__ --exclude .DS_Store --exclude data --exclude 'paper_outputs/robustness/cache' kfgi_최종/ kfgi_github_clean/
```

3. legacy 폴더를 바로 삭제하지 말고 압축 보관

```bash
cd /Users/hyowon/Desktop/uni/3-1/기계학습
tar -czf kfgi_legacy_archive_20260719.tar.gz kfgi 논문용_archive_before_restructure_20260718_161202
```

4. 압축파일 확인 후 삭제

```bash
ls -lh kfgi_legacy_archive_20260719.tar.gz
```

5. 삭제는 최종 확인 후 진행

```bash
rm -rf kfgi
rm -rf 논문용_archive_before_restructure_20260718_161202
rm -f .DS_Store
rm -f requiremnets.txt
```

## 최종 권장 구조

```text
기계학습/
├── data/                       # local-only raw/intermediate data
├── kfgi_최종/                  # working final repo, local master copy
├── kfgi_github_clean/          # GitHub upload copy, no old .git history
├── 논문용/                     # paper submission package
├── kfgi_legacy_archive_*.tar.gz # temporary archive, external backup recommended
└── 기타 hwp/pptx 문서
```

GitHub에는 `kfgi_github_clean/`을 올리고, 실제 작업은 `kfgi_최종/`에서 계속하는 방식을 추천한다.
