# Data Availability

본 저장소에는 연구 재현을 위한 코드, 요약 산출물, 논문용 figure/table을 포함합니다. 단, 다음 데이터는 GitHub에 포함하지 않습니다.

- NAVER 금융 댓글 원문
- KRX에서 다운로드한 원본 CSV 전체
- 연도별 crawling 중간 결과
- 독성/정치/종목 관련성 필터링 중간 결과
- 감성 점수 산출 중간 결과

제외 이유는 다음과 같습니다.

- 댓글 원문은 서비스 약관과 개인정보/저작권 측면에서 공개 저장소 업로드에 주의가 필요합니다.
- KRX 원본 파일과 중간 산출물은 용량이 커서 GitHub 일반 저장소 관리에 적합하지 않습니다.
- 논문 재현에는 원자료 자체보다 수집 기준, 처리 코드, 최종 검증 결과가 중요합니다.

논문 제출 및 내부 검증용 전체 산출물은 로컬 `논문용` 폴더에 보관합니다.

```text
/Users/hyowon/Desktop/uni/3-1/기계학습/논문용
```

GitHub 저장소에서 제외한 raw data, 중간 산출물, 과거 실험 파일은 삭제하지 않고 다음 로컬 아카이브에 보관합니다.

```text
/Users/hyowon/Desktop/uni/3-1/기계학습/kfgi_최종_local_archive_ignored
```

GitHub에는 다음 요약 산출물을 포함합니다.

- `paper_outputs/figures/`: 논문용 최종 figure
- `paper_outputs/tables/`: 성능, 검정, 민감도 분석 table
- `docs/EXPERIMENT_SUMMARY_10Y.md`: 10개년 실험 요약

원자료가 필요한 경우 KRX 정보데이터시스템과 NAVER 금융에서 동일 기준으로 재수집해야 합니다.
