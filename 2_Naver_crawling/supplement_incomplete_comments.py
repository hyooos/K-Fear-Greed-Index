import argparse
import csv
import importlib.util
import time
from collections import Counter
from pathlib import Path


def load_crawler():
    path = Path(__file__).with_name("crawling_naver_10years.py")
    spec = importlib.util.spec_from_file_location("crawling_naver_10years", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_existing_comments(path):
    counts = Counter()
    seen = set()
    if not path.exists():
        return counts, seen
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            news_id = row.get("news_id", "")
            comment_id = row.get("comment_id", "")
            counts[news_id] += 1
            if news_id and comment_id:
                seen.add((news_id, comment_id))
    return counts, seen


def read_article_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows):
    if not rows:
        return
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)


def collect_with_retry(crawler, opener, article, args):
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            return crawler.collect_comments(
                opener=opener,
                article=article,
                max_comments=0,
                page_size=args.comment_page_size,
                max_pages=args.max_comment_pages,
                sleep_sec=args.sleep,
                same_day_only=False,
                sort=args.sort,
            )
        except Exception as exc:
            last_error = exc
            time.sleep(args.retry_sleep * attempt)
    print(f"[WARN] supplement failed {article.news_id}: {last_error}")
    return []


def supplement_year(crawler, year, args):
    base = Path(args.out_dir)
    article_path = base / "article" / f"news_{year}.csv"
    comment_path = base / "comments" / f"comments_{year}.csv"
    if not article_path.exists() or not comment_path.exists():
        print(f"[SKIP] {year}: missing files")
        return

    counts, seen = read_existing_comments(comment_path)
    article_rows = read_article_rows(article_path)
    targets = []
    for row in article_rows:
        try:
            total = int(float(row.get("comment_total_all") or 0))
        except ValueError:
            total = 0
        news_id = row.get("news_id", "")
        if total > counts[news_id]:
            targets.append((total - counts[news_id], total, row))

    targets.sort(reverse=True, key=lambda item: item[0])
    if args.limit:
        targets = targets[: args.limit]

    print(f"[YEAR {year}] targets={len(targets)}")
    opener = crawler.make_opener()
    added_total = 0
    for idx, (gap, total, row) in enumerate(targets, start=1):
        news_id = row["news_id"]
        oid, aid = news_id.split("_", 1)
        article = crawler.Article(
            loop_date=row.get("loop_date", ""),
            sid2=int(row.get("section") or 0),
            url=row.get("url") or crawler.canonical_article_url(oid, aid),
            oid=oid,
            aid=aid,
            title=row.get("title", ""),
            keyword=row.get("keyword", ""),
            object_id=f"news{oid},{aid}",
            template_id="default_economy",
            ticket="news",
            pub_date=row.get("pub_date", ""),
            comment_total=total,
        )
        rows = collect_with_retry(crawler, opener, article, args)
        new_rows = []
        for item in rows:
            key = (item[0], str(item[2]))
            if key in seen:
                continue
            seen.add(key)
            new_rows.append(item)
        append_rows(comment_path, new_rows)
        added_total += len(new_rows)
        print(
            f"[{year} {idx}/{len(targets)}] {news_id} "
            f"gap={gap} fetched={len(rows)} added={len(new_rows)} title={row.get('title','')[:50]}"
        )
        time.sleep(args.sleep)
    print(f"[YEAR {year}] added={added_total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/NAVER_by_year_recollect_allcomments")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--sort", default="FAVORITE", choices=["NEW", "FAVORITE", "RELATIVE"])
    parser.add_argument("--comment_page_size", type=int, default=100)
    parser.add_argument("--max_comment_pages", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.03)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    crawler = load_crawler()
    for year in args.years:
        supplement_year(crawler, year, args)


if __name__ == "__main__":
    main()
