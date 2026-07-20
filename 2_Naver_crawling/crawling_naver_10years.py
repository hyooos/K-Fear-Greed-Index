# crawling_naver_10years.py
#
# 표준 라이브러리만 사용해 NAVER 금융/증권 뉴스 댓글을 수집합니다.
# 기본 기간은 2016-01-01 ~ 2025-12-31이며, 새/구 기사 목록 URL을 모두 시도합니다.

import argparse
import csv
import html
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, build_opener


KEYWORDS = ("증시", "국내증시", "주식시장", "금리")
SID2_NAMES = {258: "증권", 259: "금융"}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

OLD_LIST_URL = "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid1=101&sid2={sid2}&date={date}&page={page}"
BREAKING_LIST_URL = "https://news.naver.com/breakingnews/section/101/{sid2}?date={date}"

CBOX_API = "https://apis.naver.com/commentBox/cbox5/web_naver_list_jsonp.json"


@dataclass
class Article:
    loop_date: str
    sid2: int
    url: str
    oid: str
    aid: str
    title: str
    keyword: str
    object_id: str
    template_id: str = ""
    ticket: str = "news"
    pub_date: str = ""
    comment_total: int = 0

    @property
    def news_id(self) -> str:
        return f"{self.oid}_{self.aid}"


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._href: Optional[str] = None
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        attr = dict(attrs)
        href = attr.get("href")
        if href:
            self._href = html.unescape(href)
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href:
            title = normalize_space(" ".join(self._chunks))
            self.links.append((self._href, title))
            self._href = None
            self._chunks = []


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def make_opener():
    opener = build_opener()
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Accept-Language", "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"),
    ]
    return opener


def http_get(opener, url: str, referer: Optional[str] = None, timeout: int = 8) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/json,*/*",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    req = Request(url, headers=headers)
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def safe_sleep(base: float) -> None:
    if base <= 0:
        return
    time.sleep(base + random.uniform(0, base * 0.35))


def daterange_yyyymmdd(start: str, end: str) -> Iterable[str]:
    cur = datetime.strptime(start, "%Y%m%d")
    last = datetime.strptime(end, "%Y%m%d")
    while cur <= last:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def extract_oid_aid(url: str) -> Optional[Tuple[str, str]]:
    parsed = urlparse(html.unescape(url))
    m = re.search(r"/(?:mnews/)?article/(\d{3})/(\d{10})", parsed.path)
    if m:
        return m.group(1), m.group(2)
    qs = parse_qs(parsed.query)
    oid = (qs.get("oid") or [""])[0]
    aid = (qs.get("aid") or [""])[0]
    if re.fullmatch(r"\d{3}", oid) and re.fullmatch(r"\d{10}", aid):
        return oid, aid
    return None


def canonical_article_url(oid: str, aid: str) -> str:
    return f"https://n.news.naver.com/mnews/article/{oid}/{aid}"


def first_matched_keyword(title: str) -> str:
    for keyword in KEYWORDS:
        if keyword in title:
            return keyword
    return ""


def parse_article_links(page_html: str, base_url: str) -> List[Tuple[str, str]]:
    result: List[Tuple[str, str]] = []
    seen = set()

    def add_link(raw_url: str, raw_title: str) -> None:
        abs_url = urljoin(base_url, html.unescape(raw_url))
        oa = extract_oid_aid(abs_url)
        if not oa:
            return
        oid, aid = oa
        key = (oid, aid)
        if key in seen:
            return
        seen.add(key)
        title = normalize_space(re.sub("<[^>]+>", " ", raw_title))
        result.append((canonical_article_url(oid, aid), title))

    for m in re.finditer(
        r'<a[^>]+(?:href|data-imp-url)="([^"]*(?:/mnews)?/article/\d{3}/\d{10}[^"]*)"[^>]*>.*?'
        r'<strong class="sa_text_strong">(.+?)</strong>',
        page_html,
        flags=re.S,
    ):
        add_link(m.group(1), m.group(2))

    if result:
        return result

    parser = LinkExtractor()
    parser.feed(page_html)

    for href, title in parser.links:
        abs_url = urljoin(base_url, href)
        oa = extract_oid_aid(abs_url)
        if not oa:
            continue
        oid, aid = oa
        key = (oid, aid)
        if key in seen:
            continue
        seen.add(key)

        if not title:
            pattern = rf"/(?:mnews/)?article/{oid}/{aid}.*?>(.*?)</a>"
            m = re.search(pattern, page_html, flags=re.S)
            title = normalize_space(re.sub("<[^>]+>", " ", m.group(1))) if m else ""
        result.append((canonical_article_url(oid, aid), title))

    return result


def fetch_section_articles_for_day(
    opener,
    date: str,
    sid2: int,
    sleep_sec: float,
    max_list_pages: int,
) -> List[Tuple[str, str]]:
    articles: List[Tuple[str, str]] = []
    seen = set()

    for page_no in range(1, max_list_pages + 1):
        url = OLD_LIST_URL.format(date=date, sid2=sid2, page=page_no)
        try:
            page = http_get(opener, url)
        except Exception as exc:
            print(f"[WARN] old list fetch failed {date} sid2={sid2} page={page_no}: {exc}")
            break

        added = 0
        for article_url, title in parse_article_links(page, url):
            oa = extract_oid_aid(article_url)
            if not oa or oa in seen:
                continue
            seen.add(oa)
            articles.append((article_url, title))
            added += 1

        if added == 0:
            break
        safe_sleep(sleep_sec)

    return articles


def parse_jsonp(text: str) -> Dict:
    m = re.search(r"^[^(]*\((.*)\)\s*;?\s*$", text, flags=re.S)
    payload = m.group(1) if m else text
    return json.loads(payload)


def get_article_pub_date(opener, article_url: str, sleep_sec: float) -> str:
    try:
        page = http_get(opener, article_url, referer="https://news.naver.com/")
    except Exception:
        return ""
    safe_sleep(sleep_sec)

    for pattern in (
        r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})T',
        r'data-date-time="(\d{4}-\d{2}-\d{2})',
        r'class="media_end_head_info_datestamp_time[^"]*"[^>]*>(\d{4})\.(\d{2})\.(\d{2})\.',
    ):
        m = re.search(pattern, page)
        if not m:
            continue
        if len(m.groups()) == 1:
            return m.group(1).replace("-", "")
        return "".join(m.groups())
    return ""


def get_cbox_params(opener, oid: str, aid: str, sleep_sec: float) -> Optional[Tuple[str, str, str]]:
    comment_url = f"https://n.news.naver.com/mnews/article/comment/{oid}/{aid}"
    try:
        page = http_get(opener, comment_url, referer=canonical_article_url(oid, aid))
    except Exception as exc:
        print(f"[WARN] comment page failed {oid}_{aid}: {exc}")
        return None
    safe_sleep(sleep_sec)

    def grab(name: str) -> str:
        m = re.search(rf'"{name}"\s*:\s*"([^"]+)"', page)
        return html.unescape(m.group(1)) if m else ""

    object_id = grab("sObjectId")
    template_id = grab("sTemplateId")
    ticket = grab("sTicket") or "news"

    if object_id and template_id:
        return object_id, template_id, ticket
    return None


def fetch_comment_page(
    opener,
    article: Article,
    page: int,
    page_size: int,
    sort: str,
) -> Dict:
    params = {
        "ticket": article.ticket,
        "pool": "cbox5",
        "lang": "ko",
        "country": "KR",
        "objectId": article.object_id,
        "templateId": article.template_id,
        "pageSize": page_size,
        "indexSize": 10,
        "page": page,
        "sort": sort,
    }
    url = f"{CBOX_API}?{urlencode(params)}"
    referer = f"https://n.news.naver.com/mnews/article/comment/{article.oid}/{article.aid}"
    text = http_get(opener, url, referer=referer)
    data = parse_jsonp(text)
    if not data.get("success"):
        raise RuntimeError(f"cbox failed: {data.get('code')} {data.get('message')}")
    return data.get("result", {})


def fetch_comment_total(opener, article: Article, sleep_sec: float) -> int:
    try:
        result = fetch_comment_page(opener, article, page=1, page_size=1, sort="FAVORITE")
    except Exception as exc:
        print(f"[WARN] count failed {article.news_id}: {exc}")
        return 0
    safe_sleep(sleep_sec)
    count = result.get("count") or {}
    page_model = result.get("pageModel") or {}
    return int(count.get("comment") or count.get("total") or page_model.get("totalRows") or 0)


def parse_comment_time(value: str) -> str:
    return value or ""


def comment_day(value: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value or "")
    return "".join(m.groups()) if m else ""


def collect_comments(
    opener,
    article: Article,
    max_comments: int,
    page_size: int,
    max_pages: int,
    sleep_sec: float,
    same_day_only: bool,
    sort: str,
) -> List[List]:
    rows: List[List] = []
    seen = set()

    for page in range(1, max_pages + 1):
        try:
            result = fetch_comment_page(opener, article, page=page, page_size=page_size, sort=sort)
        except Exception as exc:
            print(f"[WARN] comments failed {article.news_id} page={page}: {exc}")
            break

        comments = result.get("commentList") or []
        if not comments:
            break

        for item in comments:
            comment_id = str(item.get("commentNo") or "")
            if not comment_id or comment_id in seen:
                continue
            seen.add(comment_id)

            comment_at = parse_comment_time(item.get("regTime") or item.get("modTime") or "")
            if same_day_only and article.pub_date and comment_day(comment_at) != article.pub_date:
                continue

            rows.append([
                article.news_id,
                article.pub_date,
                comment_id,
                comment_at,
                normalize_space(item.get("contents") or ""),
                int(item.get("sympathyCount") or 0),
                int(item.get("antipathyCount") or 0),
            ])
            if max_comments > 0 and len(rows) >= max_comments:
                return rows

        page_model = result.get("pageModel") or {}
        if page >= int(page_model.get("totalPages") or page):
            break
        safe_sleep(sleep_sec)

    return rows


def process_one_date(date_idx: int, total_dates: int, loop_date: str, args: argparse.Namespace) -> Tuple[List[str], List[List], List[List]]:
    opener = make_opener()
    logs = [f"\n[{date_idx}/{total_dates}] {loop_date}"]
    all_news_rows: List[List] = []
    all_comment_rows: List[List] = []

    for sid2 in (259, 258):
        links = fetch_section_articles_for_day(
            opener, loop_date, sid2, sleep_sec=args.sleep, max_list_pages=args.max_list_pages
        )
        candidates: List[Article] = []

        for url, title in links:
            article = make_article_from_link(
                opener,
                loop_date,
                sid2,
                url,
                title,
                sleep_sec=args.sleep,
                require_title_keyword=args.require_title_keyword,
            )
            if not article:
                continue
            if not args.allow_pubdate_mismatch and article.pub_date != loop_date:
                continue
            article.comment_total = fetch_comment_total(opener, article, sleep_sec=args.sleep)
            if article.comment_total <= 0:
                continue
            candidates.append(article)

        candidates.sort(key=lambda a: a.comment_total, reverse=True)
        top_articles: List[Article] = []
        seen_news = set()
        for article in candidates:
            if article.news_id in seen_news:
                continue
            seen_news.add(article.news_id)
            top_articles.append(article)
            if len(top_articles) >= args.topk:
                break

        logs.append(f"  sid2={sid2} {SID2_NAMES.get(sid2, '')}: candidates={len(candidates)} top={len(top_articles)}")

        for rank, article in enumerate(top_articles, start=1):
            rows = collect_comments(
                opener=opener,
                article=article,
                max_comments=args.per_article,
                page_size=args.comment_page_size,
                max_pages=args.max_comment_pages,
                sleep_sec=args.sleep,
                same_day_only=args.same_day_only,
                sort=args.sort,
            )

            all_news_rows.append([
                article.loop_date, article.pub_date, article.news_id, article.sid2,
                SID2_NAMES.get(article.sid2, ""), article.keyword, article.title,
                article.comment_total, rank, article.url
            ])
            all_comment_rows.extend(rows)
            logs.append(f"    rank={rank} comments={len(rows)}/{article.comment_total} {article.news_id} {article.title[:60]}")

    return logs, all_news_rows, all_comment_rows


def ensure_csv(path: str, header: Sequence[str], reset: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if reset or not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(header)


def append_rows(path: str, rows: Sequence[Sequence]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)


def load_processed_news(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["news_id"] for row in reader if row.get("news_id")}


def load_processed_comments(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["comment_id"] for row in reader if row.get("comment_id")}


def make_article_from_link(
    opener,
    loop_date: str,
    sid2: int,
    url: str,
    title: str,
    sleep_sec: float,
    require_title_keyword: bool,
) -> Optional[Article]:
    keyword = first_matched_keyword(title)
    if require_title_keyword and not keyword:
        return None
    oa = extract_oid_aid(url)
    if not oa:
        return None
    oid, aid = oa
    pub_date = loop_date
    return Article(
        loop_date=loop_date,
        sid2=sid2,
        url=canonical_article_url(oid, aid),
        oid=oid,
        aid=aid,
        title=title,
        keyword=keyword or "전체",
        object_id=f"news{oid},{aid}",
        template_id="default_economy",
        ticket="news",
        pub_date=pub_date,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--topk", type=int, default=5, help="날짜/섹션별 댓글 많은 기사 수")
    ap.add_argument("--per_article", type=int, default=0, help="0이면 가능한 댓글 전체 저장")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--out_dir", default="data/NAVER_10years")
    ap.add_argument("--test_days", type=int, default=0, help="앞에서 N일만 테스트")
    ap.add_argument("--comment_page_size", type=int, default=100)
    ap.add_argument("--max_comment_pages", type=int, default=1000)
    ap.add_argument("--max_list_pages", type=int, default=20, help="날짜/섹션별 기사 목록 페이지 최대 순회 수")
    ap.add_argument("--require_title_keyword", action="store_true", help="제목에 증시/주식시장/금리 키워드가 있는 기사만 수집")
    ap.add_argument("--sort", default="NEW", choices=["NEW", "FAVORITE", "RELATIVE"])
    ap.add_argument("--same_day_only", action="store_true", help="기사 작성일과 같은 날 댓글만 저장")
    ap.add_argument("--allow_pubdate_mismatch", action="store_true", help="목록 날짜와 실제 기사 작성일이 달라도 저장")
    ap.add_argument("--reset", action="store_true", help="기존 출력 CSV를 덮어쓰기")
    ap.add_argument("--workers", type=int, default=1, help="동시에 처리할 날짜 수")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    opener = make_opener()

    suffix = args.start[:4] if args.start[:4] == args.end[:4] else f"{args.start}_{args.end}"
    news_path = os.path.join(args.out_dir, "article", f"news_{suffix}.csv")
    comments_path = os.path.join(args.out_dir, "comments", f"comments_{suffix}.csv")

    ensure_csv(news_path, [
        "loop_date", "pub_date", "news_id", "section", "section_name", "keyword",
        "title", "comment_total_all", "rank_in_section", "url"
    ], reset=args.reset)
    ensure_csv(comments_path, [
        "news_id", "pub_date", "comment_id", "comment_at",
        "text_raw", "like_count", "dislike_count"
    ], reset=args.reset)

    processed_news = set() if args.reset else load_processed_news(news_path)
    processed_comments = set() if args.reset else load_processed_comments(comments_path)
    dates = list(daterange_yyyymmdd(args.start, args.end))
    if args.test_days > 0:
        dates = dates[:args.test_days]

    workers = max(1, args.workers)
    if workers == 1:
        for date_idx, loop_date in enumerate(dates, start=1):
            logs, news_rows, comment_rows = process_one_date(date_idx, len(dates), loop_date, args)
            for line in logs:
                print(line, flush=True)
            new_news_rows = [row for row in news_rows if row[2] not in processed_news]
            new_news_ids = {row[2] for row in new_news_rows}
            new_comment_rows = []
            for row in comment_rows:
                if row[0] not in new_news_ids or row[2] in processed_comments:
                    continue
                processed_comments.add(row[2])
                new_comment_rows.append(row)
            processed_news.update(new_news_ids)
            append_rows(news_path, new_news_rows)
            append_rows(comments_path, new_comment_rows)
            safe_sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_one_date, date_idx, len(dates), loop_date, args)
                for date_idx, loop_date in enumerate(dates, start=1)
            ]
            for future in as_completed(futures):
                logs, news_rows, comment_rows = future.result()
                for line in logs:
                    print(line, flush=True)
                new_news_rows = [row for row in news_rows if row[2] not in processed_news]
                new_news_ids = {row[2] for row in new_news_rows}
                new_comment_rows = []
                for row in comment_rows:
                    if row[0] not in new_news_ids or row[2] in processed_comments:
                        continue
                    processed_comments.add(row[2])
                    new_comment_rows.append(row)
                processed_news.update(new_news_ids)
                append_rows(news_path, new_news_rows)
                append_rows(comments_path, new_comment_rows)

    print(f"\nDONE\nnews: {news_path}\ncomments: {comments_path}")


if __name__ == "__main__":
    main()
