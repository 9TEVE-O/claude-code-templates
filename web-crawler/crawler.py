#!/usr/bin/env python3
"""Web crawler — usable as a library or CLI."""
import argparse
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class CrawlResult:
    url: str
    status: int
    links: list = field(default_factory=list)
    error: Optional[str] = None


def _normalize(url: str, base: str) -> Optional[str]:
    """Resolve relative URL against base and strip fragments."""
    try:
        resolved = urllib.parse.urljoin(base, url)
        parsed = urllib.parse.urlparse(resolved)
        if parsed.scheme not in ("http", "https"):
            return None
        return urllib.parse.urlunparse(parsed._replace(fragment=""))
    except Exception:
        return None


def _matches(url: str, patterns: list) -> bool:
    return any(p in url for p in patterns)


def crawl(
    seed: str,
    depth: int = 2,
    max_pages: int = 50,
    delay: float = 0.5,
    include: Optional[list] = None,
    exclude: Optional[list] = None,
    session: Optional[requests.Session] = None,
) -> Generator[CrawlResult, None, None]:
    """BFS crawl from *seed*, yielding a CrawlResult for each page visited."""
    include = [p for p in (include or []) if p]
    exclude = [p for p in (exclude or []) if p]

    parsed_seed = urllib.parse.urlparse(seed)
    home_netloc = parsed_seed.netloc

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "WebCrawler/1.0 (educational)"

    enqueued: set = {seed}
    # queue entries: (url, current_depth)
    queue: deque = deque([(seed, 0)])
    count = 0

    while queue and count < max_pages:
        url, cur_depth = queue.popleft()

        if urllib.parse.urlparse(url).netloc != home_netloc:
            continue
        if include and not _matches(url, include):
            continue
        if exclude and _matches(url, exclude):
            continue

        count += 1
        result = CrawlResult(url=url, status=0)

        try:
            resp = session.get(url, timeout=10, allow_redirects=True)
            result.status = resp.status_code

            if resp.status_code == 200 and "text/html" in resp.headers.get("Content-Type", ""):
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup.find_all("a", href=True):
                    link = _normalize(tag["href"], resp.url)
                    if link and link not in enqueued:
                        enqueued.add(link)
                        result.links.append(link)
                        if cur_depth < depth:
                            queue.append((link, cur_depth + 1))
        except Exception as exc:
            result.error = str(exc)

        yield result

        if count < max_pages and queue:
            time.sleep(delay)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl a website from a seed URL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", help="Seed URL")
    parser.add_argument("--depth", type=int, default=2, help="Max crawl depth")
    parser.add_argument("--pages", type=int, default=50, help="Max pages to crawl")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    parser.add_argument(
        "--include", action="append", default=[], metavar="PATTERN",
        help="Only follow URLs containing PATTERN (repeatable)",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="PATTERN",
        help="Skip URLs containing PATTERN (repeatable)",
    )
    args = parser.parse_args()

    print(f"Crawling {args.url}  depth={args.depth}  pages={args.pages}  delay={args.delay}s")
    if args.include:
        print(f"Include: {args.include}")
    if args.exclude:
        print(f"Exclude: {args.exclude}")
    print("-" * 72)

    total = errors = 0
    for r in crawl(
        seed=args.url,
        depth=args.depth,
        max_pages=args.pages,
        delay=args.delay,
        include=args.include,
        exclude=args.exclude,
    ):
        total += 1
        badge = f"[{r.status}]" if r.status else "[ERR]"
        extra = f"  → {len(r.links)} links" if r.links else ""
        if r.error:
            extra += f"  ! {r.error}"
            errors += 1
        print(f"{total:4d}  {badge}  {r.url}{extra}")

    print("-" * 72)
    print(f"Done: {total} pages, {errors} errors.")


if __name__ == "__main__":
    main()
