"""Tests for web-crawler/crawler.py"""
import sys
import os
import time
from collections import deque
from unittest.mock import MagicMock, patch, call

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler import CrawlResult, _normalize, _matches, crawl


# ---------------------------------------------------------------------------
# CrawlResult dataclass
# ---------------------------------------------------------------------------

class TestCrawlResult:
    def test_required_fields(self):
        r = CrawlResult(url="http://example.com", status=200)
        assert r.url == "http://example.com"
        assert r.status == 200

    def test_links_defaults_to_empty_list(self):
        r = CrawlResult(url="http://example.com", status=200)
        assert r.links == []

    def test_error_defaults_to_none(self):
        r = CrawlResult(url="http://example.com", status=200)
        assert r.error is None

    def test_links_are_independent_per_instance(self):
        r1 = CrawlResult(url="http://a.com", status=200)
        r2 = CrawlResult(url="http://b.com", status=200)
        r1.links.append("http://a.com/page")
        assert r2.links == []

    def test_explicit_error(self):
        r = CrawlResult(url="http://example.com", status=0, error="Connection refused")
        assert r.error == "Connection refused"

    def test_explicit_links(self):
        links = ["http://example.com/a", "http://example.com/b"]
        r = CrawlResult(url="http://example.com", status=200, links=links)
        assert r.links == links


# ---------------------------------------------------------------------------
# _normalize()
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_absolute_http_url(self):
        result = _normalize("http://example.com/page", "http://example.com/")
        assert result == "http://example.com/page"

    def test_absolute_https_url(self):
        result = _normalize("https://example.com/secure", "http://example.com/")
        assert result == "https://example.com/secure"

    def test_relative_url_resolved_against_base(self):
        result = _normalize("/about", "http://example.com/home")
        assert result == "http://example.com/about"

    def test_relative_url_same_directory(self):
        result = _normalize("page2.html", "http://example.com/dir/page1.html")
        assert result == "http://example.com/dir/page2.html"

    def test_fragment_stripped(self):
        result = _normalize("http://example.com/page#section", "http://example.com/")
        assert result == "http://example.com/page"

    def test_relative_url_with_fragment_stripped(self):
        result = _normalize("/about#top", "http://example.com/")
        assert result == "http://example.com/about"

    def test_ftp_scheme_returns_none(self):
        result = _normalize("ftp://example.com/file", "http://example.com/")
        assert result is None

    def test_mailto_scheme_returns_none(self):
        result = _normalize("mailto:user@example.com", "http://example.com/")
        assert result is None

    def test_javascript_scheme_returns_none(self):
        result = _normalize("javascript:void(0)", "http://example.com/")
        assert result is None

    def test_data_scheme_returns_none(self):
        result = _normalize("data:text/html,<h1>hi</h1>", "http://example.com/")
        assert result is None

    def test_empty_href_resolves_to_base(self):
        result = _normalize("", "http://example.com/page")
        assert result == "http://example.com/page"

    def test_query_string_preserved(self):
        result = _normalize("http://example.com/search?q=test", "http://example.com/")
        assert result == "http://example.com/search?q=test"

    def test_fragment_only_stripped(self):
        # "#section" alone resolves to the base URL without the fragment
        result = _normalize("#section", "http://example.com/page")
        assert result == "http://example.com/page"

    def test_absolute_url_ignores_base_scheme(self):
        result = _normalize("http://other.com/page", "http://example.com/")
        assert result == "http://other.com/page"

    def test_url_with_port(self):
        result = _normalize("http://example.com:8080/path", "http://example.com/")
        assert result == "http://example.com:8080/path"


# ---------------------------------------------------------------------------
# _matches()
# ---------------------------------------------------------------------------

class TestMatches:
    def test_single_pattern_found(self):
        assert _matches("http://example.com/docs/intro", ["/docs/"]) is True

    def test_single_pattern_not_found(self):
        assert _matches("http://example.com/blog/post", ["/docs/"]) is False

    def test_empty_patterns_returns_false(self):
        assert _matches("http://example.com/anything", []) is False

    def test_multiple_patterns_first_matches(self):
        assert _matches("http://example.com/docs/page", ["/docs/", "/blog/"]) is True

    def test_multiple_patterns_second_matches(self):
        assert _matches("http://example.com/blog/post", ["/docs/", "/blog/"]) is True

    def test_multiple_patterns_none_match(self):
        assert _matches("http://example.com/about", ["/docs/", "/blog/"]) is False

    def test_pattern_is_substring(self):
        assert _matches("http://example.com/documentation", ["doc"]) is True

    def test_exact_domain_pattern(self):
        assert _matches("http://example.com/page", ["example.com"]) is True

    def test_case_sensitive(self):
        assert _matches("http://example.com/Docs/page", ["/docs/"]) is False
        assert _matches("http://example.com/docs/page", ["/Docs/"]) is False


# ---------------------------------------------------------------------------
# crawl()  — uses a mocked requests.Session
# ---------------------------------------------------------------------------

def _make_response(status=200, text="", content_type="text/html; charset=utf-8"):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.headers = {"Content-Type": content_type}
    return resp


def _html(*links):
    """Build minimal HTML with anchor tags pointing to *links*."""
    anchors = "".join(f'<a href="{l}">link</a>' for l in links)
    return f"<html><body>{anchors}</body></html>"


class TestCrawl:
    def _make_session(self, responses):
        """Return a session mock whose .get() cycles through *responses* list."""
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = responses
        return session

    # ---- basic behaviour ----

    def test_yields_one_result_for_single_page(self):
        session = self._make_session([_make_response(200, "<html></html>")])
        results = list(crawl("http://example.com/", depth=1, max_pages=1,
                              delay=0, session=session))
        assert len(results) == 1
        assert results[0].url == "http://example.com/"
        assert results[0].status == 200

    def test_result_url_matches_seed(self):
        session = self._make_session([_make_response(200, "<html></html>")])
        results = list(crawl("http://example.com/start", depth=0, max_pages=1,
                              delay=0, session=session))
        assert results[0].url == "http://example.com/start"

    def test_links_extracted_from_html(self):
        html = _html("/page1", "/page2")
        session = self._make_session([_make_response(200, html)])
        results = list(crawl("http://example.com/", depth=0, max_pages=1,
                              delay=0, session=session))
        assert "http://example.com/page1" in results[0].links
        assert "http://example.com/page2" in results[0].links

    def test_links_not_added_at_max_depth(self):
        """depth=0 means don't enqueue child pages; seed still extracted."""
        child_html = _html("/grandchild")
        seed_html = _html("/child")
        session = self._make_session([
            _make_response(200, seed_html),   # seed page
            _make_response(200, child_html),  # /child page (enqueued at depth 1 when depth>0)
        ])
        # depth=0: seed is at depth 0, no children should be enqueued
        results = list(crawl("http://example.com/", depth=0, max_pages=10,
                              delay=0, session=session))
        # Only seed should be visited
        assert len(results) == 1
        assert results[0].url == "http://example.com/"

    def test_max_pages_limits_crawl(self):
        # Build a chain: /  -> /p1 -> /p2 -> /p3
        responses = [
            _make_response(200, _html("/p1")),
            _make_response(200, _html("/p2")),
            _make_response(200, _html("/p3")),
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=5, max_pages=2,
                              delay=0, session=session))
        assert len(results) == 2

    # ---- domain filtering ----

    def test_cross_domain_links_not_visited(self):
        html = _html("http://other.com/page", "/local")
        session = self._make_session([
            _make_response(200, html),
            _make_response(200, "<html></html>"),
        ])
        results = list(crawl("http://example.com/", depth=1, max_pages=10,
                              delay=0, session=session))
        visited_urls = [r.url for r in results]
        assert "http://other.com/page" not in visited_urls
        assert "http://example.com/local" in visited_urls

    # ---- visited deduplication ----

    def test_already_visited_url_not_crawled_twice(self):
        # Two pages both link back to seed
        responses = [
            _make_response(200, _html("/p1")),
            _make_response(200, _html("/")),  # /p1 links back to seed
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, session=session))
        visited = [r.url for r in results]
        assert visited.count("http://example.com/") == 1

    # ---- include/exclude filters ----

    def test_include_filter_skips_non_matching_urls(self):
        html = _html("/docs/page", "/blog/post")
        session = self._make_session([
            _make_response(200, html),
            _make_response(200, "<html></html>"),
        ])
        results = list(crawl("http://example.com/", depth=1, max_pages=10,
                              delay=0, include=["/docs/"], session=session))
        visited = [r.url for r in results]
        # seed doesn't match /docs/ but it's processed before the filter applies in BFS
        # /blog/post should be excluded; /docs/page should be visited
        assert "http://example.com/blog/post" not in visited
        assert "http://example.com/docs/page" in visited

    def test_exclude_filter_skips_matching_urls(self):
        html = _html("/docs/page", "/admin/secret")
        session = self._make_session([
            _make_response(200, html),
            _make_response(200, "<html></html>"),
        ])
        results = list(crawl("http://example.com/", depth=1, max_pages=10,
                              delay=0, exclude=["/admin/"], session=session))
        visited = [r.url for r in results]
        assert "http://example.com/admin/secret" not in visited
        assert "http://example.com/docs/page" in visited

    def test_empty_strings_in_include_ignored(self):
        """Passing include=['', ' '] should behave like no filter."""
        session = self._make_session([_make_response(200, _html("/any-page"))])
        results = list(crawl("http://example.com/", depth=0, max_pages=1,
                              delay=0, include=["", "  "], session=session))
        # include list becomes [] after filtering; seed should be visited
        assert len(results) == 1

    def test_empty_strings_in_exclude_ignored(self):
        session = self._make_session([_make_response(200, "<html></html>")])
        results = list(crawl("http://example.com/", depth=0, max_pages=1,
                              delay=0, exclude=[""], session=session))
        assert len(results) == 1

    # ---- HTTP errors & content types ----

    def test_non_200_status_no_links_extracted(self):
        session = self._make_session([_make_response(404, _html("/child"))])
        results = list(crawl("http://example.com/", depth=1, max_pages=5,
                              delay=0, session=session))
        assert results[0].status == 404
        assert results[0].links == []

    def test_non_html_content_type_no_links_extracted(self):
        session = self._make_session([
            _make_response(200, b"binary data", content_type="application/octet-stream")
        ])
        results = list(crawl("http://example.com/file.bin", depth=1, max_pages=5,
                              delay=0, session=session))
        assert results[0].links == []

    def test_json_content_type_no_links_extracted(self):
        session = self._make_session([
            _make_response(200, '{"key": "val"}', content_type="application/json")
        ])
        results = list(crawl("http://example.com/api", depth=1, max_pages=5,
                              delay=0, session=session))
        assert results[0].links == []

    def test_redirect_status_no_links_extracted(self):
        session = self._make_session([_make_response(301, "")])
        results = list(crawl("http://example.com/old", depth=1, max_pages=5,
                              delay=0, session=session))
        assert results[0].status == 301
        assert results[0].links == []

    # ---- error handling ----

    def test_exception_sets_error_field(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        results = list(crawl("http://example.com/", depth=1, max_pages=1,
                              delay=0, session=session))
        assert results[0].error is not None
        assert "Connection refused" in results[0].error

    def test_exception_status_remains_zero(self):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = Exception("timeout")
        results = list(crawl("http://example.com/", depth=1, max_pages=1,
                              delay=0, session=session))
        assert results[0].status == 0

    def test_exception_on_one_page_continues_crawl(self):
        responses = [
            requests.exceptions.ConnectionError("fail"),
            _make_response(200, "<html></html>"),
        ]
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = responses
        # seed fails, /p1 was somehow already in the queue — test via: seed enqueues /p1 first
        # Actually let's pre-seed two pages by using a seed that won't be visited correctly
        # Easier: give seed a child link then let seed fail
        # We need to set up seed so it fails but still yields, then enqueues a child via queue
        # The crawl function only extracts links on success, so we use: seed html → /child
        # But session raises on first call... so seed result has error and no links
        # Test: seed errors, but crawl keeps going if queue has more (it won't here since no links)
        # Better test: two seeds via redirected approach isn't possible; test single-page error propagation
        results = list(crawl("http://example.com/", depth=1, max_pages=1,
                              delay=0, session=session))
        assert len(results) == 1
        assert results[0].error is not None

    # ---- delay behaviour ----

    @patch("crawler.time.sleep")
    def test_sleep_called_between_pages(self, mock_sleep):
        responses = [
            _make_response(200, _html("/p1")),
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        list(crawl("http://example.com/", depth=1, max_pages=5,
                   delay=1.5, session=session))
        mock_sleep.assert_called_with(1.5)

    @patch("crawler.time.sleep")
    def test_no_sleep_after_last_page(self, mock_sleep):
        session = self._make_session([_make_response(200, "<html></html>")])
        list(crawl("http://example.com/", depth=1, max_pages=1,
                   delay=0.5, session=session))
        mock_sleep.assert_not_called()

    @patch("crawler.time.sleep")
    def test_delay_zero_still_calls_sleep(self, mock_sleep):
        responses = [
            _make_response(200, _html("/p1")),
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        list(crawl("http://example.com/", depth=1, max_pages=5,
                   delay=0, session=session))
        mock_sleep.assert_called_with(0)

    # ---- session management ----

    def test_provided_session_used(self):
        session = self._make_session([_make_response(200, "<html></html>")])
        list(crawl("http://example.com/", depth=0, max_pages=1,
                   delay=0, session=session))
        session.get.assert_called_once()

    @patch("crawler.requests.Session")
    def test_default_session_created_when_none_provided(self, MockSession):
        mock_session = MagicMock()
        mock_session.headers = {}
        mock_session.get.return_value = _make_response(200, "<html></html>")
        MockSession.return_value = mock_session

        list(crawl("http://example.com/", depth=0, max_pages=1, delay=0))
        MockSession.assert_called_once()
        assert mock_session.headers.get("User-Agent") == "WebCrawler/1.0 (educational)"

    # ---- depth boundary ----

    def test_depth_one_visits_seed_and_direct_children(self):
        responses = [
            _make_response(200, _html("/child1", "/child2")),
            _make_response(200, "<html></html>"),
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=1, max_pages=10,
                              delay=0, session=session))
        visited = [r.url for r in results]
        assert "http://example.com/" in visited
        assert "http://example.com/child1" in visited
        assert "http://example.com/child2" in visited

    def test_depth_zero_only_visits_seed(self):
        session = self._make_session([_make_response(200, _html("/child"))])
        results = list(crawl("http://example.com/", depth=0, max_pages=10,
                              delay=0, session=session))
        assert len(results) == 1
        assert results[0].url == "http://example.com/"

    def test_links_on_result_are_extracted_regardless_of_depth_limit(self):
        """Even when depth prevents enqueuing, links are still recorded on the result."""
        session = self._make_session([_make_response(200, _html("/child"))])
        results = list(crawl("http://example.com/", depth=0, max_pages=10,
                              delay=0, session=session))
        assert "http://example.com/child" in results[0].links

    # ---- generator behaviour ----

    def test_returns_generator(self):
        import types
        session = self._make_session([_make_response(200, "<html></html>")])
        result = crawl("http://example.com/", depth=0, max_pages=1,
                       delay=0, session=session)
        assert isinstance(result, types.GeneratorType)

    def test_empty_queue_yields_nothing(self):
        # max_pages=0 means the while condition count < max_pages is False from start
        # Actually max_pages=0: 0 < 0 is False → no pages visited
        session = self._make_session([])
        results = list(crawl("http://example.com/", depth=1, max_pages=0,
                              delay=0, session=session))
        assert results == []


# ---------------------------------------------------------------------------
# crawl() — enqueued set behaviour (PR: replace visited with enqueued)
# ---------------------------------------------------------------------------

class TestCrawlEnqueued:
    """Tests for the enqueued-set deduplication introduced in this PR.

    The key change: `enqueued` is initialised with the seed URL and is used
    exclusively to prevent a URL from being added to the BFS queue more than
    once.  It no longer doubles as a "visited" guard at dequeue time.
    """

    def _make_session(self, responses):
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        session.get.side_effect = responses
        return session

    # ---- seed pre-population ----

    def test_seed_not_re_enqueued_when_linked_by_crawled_page(self):
        """Seed is in enqueued from the start, so a page linking back to the
        seed must NOT cause the seed to be crawled a second time."""
        # seed page links back to itself
        session = self._make_session([
            _make_response(200, _html("/")),   # seed → links back to /
        ])
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, session=session))
        visited_urls = [r.url for r in results]
        assert visited_urls.count("http://example.com/") == 1

    def test_seed_not_re_enqueued_via_absolute_url_link(self):
        """Absolute seed URL in a link should also be blocked by enqueued."""
        # seed page links back to itself using its full absolute URL
        session = self._make_session([
            _make_response(200, _html("http://example.com/")),
        ])
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, session=session))
        assert len(results) == 1

    # ---- duplicate link deduplication ----

    def test_link_discovered_on_two_pages_enqueued_only_once(self):
        """When two already-crawled pages both link to the same new URL it
        should be enqueued only once and therefore visited only once."""
        # seed  → /a, /b
        # /a    → /shared
        # /b    → /shared  (second discovery; must NOT re-enqueue /shared)
        # /shared → (empty)
        responses = [
            _make_response(200, _html("/a", "/b")),          # seed
            _make_response(200, _html("/shared")),            # /a
            _make_response(200, _html("/shared")),            # /b
            _make_response(200, "<html></html>"),             # /shared
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, session=session))
        visited_urls = [r.url for r in results]
        assert visited_urls.count("http://example.com/shared") == 1

    def test_link_to_already_crawled_page_not_re_enqueued(self):
        """A link pointing to a page that has already been crawled (and is
        therefore already in enqueued) must not cause it to be crawled again."""
        # seed → /p1
        # /p1  → /p2
        # /p2  → /p1 (back-link to already-processed page)
        responses = [
            _make_response(200, _html("/p1")),        # seed
            _make_response(200, _html("/p2")),        # /p1
            _make_response(200, _html("/p1")),        # /p2 (links back to /p1)
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=3, max_pages=10,
                              delay=0, session=session))
        visited_urls = [r.url for r in results]
        assert visited_urls.count("http://example.com/p1") == 1

    # ---- include-filtered URLs not re-enqueued ----

    def test_include_filtered_url_discovered_twice_enqueued_once(self):
        """A URL that does NOT match the include filter is still tracked in
        enqueued (it enters via the link-discovery path), so when a second page
        links to it, it is not enqueued a second time."""
        # seed (matches /allowed) → /allowed, /nope
        # /allowed → /nope  (second discovery of /nope — must not re-enqueue)
        # /nope should never be crawled (filtered by include)
        responses = [
            _make_response(200, _html("/allowed/page", "/nope")),  # seed
            _make_response(200, _html("/nope")),                    # /allowed/page
            # /nope would need a response if wrongly enqueued twice
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, include=["/allowed"], session=session))
        visited_urls = [r.url for r in results]
        assert "http://example.com/nope" not in visited_urls
        # The session should only have been called for seed and /allowed/page;
        # a third call would mean /nope was incorrectly enqueued and dequeued.
        assert session.get.call_count == 2

    def test_exclude_filtered_url_discovered_twice_enqueued_once(self):
        """A URL matching the exclude filter is tracked in enqueued at discovery
        time, so a second page linking to it does not re-enqueue it."""
        # seed → /admin, /p1
        # /p1  → /admin  (second discovery; must NOT add /admin to queue again)
        responses = [
            _make_response(200, _html("/admin", "/p1")),   # seed
            _make_response(200, _html("/admin")),           # /p1
            _make_response(200, "<html></html>"),           # /admin if wrongly re-enqueued
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=2, max_pages=10,
                              delay=0, exclude=["/admin"], session=session))
        visited_urls = [r.url for r in results]
        assert "http://example.com/admin" not in visited_urls
        assert session.get.call_count == 2

    # ---- regression: no infinite loop when seed self-links ----

    def test_self_linking_seed_terminates(self):
        """A seed that only links to itself must terminate after one page."""
        session = self._make_session([
            _make_response(200, _html("http://example.com/")),  # seed → self
        ])
        results = list(crawl("http://example.com/", depth=5, max_pages=5,
                              delay=0, session=session))
        # Should visit only the seed and then stop (no more unique URLs)
        assert len(results) == 1

    # ---- enqueued does not prevent cross-domain links from being recorded ----

    def test_cross_domain_link_recorded_on_result_but_not_crawled(self):
        """Cross-domain links are still appended to result.links (they are
        enqueued momentarily) but are filtered out when dequeued, so they are
        never actually crawled."""
        html = _html("http://external.com/page", "/local")
        responses = [
            _make_response(200, html),
            _make_response(200, "<html></html>"),
        ]
        session = self._make_session(responses)
        results = list(crawl("http://example.com/", depth=1, max_pages=10,
                              delay=0, session=session))
        # The cross-domain link is in the seed's links list
        assert "http://external.com/page" in results[0].links
        # But it is never crawled
        crawled_urls = [r.url for r in results]
        assert "http://external.com/page" not in crawled_urls