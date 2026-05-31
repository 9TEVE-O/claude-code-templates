"""Tests for web-crawler/server.py"""
import json
import sys
import os
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler import CrawlResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(data: bytes) -> list:
    """Parse raw SSE bytes into a list of decoded JSON dicts."""
    events = []
    for line in data.decode("utf-8").split("\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def _make_result(url="http://example.com/", status=200, links=None, error=None):
    r = CrawlResult(url=url, status=status, links=links or [], error=error)
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Flask test client with crawl() mocked out."""
    # Import server after sys.path is set
    import server
    server.app.config["TESTING"] = True
    with server.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexRoute:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_returns_html(self, client):
        resp = client.get("/")
        assert b"<!DOCTYPE html>" in resp.data or b"<!doctype html>" in resp.data.lower()

    def test_content_type_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.content_type


# ---------------------------------------------------------------------------
# GET /crawl — missing URL
# ---------------------------------------------------------------------------

class TestCrawlMissingUrl:
    def test_missing_url_returns_event_stream(self, client):
        resp = client.get("/crawl")
        assert resp.content_type == "text/event-stream"

    def test_missing_url_returns_fatal_error(self, client):
        resp = client.get("/crawl")
        events = _parse_sse(resp.data)
        assert len(events) == 1
        assert events[0].get("fatal") is True
        assert "error" in events[0]

    def test_blank_url_returns_fatal_error(self, client):
        resp = client.get("/crawl?url=   ")
        events = _parse_sse(resp.data)
        assert events[0].get("fatal") is True

    def test_missing_url_error_message(self, client):
        resp = client.get("/crawl")
        events = _parse_sse(resp.data)
        assert "No URL provided" in events[0]["error"]


# ---------------------------------------------------------------------------
# GET /crawl — parameter parsing and clamping
# ---------------------------------------------------------------------------

class TestCrawlParameterParsing:
    def _crawl_call_kwargs(self, client, **query_params):
        """Run /crawl and capture the kwargs passed to crawl()."""
        qs = "&".join(f"{k}={v}" for k, v in query_params.items())
        url = f"/crawl?url=http://example.com/&{qs}"
        captured = {}

        def fake_crawl(**kwargs):
            captured.update(kwargs)
            yield _make_result()

        with patch("server.crawl", side_effect=fake_crawl):
            client.get(url)
        return captured

    # depth clamping
    def test_depth_default_is_2(self, client):
        kw = self._crawl_call_kwargs(client)
        assert kw["depth"] == 2

    def test_depth_clamped_to_minimum_1(self, client):
        kw = self._crawl_call_kwargs(client, depth=0)
        assert kw["depth"] == 1

    def test_depth_clamped_to_maximum_10(self, client):
        kw = self._crawl_call_kwargs(client, depth=99)
        assert kw["depth"] == 10

    def test_depth_within_range_unchanged(self, client):
        kw = self._crawl_call_kwargs(client, depth=5)
        assert kw["depth"] == 5

    def test_depth_boundary_minimum_exactly_1(self, client):
        kw = self._crawl_call_kwargs(client, depth=1)
        assert kw["depth"] == 1

    def test_depth_boundary_maximum_exactly_10(self, client):
        kw = self._crawl_call_kwargs(client, depth=10)
        assert kw["depth"] == 10

    # pages clamping
    def test_pages_default_is_50(self, client):
        kw = self._crawl_call_kwargs(client)
        assert kw["max_pages"] == 50

    def test_pages_clamped_to_minimum_1(self, client):
        kw = self._crawl_call_kwargs(client, pages=0)
        assert kw["max_pages"] == 1

    def test_pages_clamped_to_maximum_500(self, client):
        kw = self._crawl_call_kwargs(client, pages=9999)
        assert kw["max_pages"] == 500

    def test_pages_within_range_unchanged(self, client):
        kw = self._crawl_call_kwargs(client, pages=100)
        assert kw["max_pages"] == 100

    # delay clamping
    def test_delay_default_is_0_5(self, client):
        kw = self._crawl_call_kwargs(client)
        assert kw["delay"] == pytest.approx(0.5)

    def test_delay_clamped_to_minimum_0(self, client):
        kw = self._crawl_call_kwargs(client, delay=-1.0)
        assert kw["delay"] == pytest.approx(0.0)

    def test_delay_clamped_to_maximum_10(self, client):
        kw = self._crawl_call_kwargs(client, delay=99.9)
        assert kw["delay"] == pytest.approx(10.0)

    def test_delay_within_range_unchanged(self, client):
        kw = self._crawl_call_kwargs(client, delay=2.5)
        assert kw["delay"] == pytest.approx(2.5)

    # include/exclude
    def test_include_single_pattern(self, client):
        kw = self._crawl_call_kwargs(client, include="/docs/")
        assert kw["include"] == ["/docs/"]

    def test_include_multiple_patterns_comma_separated(self, client):
        kw = self._crawl_call_kwargs(client, include="/docs/,/blog/")
        assert kw["include"] == ["/docs/", "/blog/"]

    def test_include_strips_whitespace(self, client):
        kw = self._crawl_call_kwargs(client, **{"include": " /docs/ , /blog/ "})
        assert kw["include"] == ["/docs/", "/blog/"]

    def test_include_empty_string_gives_empty_list(self, client):
        kw = self._crawl_call_kwargs(client, include="")
        assert kw["include"] == []

    def test_exclude_single_pattern(self, client):
        kw = self._crawl_call_kwargs(client, exclude="/admin/")
        assert kw["exclude"] == ["/admin/"]

    def test_exclude_multiple_patterns(self, client):
        kw = self._crawl_call_kwargs(client, exclude="/admin/,/tag/")
        assert kw["exclude"] == ["/admin/", "/tag/"]

    def test_exclude_empty_string_gives_empty_list(self, client):
        kw = self._crawl_call_kwargs(client, exclude="")
        assert kw["exclude"] == []

    def test_seed_url_passed_to_crawl(self, client):
        kw = self._crawl_call_kwargs(client)
        assert kw["seed"] == "http://example.com/"


# ---------------------------------------------------------------------------
# GET /crawl — SSE response format and content
# ---------------------------------------------------------------------------

class TestCrawlSseResponse:
    @pytest.fixture
    def single_page_client(self, client):
        result = _make_result(url="http://example.com/", status=200,
                              links=["http://example.com/a", "http://example.com/b"])
        with patch("server.crawl", return_value=iter([result])):
            yield client

    def test_content_type_is_event_stream(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        assert resp.content_type == "text/event-stream"

    def test_cache_control_header(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        assert resp.headers.get("Cache-Control") == "no-cache"

    def test_x_accel_buffering_header(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        assert resp.headers.get("X-Accel-Buffering") == "no"

    def test_events_are_valid_json(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        assert len(events) >= 1

    def test_page_event_fields(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = events[0]
        assert "count" in page_event
        assert "max" in page_event
        assert "url" in page_event
        assert "status" in page_event
        assert "links" in page_event
        assert "error" in page_event

    def test_page_event_count_increments(self, client):
        results = [
            _make_result(url="http://example.com/p1", status=200),
            _make_result(url="http://example.com/p2", status=200),
        ]
        with patch("server.crawl", return_value=iter(results)):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_events = [e for e in events if "count" in e]
        assert page_events[0]["count"] == 1
        assert page_events[1]["count"] == 2

    def test_page_event_url_matches_result(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "url" in e][0]
        assert page_event["url"] == "http://example.com/"

    def test_page_event_status_matches_result(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "status" in e][0]
        assert page_event["status"] == 200

    def test_page_event_links_is_count_not_list(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "links" in e][0]
        assert page_event["links"] == 2  # len(["http://example.com/a", "http://example.com/b"])

    def test_page_event_error_is_none_when_no_error(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "error" in e and "url" in e][0]
        assert page_event["error"] is None

    def test_page_event_error_propagated(self, client):
        result = _make_result(url="http://example.com/", status=0, error="Connection refused")
        with patch("server.crawl", return_value=iter([result])):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "url" in e][0]
        assert page_event["error"] == "Connection refused"

    def test_done_event_emitted_at_end(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        done_events = [e for e in events if e.get("done") is True]
        assert len(done_events) == 1

    def test_done_event_total_is_correct(self, client):
        results = [
            _make_result(url="http://example.com/p1", status=200),
            _make_result(url="http://example.com/p2", status=200),
            _make_result(url="http://example.com/p3", status=200),
        ]
        with patch("server.crawl", return_value=iter(results)):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        done_event = next(e for e in events if e.get("done") is True)
        assert done_event["total"] == 3

    def test_page_event_max_reflects_pages_param(self, client):
        result = _make_result()
        with patch("server.crawl", return_value=iter([result])):
            resp = client.get("/crawl?url=http://example.com/&pages=42")
        events = _parse_sse(resp.data)
        page_event = [e for e in events if "max" in e][0]
        assert page_event["max"] == 42

    def test_sse_format_uses_data_prefix(self, single_page_client):
        resp = single_page_client.get("/crawl?url=http://example.com/")
        text = resp.data.decode("utf-8")
        for line in text.strip().split("\n\n"):
            if line.strip():
                assert line.startswith("data: ")

    def test_zero_pages_produces_only_done_event(self, client):
        with patch("server.crawl", return_value=iter([])):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        assert len(events) == 1
        assert events[0].get("done") is True
        assert events[0]["total"] == 0


# ---------------------------------------------------------------------------
# GET /crawl — exception handling inside generate()
# ---------------------------------------------------------------------------

class TestCrawlExceptionHandling:
    def test_exception_in_generate_yields_fatal_event(self, client):
        def exploding_crawl(**kwargs):
            raise RuntimeError("Unexpected crash")
            yield  # make it a generator

        with patch("server.crawl", side_effect=exploding_crawl):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        fatal_events = [e for e in events if e.get("fatal") is True]
        assert len(fatal_events) == 1

    def test_exception_message_in_fatal_event(self, client):
        def exploding_crawl(**kwargs):
            raise ValueError("Something went wrong")
            yield

        with patch("server.crawl", side_effect=exploding_crawl):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        fatal_event = next(e for e in events if e.get("fatal") is True)
        assert "Something went wrong" in fatal_event["error"]

    def test_exception_mid_stream_yields_fatal_event(self, client):
        def partial_then_fail(**kwargs):
            yield _make_result(url="http://example.com/p1", status=200)
            raise ConnectionError("dropped")

        with patch("server.crawl", side_effect=partial_then_fail):
            resp = client.get("/crawl?url=http://example.com/")
        events = _parse_sse(resp.data)
        fatal_events = [e for e in events if e.get("fatal") is True]
        assert len(fatal_events) == 1
        # The first page event should still have been emitted
        page_events = [e for e in events if "url" in e]
        assert len(page_events) == 1
