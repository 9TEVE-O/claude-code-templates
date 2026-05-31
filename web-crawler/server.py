#!/usr/bin/env python3
"""Flask web server — streams crawl results via Server-Sent Events on port 3000."""
import json
from flask import Flask, render_template, request, Response, stream_with_context
from crawler import crawl

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/crawl")
def start_crawl():
    url = request.args.get("url", "").strip()
    if not url:
        return Response("data: {\"error\": \"No URL provided\", \"fatal\": true}\n\n",
                        mimetype="text/event-stream")

    depth = max(1, min(int(request.args.get("depth", 2)), 10))
    pages = max(1, min(int(request.args.get("pages", 50)), 500))
    delay = max(0.0, min(float(request.args.get("delay", 0.5)), 10.0))
    include = [p.strip() for p in request.args.get("include", "").split(",") if p.strip()]
    exclude = [p.strip() for p in request.args.get("exclude", "").split(",") if p.strip()]

    def generate():
        try:
            count = 0
            for r in crawl(
                seed=url,
                depth=depth,
                max_pages=pages,
                delay=delay,
                include=include,
                exclude=exclude,
            ):
                count += 1
                payload = {
                    "count": count,
                    "max": pages,
                    "url": r.url,
                    "status": r.status,
                    "links": len(r.links),
                    "error": r.error,
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'done': True, 'total': count})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'fatal': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print("Starting web crawler UI at http://localhost:3000")
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)
