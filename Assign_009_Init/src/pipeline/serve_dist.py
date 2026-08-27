"""Serve dist/ over HTTP so widgets can fetch JSON (file:// blocks fetch)."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
DEFAULT_PORT = 8765


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Session 9 dist/ for local widgets")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-open", action="store_true", help="Do not open browser")
    args = parser.parse_args()

    if not (DIST / "index.html").is_file():
        raise SystemExit(
            f"Missing {DIST / 'index.html'}. Run first:\n"
            "  python -m src.pipeline.run_all\n"
            "  python tests/build_dist.py"
        )
    if not (DIST / "data" / "widget_weights.json").is_file():
        raise SystemExit(
            f"Missing widget weights. Run:\n"
            "  python -m src.pipeline.export_widget\n"
            "  python tests/build_dist.py"
        )

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(DIST), **kw)

    url = f"http://127.0.0.1:{args.port}/"
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as httpd:
        print(f"Serving {DIST}")
        print(f"Open: {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_open:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
