#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SeiyuuRoleProfiler static page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, default=Path("site"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(args.directory))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving SeiyuuRoleProfiler on http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
