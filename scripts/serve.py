#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class NoCacheHTTPRequestHandler(SimpleHTTPRequestHandler):
    repo_root: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/semantic-clusters":
            self.serve_semantic_clusters(parsed.query)
            return
        if parsed.path == "/api/query-expansion":
            self.serve_query_expansion(parsed.query)
            return
        super().do_GET()

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_query_expansion(self, query: str) -> None:
        params = parse_qs(query)
        descriptor_query = (params.get("q") or [""])[0].strip()
        mode = (params.get("mode") or ["unit"])[0].strip()
        if not descriptor_query:
            self.send_json(400, {"error": "missing q"})
            return
        if mode not in {"unit", "favorites_weighted"}:
            self.send_json(400, {"error": "mode must be unit or favorites_weighted"})
            return

        descriptor_source = self.repo_root / "site" / "mvp_visualizer" / f"rankings_{mode}.json"
        if not descriptor_source.exists():
            self.send_json(500, {"error": f"missing descriptor source: {descriptor_source}"})
            return

        python = self.repo_root / ".venv" / "bin" / "python"
        if not python.exists():
            python = Path(sys.executable)
        command = [
            str(python),
            "scripts/cache_query_expansions.py",
            descriptor_query,
            "--descriptor-source",
            str(descriptor_source),
        ]
        try:
            subprocess.run(
                command,
                cwd=self.repo_root,
                check=True,
                text=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as error:
            self.send_json(
                500,
                {
                    "error": "query expansion failed",
                    "stdout": error.stdout[-4000:],
                    "stderr": error.stderr[-4000:],
                },
            )
            return
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "query expansion timed out"})
            return

        query_key = " ".join(descriptor_query.strip().lower().replace("_", " ").split())
        cache_path = self.repo_root / "run" / "query_expansions" / "query_expansions.jsonl"
        best = None
        if cache_path.exists():
            with cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("query") == query_key and row.get("descriptor_source") == str(descriptor_source):
                        best = row
        if not best:
            self.send_json(500, {"error": "query expansion completed but cache row was not found"})
            return
        self.send_json(
            200,
            {
                "query": best.get("query"),
                "expanded": best.get("expanded") or [],
                "model": best.get("model"),
                "model_digest": best.get("model_digest"),
                "temperature": best.get("temperature"),
                "seed": best.get("seed"),
                "think": best.get("think"),
                "prompt_version": best.get("prompt_version"),
                "descriptor_source": best.get("descriptor_source"),
                "descriptor_sha256": best.get("descriptor_sha256"),
            },
        )

    def serve_semantic_clusters(self, query: str) -> None:
        params = parse_qs(query)
        seiyuu = (params.get("q") or [""])[0].strip()
        try:
            k = int((params.get("k") or ["2"])[0])
        except ValueError:
            self.send_json(400, {"error": "k must be one of 1, 2, 3, or 4"})
            return
        if not seiyuu:
            self.send_json(400, {"error": "missing q"})
            return
        if k not in {1, 2, 3, 4}:
            self.send_json(400, {"error": "k must be one of 1, 2, 3, or 4"})
            return

        python = self.repo_root / ".venv" / "bin" / "python"
        if not python.exists():
            python = Path(sys.executable)
        command = [
            str(python),
            "scripts/seiyuu_character_semantic_clusters.py",
            seiyuu,
            "--k",
            str(k),
            "--row-weight",
            "sqrt_log_combined_favourites",
            "--shared-role-weight",
            "inverse_sqrt",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                check=True,
                text=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as error:
            self.send_json(
                500,
                {
                    "error": "semantic clustering failed",
                    "stdout": error.stdout[-4000:],
                    "stderr": error.stderr[-4000:],
                },
            )
            return
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "semantic clustering timed out"})
            return

        output_path = None
        for line in completed.stdout.splitlines():
            if line.startswith("wrote "):
                output_path = self.repo_root / line.removeprefix("wrote ").strip()
                break
        if output_path is None or not output_path.exists():
            self.send_json(
                500,
                {
                    "error": "semantic clustering completed but output JSON was not found",
                    "stdout": completed.stdout[-4000:],
                },
            )
            return
        self.send_json(200, json.loads(output_path.read_text(encoding="utf-8")))

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SeiyuuRoleProfiler static page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, default=Path("site"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    handler = functools.partial(NoCacheHTTPRequestHandler, directory=str(args.directory))
    handler.func.repo_root = repo_root
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving SeiyuuRoleProfiler on http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
