#!/usr/bin/env python3
"""Zero-dependency local server for browsing GPT-VLN episode replays."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


WEB_ROOT = Path(__file__).resolve().parent


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def safe_child(root: Path, relative: str) -> Path | None:
    try:
        candidate = (root / unquote(relative)).resolve()
        candidate.relative_to(root)
        return candidate
    except (OSError, ValueError):
        return None


def episode_summary(outputs_root: Path, episode_dir: Path) -> dict[str, object] | None:
    result_path = episode_dir / "result.json"
    index_path = episode_dir / "raw_action_frames" / "index.json"
    trace_path = episode_dir / "trace.json"
    if not (result_path.is_file() and index_path.is_file()):
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        frames = json.loads(index_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else []
    except (OSError, json.JSONDecodeError):
        return None

    relative = episode_dir.relative_to(outputs_root).as_posix()
    return {
        "key": relative,
        "run": episode_dir.parent.relative_to(outputs_root).as_posix(),
        "episode_id": str(result.get("episode_id", episode_dir.name.removeprefix("episode_"))),
        "instruction": result.get("instruction", ""),
        "frames": len(frames),
        "turns": result.get("turns", len(trace)),
        "success": result.get("metrics", {}).get("success"),
    }


class ReplayHandler(BaseHTTPRequestHandler):
    outputs_root: Path

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json_bytes(value), "application/json; charset=utf-8", status)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_bytes(body, content_type)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/episodes":
            episodes = []
            if self.outputs_root.is_dir():
                for result_path in self.outputs_root.glob("**/episode_*/result.json"):
                    summary = episode_summary(self.outputs_root, result_path.parent)
                    if summary:
                        episodes.append(summary)
            episodes.sort(key=lambda item: (str(item["run"]), str(item["episode_id"])))
            self.send_json({"outputs_root": str(self.outputs_root), "episodes": episodes})
            return

        if path.startswith("/api/episode/"):
            key = path.removeprefix("/api/episode/")
            episode_dir = safe_child(self.outputs_root, key)
            if not episode_dir or not episode_dir.is_dir():
                self.send_json({"error": "Episode not found"}, HTTPStatus.NOT_FOUND)
                return
            try:
                result = json.loads((episode_dir / "result.json").read_text(encoding="utf-8"))
                frames = json.loads(
                    (episode_dir / "raw_action_frames" / "index.json").read_text(encoding="utf-8")
                )
                trace_path = episode_dir / "trace.json"
                trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else []
            except (OSError, json.JSONDecodeError) as exc:
                self.send_json({"error": f"Invalid episode data: {exc}"}, HTTPStatus.UNPROCESSABLE_ENTITY)
                return

            base = "/data/" + quote(key, safe="/")
            assets = {}
            for name in ("top_map_gt_pred.png",):
                if (episode_dir / name).is_file():
                    assets[name] = f"{base}/{name}"
            for png in sorted(episode_dir.glob("turn_*.png")):
                assets[png.name] = f"{base}/{png.name}"
            for frame in frames:
                frame["url"] = f"{base}/{quote(str(frame.get('path', '')), safe='/')}"
            self.send_json({"result": result, "frames": frames, "trace": trace, "assets": assets})
            return

        if path.startswith("/data/"):
            file_path = safe_child(self.outputs_root, path.removeprefix("/data/"))
            if not file_path:
                self.send_error(HTTPStatus.FORBIDDEN)
            else:
                self.send_file(file_path)
            return

        relative = "index.html" if path == "/" else path.lstrip("/")
        static_path = safe_child(WEB_ROOT, relative)
        if not static_path:
            self.send_error(HTTPStatus.FORBIDDEN)
        else:
            self.send_file(static_path)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play GPT-VLN trajectories in a browser")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=WEB_ROOT.parent / "outputs",
        help="Directory containing run/episode_* folders (default: ./outputs)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    outputs_root = args.outputs.expanduser().resolve()
    handler = type("ConfiguredReplayHandler", (ReplayHandler,), {"outputs_root": outputs_root})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"GPT-VLN replay: http://{args.host}:{args.port}")
    print(f"Episode data: {outputs_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
