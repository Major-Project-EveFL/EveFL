"""
A small web server for the EveFL demonstration app, written with the standard library only (no new dependency).

It serves the static front end in `evefl/webapp/static/` and a JSON API (`evefl/webapp/api.py`). Because it is shown on a laptop in
front of an audience, it is deliberately careful: it binds to the loopback interface only, refuses any Host header that is not a
loopback name (so a web page on another site cannot reach it through DNS rebinding), refuses path traversal, limits the size of a
request body, sends a restrictive Content-Security-Policy, and never puts a stack trace in a response.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from evefl import selfcheck
from evefl.webapp import api

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 20 * 1024 * 1024            # the largest accepted request body (an uploaded picture)
STATIC_ROOT = Path(__file__).resolve().parent / "static"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_PORT = 8765

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".ico": "image/x-icon", ".txt": "text/plain; charset=utf-8",
}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"),
}


@dataclass
class Request:
    method: str
    path: str
    query: Dict[str, List[str]]
    body: bytes
    server: "DemoServer"


@dataclass
class Response:
    status: int = 200
    content_type: str = "application/json; charset=utf-8"
    body: bytes = b""
    stream: Optional[Iterator[bytes]] = None
    headers: Dict[str, str] = field(default_factory=dict)


def json_response(data: Any, status: int = 200) -> Response:
    return Response(status=status, body=json.dumps(api.to_jsonable(data), allow_nan=False).encode("utf-8"))


def error_response(status: int, message: str) -> Response:
    return json_response({"error": message}, status)


def _sse(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(api.to_jsonable(data), allow_nan=False)}\n\n".encode("utf-8")


# ---- routes -----------------------------------------------------------------------------------------------------------

def _summary(_request: Request) -> Response:
    return json_response(api.summary())


def _check_plan(request: Request) -> Response:
    return json_response(api.check_plan(request.server.checkpoint_dir))


def _check_stream(request: Request) -> Response:
    def events() -> Iterator[bytes]:
        for kind, data in api.check_events(request.server.checkpoint_dir):
            yield _sse(kind, data)

    return Response(content_type="text/event-stream; charset=utf-8", stream=events())


ROUTES: Dict[Tuple[str, str], Callable[[Request], Response]] = {
    ("GET", "/api/summary"): _summary,
    ("GET", "/api/selfcheck/plan"): _check_plan,
    ("GET", "/api/selfcheck/stream"): _check_stream,
}


def static_response(url_path: str) -> Response:
    """Serve a file from the static folder; anything that is not a plain file inside it is a 404."""
    relative = "index.html" if url_path == "/" else unquote(url_path[len("/static/"):])
    parts = relative.split("/")
    if (not relative or "\x00" in relative or "\\" in relative or relative.startswith("/") or any(p in ("", ".", "..") for p in parts)):
        return error_response(404, "not found")
    candidate = (STATIC_ROOT / relative).resolve()
    if STATIC_ROOT.resolve() not in candidate.parents or not candidate.is_file():
        return error_response(404, "not found")
    content_type = CONTENT_TYPES.get(candidate.suffix.lower())
    if content_type is None:
        return error_response(404, "not found")
    return Response(content_type=content_type, body=candidate.read_bytes())


# ---- server -----------------------------------------------------------------------------------------------------------

class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Tuple[str, int], checkpoint_dir: Path) -> None:
        super().__init__(address, Handler)
        self.checkpoint_dir = Path(checkpoint_dir)
        port = self.server_address[1]
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}


class Handler(BaseHTTPRequestHandler):
    server: DemoServer
    server_version = "EveFLDemo"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature fixed by the base class
        logger.debug("%s %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._send(error_response(405, "method not allowed"))

    do_DELETE = do_PATCH = do_PUT

    def _dispatch(self, method: str) -> None:
        if self.headers.get("Host", "") not in self.server.allowed_hosts:
            self._send(error_response(403, "forbidden host"))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(error_response(400, "bad Content-Length"))
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._send(error_response(413, "request body too large"))
            return
        split = urlsplit(self.path)
        body = self.rfile.read(length) if length else b""
        request = Request(method, split.path, parse_qs(split.query), body, self.server)
        handler = ROUTES.get((method, split.path))
        try:
            if handler is not None:
                response = handler(request)
            elif method == "GET" and (split.path == "/" or split.path.startswith("/static/")):
                response = static_response(split.path)
            else:
                response = error_response(404, "not found")
        except Exception:  # noqa: BLE001 - a handler bug must become a clean 500, never a trace on the screen
            logger.exception("handler failed for %s %s", method, split.path)
            response = error_response(500, "internal error; see the server log")
        self._send(response)

    def _send(self, response: Response) -> None:
        try:
            self.send_response(response.status)
            self.send_header("Content-Type", response.content_type)
            if response.stream is None:
                self.send_header("Content-Length", str(len(response.body)))
            for name, value in {**SECURITY_HEADERS, **response.headers}.items():
                self.send_header(name, value)
            self.end_headers()
            if response.stream is None:
                self.wfile.write(response.body)
                return
            for chunk in response.stream:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("client went away")


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT, checkpoint_dir: Optional[Path] = None) -> DemoServer:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"the demo server binds to the loopback interface only, got {host!r}")
    return DemoServer((host, port), checkpoint_dir if checkpoint_dir is not None else selfcheck.default_checkpoint_dir())


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    httpd = make_server(host, port)
    url = f"http://localhost:{httpd.server_address[1]}/"
    threading.Thread(target=selfcheck.warm_up, name="evefl-warm-up", daemon=True).start()
    logger.info("EveFL demo running at %s (Ctrl+C to stop)", url)
    if open_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        httpd.server_close()
