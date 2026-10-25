"""
The demonstration web app's server (`evefl/webapp/server.py`) and its API layer (`evefl/webapp/api.py`).

It uses only the standard library, binds to the loopback interface, and is meant to be shown in front of an audience, so the tests
cover the ordinary behaviour (pages, JSON, the live check as a stream) and the safety properties that matter for a local server:
no path traversal, no foreign Host header (DNS rebinding), a body size limit, no stack trace in an error response.
"""

import http.client
import json
import threading

import pytest

from evefl import showcase
from evefl.webapp import api, server


@pytest.fixture(scope="module")
def running(tmp_path_factory):
    httpd = server.make_server(port=0, checkpoint_dir=tmp_path_factory.mktemp("no_checkpoints"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _request(httpd, method, path, *, body=None, headers=None):
    host, port = httpd.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=60)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def _json(httpd, path):
    status, headers, body = _request(httpd, "GET", path)
    assert headers["Content-Type"].startswith("application/json"), headers
    return status, json.loads(body)


# ---- pages and static files -----------------------------------------------------------------------------------------

def test_the_home_page_is_served_with_security_headers(running):
    status, headers, body = _request(running, "GET", "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    assert b"EveFL" in body and b"/static/js/app.js" in body
    assert headers["X-Content-Type-Options"] == "nosniff"
    csp = headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "script-src 'self'" in csp and "http" not in csp.replace("'self'", "")


def test_static_files_are_served_with_the_right_type(running):
    status, headers, body = _request(running, "GET", "/static/css/app.css")
    assert status == 200 and headers["Content-Type"].startswith("text/css") and len(body) > 500
    status, headers, _ = _request(running, "GET", "/static/js/app.js")
    assert status == 200 and "javascript" in headers["Content-Type"]


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/server.py", "/static/..%2fapi.py", "/static/css/../../api.py",
                                  "/static//etc/passwd", "/static/..\\server.py", "/static/css/%00.css"])
def test_path_traversal_is_refused(running, path):
    status, _, body = _request(running, "GET", path)
    assert status in (400, 403, 404)
    assert b"import" not in body and b"def " not in body


def test_an_unknown_route_is_a_json_404_without_a_stack_trace(running):
    status, headers, body = _request(running, "GET", "/api/does-not-exist")
    assert status == 404 and headers["Content-Type"].startswith("application/json")
    assert "Traceback" not in body.decode() and json.loads(body)["error"]


# ---- host and size checks -------------------------------------------------------------------------------------------

def test_a_request_with_a_foreign_host_header_is_refused(running):
    status, _, _ = _request(running, "GET", "/api/summary", headers={"Host": "evil.example.com"})
    assert status == 403


def test_a_request_with_the_loopback_host_name_is_accepted(running):
    port = running.server_address[1]
    status, _, _ = _request(running, "GET", "/api/summary", headers={"Host": f"localhost:{port}"})
    assert status == 200


def test_an_oversized_body_is_refused_before_it_is_read(running):
    status, _, body = _request(running, "POST", "/api/summary", body=b"x", headers={"Content-Length": str(server.MAX_BODY_BYTES + 1)})
    assert status in (405, 413)


def test_the_server_binds_to_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        server.make_server(host="0.0.0.0", port=0)


# ---- the API --------------------------------------------------------------------------------------------------------

def test_summary_carries_the_recorded_headline_numbers(running):
    status, data = _json(running, "/api/summary")
    h = showcase.headline_numbers()
    assert status == 200
    assert data["headline"]["served_model_test_auc"] == h["served_model_test_auc"]
    assert data["thresholds"] == {"secure_max": 0.05, "caution_max": 0.11}
    assert data["headline"]["labels"]["served_model_test_auc"] == "measured"


def test_the_check_plan_lists_every_check_before_any_runs(running):
    status, data = _json(running, "/api/selfcheck/plan")
    names = [c["name"] for c in data["checks"]]
    assert status == 200 and {"controller", "bb84", "link_protection", "secure_aggregation", "model"} <= set(names)
    assert all(c["title"] and c["what"] and c["kind"] for c in data["checks"])


def test_the_live_check_streams_one_event_per_check_then_a_summary(running):
    status, headers, body = _request(running, "GET", "/api/selfcheck/stream")
    assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
    events = [e for e in body.decode().split("\n\n") if e.strip()]
    kinds = [e.splitlines()[0] for e in events]
    assert kinds.count("event: result") >= 7 and kinds[-1] == "event: done"
    results = [json.loads(e.splitlines()[1][len("data: "):]) for e in events if e.startswith("event: result")]
    assert all(r["status"] in ("pass", "skip") for r in results), [(r["name"], r["detail"]) for r in results if r["status"] == "fail"]
    done = json.loads(events[-1].splitlines()[1][len("data: "):])
    assert done["healthy"] is True and done["total"] == len(results)
    model = next(r for r in results if r["name"] == "model")
    assert model["status"] == "skip" and "no checkpoint" in model["detail"].lower()      # the fixture has no checkpoints


def test_json_responses_never_contain_nan(running):
    assert api.to_jsonable({"a": float("nan"), "b": [float("inf"), 1.5]}) == {"a": None, "b": [None, 1.5]}


def test_a_failing_handler_returns_a_json_500_without_the_trace(running, monkeypatch):
    def boom(_request):
        raise RuntimeError("secret internal detail")

    monkeypatch.setitem(server.ROUTES, ("GET", "/api/boom"), boom)
    status, headers, body = _request(running, "GET", "/api/boom")
    assert status == 500 and headers["Content-Type"].startswith("application/json")
    text = body.decode()
    assert "secret internal detail" not in text and "Traceback" not in text and json.loads(text)["error"]
