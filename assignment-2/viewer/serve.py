"""Local viewer for artifact trajectories and baseline error labels.

Run from assignment-2:

    uv run python viewer/serve.py
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS = ROOT / "output" / "baseline_predictions.json"
PAGE = Path(__file__).with_name("index.html")
HOST = "127.0.0.1"
PORT = 8765


def load_predictions() -> dict[str, list[dict]]:
    if not PREDICTIONS.is_file():
        return {}
    payload = json.loads(PREDICTIONS.read_text())
    return {
        item["run_id"]: item.get("errors") or []
        for item in payload.get("predictions") or []
    }


def run_ids() -> list[str]:
    return sorted(path.name for path in ARTIFACTS.iterdir() if (path / "result.json").is_file())


def tool_text(message: dict) -> tuple[str, int | None]:
    extra = message.get("extra") or {}
    if "raw_output" in extra:
        return extra.get("raw_output") or "", extra.get("returncode")
    return message.get("content") or "", extra.get("returncode")


def commands_of(message: dict) -> list[str]:
    commands = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or ""
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            commands.append(raw)
            continue
        if isinstance(arguments, dict) and arguments.get("command"):
            commands.append(str(arguments["command"]))
            continue
        commands.append(raw)
    return commands


def turns_of(messages: list[dict]) -> list[dict]:
    turns = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role") or "unknown"
        content = message.get("content") or ""
        output = ""
        returncode = None
        if role == "tool":
            output, returncode = tool_text(message)
            content = ""
        elif role == "exit":
            extra = message.get("extra") or {}
            output = extra.get("exit_status") or ""
            content = ""
        turns.append(
            {
                "index": index,
                "role": role,
                "content": content,
                "commands": commands_of(message),
                "output": output,
                "returncode": returncode,
            }
        )
    return turns


def run_payload(run_id: str, predictions: dict[str, list[dict]]) -> dict | None:
    directory = ARTIFACTS / run_id
    result_path = directory / "result.json"
    if not result_path.is_file():
        return None
    record = json.loads(result_path.read_text())
    figure = directory / "figure.png"
    return {
        "run_id": record.get("run_id", run_id),
        "task_id": record.get("task_id", ""),
        "instructions": (record.get("task") or {}).get("instructions") or "",
        "has_figure": figure.is_file(),
        "errors": predictions.get(run_id, []),
        "turns": turns_of((record.get("trajectory") or {}).get("messages") or []),
    }


def index_payload(predictions: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for run_id in run_ids():
        errors = predictions.get(run_id, [])
        figure = ARTIFACTS / run_id / "figure.png"
        rows.append(
            {
                "run_id": run_id,
                "families": [error.get("family", "") for error in errors],
                "has_figure": figure.is_file(),
            }
        )
    return rows


class ViewerHandler(BaseHTTPRequestHandler):
    predictions = load_predictions()

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self._send(status, body, "application/json")

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._send_json(index_payload(self.predictions))
            return
        if path.startswith("/api/runs/"):
            payload = run_payload(path.removeprefix("/api/runs/"), self.predictions)
            if payload is None:
                self._send_json({"error": "run not found"}, status=404)
                return
            self._send_json(payload)
            return
        if path.startswith("/figures/"):
            self._send_figure(path.removeprefix("/figures/"))
            return
        self._send_json({"error": "not found"}, status=404)

    def _send_figure(self, run_id: str) -> None:
        figure = (ARTIFACTS / run_id / "figure.png").resolve()
        if not figure.is_relative_to(ARTIFACTS.resolve()) or not figure.is_file():
            self._send_json({"error": "figure not found"}, status=404)
            return
        self._send(200, figure.read_bytes(), "image/png")


def main() -> None:
    server = None
    for port in range(PORT, PORT + 20):
        try:
            server = ThreadingHTTPServer((HOST, port), ViewerHandler)
        except OSError:
            continue
        print(f"viewing artifacts at http://{HOST}:{port}", flush=True)
        break
    if server is None:
        raise SystemExit(f"no free port from {PORT} to {PORT + 19}")
    server.serve_forever()


if __name__ == "__main__":
    main()
