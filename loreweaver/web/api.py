"""FastAPI routes for the LoreWeaver debugging UI."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from loreweaver.web.inspectors import DebugInspector
from loreweaver.web.jobs import JobManager, command_specs


NEO4J_CONTAINER = "loreweaver-neo4j"
NEO4J_HTTP_URL = "http://localhost:7474"
NEO4J_BOLT_URL = "bolt://localhost:7687"


def build_api_router(inspector: DebugInspector, jobs: JobManager) -> APIRouter:
    router = APIRouter(prefix="/api")

    def handle(action: Callable[[], Any]) -> Any:
        try:
            return action()
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @router.get("/overview")
    def overview() -> Any:
        return handle(inspector.overview)

    @router.get("/documents")
    def documents() -> Any:
        return handle(inspector.documents)

    @router.get("/windows")
    def windows(
        document_id: str | None = None,
        status: str = "all",
        limit: int = 200,
    ) -> Any:
        return handle(lambda: inspector.windows(document_id=document_id, status=status, limit=limit))

    @router.get("/windows/{window_id}")
    def window_detail(window_id: str) -> Any:
        return handle(lambda: inspector.window_detail(window_id))

    @router.get("/spans")
    def spans(
        document_id: str | None = None,
        locator_status: str = "all",
        query: str = "",
        limit: int = 200,
    ) -> Any:
        return handle(
            lambda: inspector.spans(
                document_id=document_id,
                locator_status=locator_status,
                query=query,
                limit=limit,
            )
        )

    @router.get("/spans/{span_id}")
    def span_detail(span_id: str) -> Any:
        return handle(lambda: inspector.span_detail(span_id))

    @router.get("/graph")
    def graph(document_id: str | None = None) -> Any:
        return handle(lambda: inspector.graph_summary(document_id=document_id))

    @router.get("/reports")
    def reports(limit: int = 80) -> Any:
        return handle(lambda: inspector.reports(limit=limit))

    @router.get("/reports/{name}")
    def report(name: str) -> Any:
        return handle(lambda: inspector.report(name))

    @router.get("/commands")
    def commands() -> Any:
        return command_specs()

    @router.post("/jobs/{command}")
    def start_job(command: str, payload: dict[str, Any] | None = None) -> Any:
        if command not in command_specs():
            raise HTTPException(status_code=404, detail=f"Unsupported command: {command}")
        job = jobs.start(command, payload or {})
        return job.snapshot()

    @router.get("/jobs")
    def list_jobs() -> Any:
        return jobs.list()

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str) -> Any:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return job.snapshot()

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> Any:
        ok = jobs.cancel(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return {"ok": True}

    @router.get("/jobs/{job_id}/events")
    def job_events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        def stream():
            while True:
                try:
                    event = job.events.get(timeout=0.5)
                    yield _sse(event)
                    if event["event"] == "terminal":
                        break
                except Exception:
                    yield ": keep-alive\n\n"
                    if job.status in {"completed", "failed"} and job.events.empty():
                        yield _sse({"event": "terminal", "payload": {"status": job.status}, "time": time.time()})
                        break

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.get("/neo4j/status")
    def neo4j_status() -> Any:
        return _neo4j_status()

    @router.post("/neo4j/start")
    def neo4j_start() -> Any:
        status = _neo4j_status()
        if status["docker_available"] is False:
            raise HTTPException(status_code=500, detail=status["error"] or "docker is not available")
        if status["exists"] and status["running"]:
            return {**status, "started": False, "url": NEO4J_HTTP_URL}
        if status["exists"]:
            result = _run(["docker", "start", NEO4J_CONTAINER], timeout=30)
        else:
            result = _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    NEO4J_CONTAINER,
                    "-p",
                    "7474:7474",
                    "-p",
                    "7687:7687",
                    "-e",
                    "NEO4J_AUTH=neo4j/loreweaver-test",
                    "neo4j:5",
                ],
                timeout=120,
            )
        if result["returncode"] != 0:
            raise HTTPException(status_code=500, detail=result)
        return {**_neo4j_status(), "started": True, "url": NEO4J_HTTP_URL}

    return router


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _neo4j_status() -> dict[str, Any]:
    base = {
        "container": NEO4J_CONTAINER,
        "url": NEO4J_HTTP_URL,
        "bolt_url": NEO4J_BOLT_URL,
        "username": "neo4j",
        "password": "loreweaver-test",
        "docker_available": True,
        "exists": False,
        "running": False,
        "status": "",
        "error": "",
    }
    result = _run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^/{NEO4J_CONTAINER}$",
            "--format",
            "{{.Names}}\t{{.Status}}",
        ],
        timeout=10,
    )
    if result["returncode"] != 0:
        return {
            **base,
            "docker_available": False,
            "error": result["stderr"] or result["stdout"] or "docker command failed",
        }
    line = result["stdout"].strip()
    if not line:
        return base
    name, _, status = line.partition("\t")
    return {
        **base,
        "exists": name == NEO4J_CONTAINER,
        "running": "Up " in status or status.startswith("Up"),
        "status": status,
    }


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}
    except subprocess.TimeoutExpired as error:
        return {
            "returncode": 124,
            "stdout": error.stdout or "",
            "stderr": error.stderr or f"Timed out after {timeout}s",
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
