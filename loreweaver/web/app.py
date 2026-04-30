"""FastAPI application factory for the LoreWeaver debugging UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from loreweaver.web.api import build_api_router
from loreweaver.web.inspectors import DebugInspector
from loreweaver.web.jobs import JobManager


def create_app(
    *,
    config_path: str = "configs/default.yaml",
    storage_config_path: str = "configs/storage.yaml",
    models_config_path: str = "configs/models.yaml",
) -> FastAPI:
    base_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    inspector = DebugInspector(
        config_path=config_path,
        storage_config_path=storage_config_path,
        models_config_path=models_config_path,
    )
    jobs = JobManager(
        config_path=config_path,
        storage_config_path=storage_config_path,
        models_config_path=models_config_path,
    )

    app = FastAPI(title="LoreWeaver Debug UI")
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")
    app.include_router(build_api_router(inspector, jobs))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html")

    return app
