from __future__ import annotations

import html
import os
from pathlib import Path


def create_app(*, root_dir: str | Path | None = None, static_dir: str | Path | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - exercised in CLI/runtime only
        raise RuntimeError(
            "Web UI support requires optional dependencies. Install with `uv pip install -e \".[webui]\"`."
        ) from exc

    from oneehr.web.service import WebUIService

    root_path = Path(root_dir or os.environ.get("ONEEHR_WEBUI_ROOT", "logs")).resolve()
    static_path = _resolve_static_dir(static_dir or os.environ.get("ONEEHR_WEBUI_STATIC"))
    service = WebUIService(root_dir=root_path)

    app = FastAPI(title="OneEHR Web UI API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "root_dir": str(root_path),
            "static_dir": None if static_path is None else str(static_path),
        }

    @app.get("/api/v1/runs")
    def list_runs_route() -> dict[str, object]:
        return service.list_runs_payload()

    @app.get("/api/v1/runs/{run_name}")
    def describe_run_route(run_name: str):
        try:
            return service.describe_run_payload(run_name=run_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/eval")
    def eval_route(run_name: str):
        try:
            return service.eval_payload(run_name=run_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/eval/tables/{table_name}")
    def eval_table_route(
        run_name: str,
        table_name: str,
        limit: int = Query(default=25, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        sort_by: str | None = None,
        sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
        filter_col: str | None = None,
        filter_value: str | None = None,
    ):
        try:
            return service.eval_table_payload(
                run_name=run_name,
                table_name=table_name,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_dir=sort_dir,
                filter_col=filter_col,
                filter_value=filter_value,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/eval/instances/{instance_id}")
    def eval_instance_route(run_name: str, instance_id: str):
        try:
            return service.eval_instance_payload(run_name=run_name, instance_id=instance_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/eval/traces/{system_name}")
    def eval_trace_route(
        run_name: str,
        system_name: str,
        limit: int = Query(default=25, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        stage: str | None = None,
        role: str | None = None,
        round_index: int | None = Query(default=None, ge=0),
    ):
        try:
            return service.eval_trace_payload(
                run_name=run_name,
                system_name=system_name,
                limit=limit,
                offset=offset,
                stage=stage,
                role=role,
                round_index=round_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis")
    def analysis_index_route(run_name: str):
        try:
            return service.analysis_index_payload(run_name=run_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis/{module_name}/dashboard")
    def analysis_dashboard_route(run_name: str, module_name: str):
        try:
            return service.analysis_dashboard_payload(run_name=run_name, module_name=module_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis/{module_name}/tables/{table_name}")
    def analysis_table_route(
        run_name: str,
        module_name: str,
        table_name: str,
        limit: int = Query(default=25, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        sort_by: str | None = None,
        sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
        filter_col: str | None = None,
        filter_value: str | None = None,
    ):
        try:
            return service.analysis_table_payload(
                run_name=run_name,
                module_name=module_name,
                table_name=table_name,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_dir=sort_dir,
                filter_col=filter_col,
                filter_value=filter_value,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis/{module_name}/cases")
    def analysis_cases_index_route(run_name: str, module_name: str):
        try:
            return service.analysis_cases_index_payload(run_name=run_name, module_name=module_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis/{module_name}/cases/{case_name}")
    def analysis_case_rows_route(
        run_name: str,
        module_name: str,
        case_name: str,
        limit: int = Query(default=25, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        sort_by: str | None = None,
        sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
        filter_col: str | None = None,
        filter_value: str | None = None,
    ):
        try:
            return service.analysis_case_rows_payload(
                run_name=run_name,
                module_name=module_name,
                case_name=case_name,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_dir=sort_dir,
                filter_col=filter_col,
                filter_value=filter_value,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/analysis/{module_name}/patient-case/{patient_id}")
    def analysis_patient_case_route(
        run_name: str,
        module_name: str,
        patient_id: str,
        limit: int = Query(default=25, ge=1, le=500),
    ):
        try:
            return service.analysis_patient_case_payload(
                run_name=run_name,
                module_name=module_name,
                patient_id=patient_id,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/comparison")
    def comparison_route(run_name: str):
        try:
            return service.comparison_payload(run_name=run_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/comparison/tables/{table_name}")
    def comparison_table_route(
        run_name: str,
        table_name: str,
        limit: int = Query(default=25, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        sort_by: str | None = None,
        sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
        filter_col: str | None = None,
        filter_value: str | None = None,
    ):
        try:
            return service.comparison_table_payload(
                run_name=run_name,
                table_name=table_name,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_dir=sort_dir,
                filter_col=filter_col,
                filter_value=filter_value,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_name}/cohorts/compare")
    def cohort_compare_route(
        run_name: str,
        split: str | None = None,
        left_role: str = Query(default="train", pattern="^(train|val|test)$"),
        right_role: str = Query(default="test", pattern="^(train|val|test)$"),
        top_k: int = Query(default=10, ge=1, le=100),
    ):
        try:
            return service.cohort_compare_payload(
                run_name=run_name,
                split=split,
                left_role=left_role,
                right_role=right_role,
                top_k=top_k,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if static_path is not None:
        assets_dir = static_path / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        def spa_index():
            return FileResponse(static_path / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_catch_all(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            candidate = static_path / full_path
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(static_path / "index.html")

    else:

        @app.get("/", include_in_schema=False)
        def instructions():
            return HTMLResponse(_missing_static_html(root_path=root_path))

    return app


def create_app_from_env():
    return create_app()


def _resolve_static_dir(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path).resolve()
        if (candidate / "index.html").exists():
            return candidate
        return None
    candidate = Path(__file__).resolve().parents[2] / "webui" / "dist"
    if (candidate / "index.html").exists():
        return candidate
    return None


def _missing_static_html(*, root_path: Path) -> str:
    root = html.escape(str(root_path))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>OneEHR Web UI</title>
    <style>
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: Georgia, "Times New Roman", serif;
        background: linear-gradient(180deg, #eef5f4 0%, #f7fbfa 55%, #ffffff 100%);
        color: #14312b;
        display: grid;
        place-items: center;
      }}
      main {{
        width: min(720px, calc(100vw - 48px));
        padding: 32px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: 0 20px 60px rgba(20, 49, 43, 0.12);
      }}
      code {{ background: #eff4f1; padding: 2px 6px; border-radius: 6px; }}
      p {{ line-height: 1.6; }}
    </style>
  </head>
  <body>
    <main>
      <h1>OneEHR Web UI</h1>
      <p>The API is live and reading runs from <code>{root}</code>.</p>
      <p>Build the frontend before serving it:</p>
      <p><code>cd webui && npm install && npm run build</code></p>
      <p>Then restart <code>oneehr webui serve</code>.</p>
    </main>
  </body>
</html>
"""
