from __future__ import annotations

import argparse
import os
from pathlib import Path


def register_webui_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("webui", help="Serve the OneEHR Web UI and API")
    sub = parser.add_subparsers(dest="webui_action")

    serve_cmd = sub.add_parser("serve", help="Serve the Web UI dashboard")
    serve_cmd.add_argument("--root", default="logs", help="Root directory containing run folders")
    serve_cmd.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_cmd.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_cmd.add_argument(
        "--frontend-dist",
        default=None,
        help="Optional path to built frontend assets. Defaults to webui/dist when present.",
    )
    serve_cmd.add_argument("--reload", action="store_true", help="Enable FastAPI reload mode")
    serve_cmd.set_defaults(handler=_run_webui_serve)


def _run_webui_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit(
            "Web UI support requires optional dependencies. Install with `uv pip install -e \".[webui]\"`."
        ) from exc

    root = str(Path(args.root).resolve())
    os.environ["ONEEHR_WEBUI_ROOT"] = root
    if args.frontend_dist is not None:
        os.environ["ONEEHR_WEBUI_STATIC"] = str(Path(args.frontend_dist).resolve())
    elif "ONEEHR_WEBUI_STATIC" in os.environ:
        del os.environ["ONEEHR_WEBUI_STATIC"]

    if args.reload:
        uvicorn.run(
            "oneehr.web:create_app_from_env",
            host=args.host,
            port=int(args.port),
            reload=True,
            factory=True,
        )
        return

    from oneehr.web import create_app

    app = create_app(root_dir=root, static_dir=args.frontend_dist)
    uvicorn.run(app, host=args.host, port=int(args.port))
