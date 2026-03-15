# Web UI

OneEHR ships a first-party web/API layer for browsing runs, structured analysis outputs, unified evaluation artifacts, and compare-run summaries. This is the artifact-first read layer over the task-oriented CLI workflow: it reads only from the existing run contract through `/api/v1`, and the browser never touches run files directly.

## What It Serves

The current UI covers:

- run discovery and run overview pages
- module dashboards for structured analysis outputs
- unified evaluation payloads backed by `eval/index.json`, `eval/summary.json`, and `eval/reports/*`
- compare-run views backed by saved comparison artifacts

The Web UI is read-only. It does not introduce a database or a parallel report format.

## Install And Build

Install backend dependencies:

```bash
uv pip install -e ".[webui]"
```

Install frontend dependencies:

```bash
cd webui
npm install
```

Build the frontend bundle once:

```bash
npm run build
```

## Serve The Dashboard

Serve the API plus built frontend:

```bash
cd ..
uv run oneehr webui serve --root logs
```

Common flags:

```bash
uv run oneehr webui serve --root logs --host 0.0.0.0 --port 8000
uv run oneehr webui serve --root logs --frontend-dist /path/to/webui/dist
uv run oneehr webui serve --root logs --reload
```

If `webui/dist` is missing, the API still serves and the root page shows a small build-instructions page.

## Frontend Development Loop

Run the FastAPI backend:

```bash
uv run oneehr webui serve --root logs
```

In another shell, run the Vite dev server:

```bash
cd webui
npm run dev
```

By default, the Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

## API Surface

The Web UI is backed by these read-only endpoints:

```text
GET /api/v1/health
GET /api/v1/runs
GET /api/v1/runs/{run_name}
GET /api/v1/runs/{run_name}/eval
GET /api/v1/runs/{run_name}/eval/tables/{table_name}
GET /api/v1/runs/{run_name}/eval/instances/{instance_id}
GET /api/v1/runs/{run_name}/eval/traces/{system_name}
GET /api/v1/runs/{run_name}/analysis
GET /api/v1/runs/{run_name}/analysis/{module}/dashboard
GET /api/v1/runs/{run_name}/analysis/{module}/tables/{table}
GET /api/v1/runs/{run_name}/analysis/{module}/cases
GET /api/v1/runs/{run_name}/analysis/{module}/cases/{name}
GET /api/v1/runs/{run_name}/analysis/{module}/patient-case/{patient_id}
GET /api/v1/runs/{run_name}/comparison
GET /api/v1/runs/{run_name}/comparison/tables/{table}
GET /api/v1/runs/{run_name}/cohorts/compare
```

These routes normalize existing JSON, CSV, and parquet artifacts into frontend-friendly view models without changing the on-disk contract.

Legacy `/cases` and `/agents` API routes are intentionally not part of the current public surface.
