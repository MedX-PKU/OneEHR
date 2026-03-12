# Web UI

OneEHR ships a first-party Web UI for run discovery, analysis dashboards, comparison views, and audit drill-downs.

The architecture is intentionally artifact-first:

- the browser reads only `/api/v1`
- the FastAPI backend reads the existing run artifact contract under `logs/<run_name>/`
- no database or alternate report format is introduced in v1

## Install

Python backend dependencies:

```bash
uv pip install -e ".[webui]"
```

Frontend dependencies:

```bash
cd webui
npm install
```

## Build And Serve

Build the frontend once:

```bash
cd webui
npm run build
```

Serve the API plus built dashboard:

```bash
uv run oneehr webui serve --root logs
```

Optional flags:

```bash
uv run oneehr webui serve --root logs --host 0.0.0.0 --port 8000
uv run oneehr webui serve --root logs --frontend-dist /path/to/webui/dist
uv run oneehr webui serve --root logs --reload
```

If the frontend has not been built yet, the root page shows a small HTML instruction page while `/api/v1/*` remains available.

## Frontend Dev

Run the FastAPI backend:

```bash
uv run oneehr webui serve --root logs
```

In another shell, start Vite:

```bash
cd webui
npm run dev
```

By default the Vite dev server proxies `/api/*` to `http://127.0.0.1:8000`.

## What The UI Covers

Current pages:

- run explorer
- run overview
- module dashboards for `dataset_profile`, `cohort_analysis`, `prediction_audit`, `temporal_analysis`, `interpretability`, and `agent_audit`
- compare-run view when `analysis/comparison/*` exists

Current drill-downs:

- failure-case artifact browsing for `prediction_audit` and `agent_audit`
- patient-level match lookup within saved case slices

## API Surface

The Web UI uses these endpoints:

```text
GET /api/v1/health
GET /api/v1/runs
GET /api/v1/runs/{run_name}
GET /api/v1/runs/{run_name}/analysis
GET /api/v1/runs/{run_name}/analysis/{module}/dashboard
GET /api/v1/runs/{run_name}/analysis/{module}/tables/{table}
GET /api/v1/runs/{run_name}/analysis/{module}/cases
GET /api/v1/runs/{run_name}/analysis/{module}/cases/{name}
GET /api/v1/runs/{run_name}/analysis/{module}/patient-case/{patient_id}
GET /api/v1/runs/{run_name}/comparison
```

These routes are read-only. They normalize existing JSON, CSV, and parquet artifacts into frontend-friendly view models without changing the on-disk contract.
