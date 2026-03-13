# Web UI

OneEHR ships a first-party Web UI for browsing runs, structured analysis outputs, durable case bundles, and agent artifacts. The UI is intentionally artifact-first: it reads only from the existing run contract through `/api/v1`, and the browser never touches run files directly.

## What It Serves

The current UI covers:

- run discovery and run overview pages
- module dashboards for `dataset_profile`, `cohort_analysis`, `prediction_audit`, `temporal_analysis`, `interpretability`, and `agent_audit`
- case inventory filters and case detail pages
- agent prediction and review summaries plus detailed record and failure browsing
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
GET /api/v1/runs/{run_name}/cases
GET /api/v1/runs/{run_name}/cases/{case_id}
GET /api/v1/runs/{run_name}/agents
GET /api/v1/runs/{run_name}/agents/{task_name}/records
GET /api/v1/runs/{run_name}/agents/{task_name}/failures
GET /api/v1/runs/{run_name}/analysis
GET /api/v1/runs/{run_name}/analysis/{module}/dashboard
GET /api/v1/runs/{run_name}/analysis/{module}/tables/{table}
GET /api/v1/runs/{run_name}/analysis/{module}/cases
GET /api/v1/runs/{run_name}/analysis/{module}/cases/{name}
GET /api/v1/runs/{run_name}/analysis/{module}/patient-case/{patient_id}
GET /api/v1/runs/{run_name}/comparison
GET /api/v1/runs/{run_name}/comparison/tables/{table}
```

These routes normalize existing JSON, CSV, and parquet artifacts into frontend-friendly view models without changing the on-disk contract.
