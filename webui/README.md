# OneEHR Web UI

React + TypeScript + Vite frontend for the OneEHR dashboard.

## Dev

```bash
cd webui
npm install
npm run dev
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`. Override the backend origin with:

```bash
VITE_ONEEHR_API_BASE_URL=http://127.0.0.1:9000 npm run dev
```

## Build

```bash
cd webui
npm run build
```

## Expected API

The app expects a FastAPI backend serving JSON under `/api/v1` with run discovery, run detail, case inventory/detail, agent summaries, analysis dashboards, case drill-down, and comparison endpoints.
