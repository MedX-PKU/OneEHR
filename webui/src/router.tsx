/* eslint-disable react-refresh/only-export-components */

import { Suspense, lazy } from 'react'
import type { ComponentType } from 'react'
import {
  Link,
  Outlet,
  createRootRoute,
  createRoute,
  createRouter,
  useParams,
} from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchRunConsole } from './lib/api'
import { EmptyState } from './ui/EmptyState'
import { LoadingPanel } from './ui/LoadingPanel'
import { StatusBadge } from './ui/StatusBadge'
import { formatIdentifierDisplay, titleCase } from './lib/format'
import { sortModulesByPriority } from './lib/modules'

function createLazyPage(load: () => Promise<{ default: ComponentType }>, label: string) {
  const Page = lazy(load)
  return function LazyPage() {
    return (
      <Suspense fallback={<LoadingPanel label={label} />}>
        <Page />
      </Suspense>
    )
  }
}

const RunsPage = createLazyPage(
  () => import('./routes/RunsPage').then((module) => ({ default: module.RunsPage })),
  'Loading experiment runs',
)

const RunOverviewPage = createLazyPage(
  () => import('./routes/RunOverviewPage').then((module) => ({ default: module.RunOverviewPage })),
  'Loading run overview',
)

const ModuleDashboardPage = createLazyPage(
  () => import('./routes/ModuleDashboardPage').then((module) => ({ default: module.ModuleDashboardPage })),
  'Loading analysis dashboard',
)

const EvalPage = createLazyPage(
  () => import('./routes/EvalPage').then((module) => ({ default: module.EvalPage })),
  'Loading evaluation console',
)

const ComparisonPage = createLazyPage(
  () => import('./routes/ComparisonPage').then((module) => ({ default: module.ComparisonPage })),
  'Loading comparison',
)

function RootLayout() {
  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <header className="topbar">
        <div>
          <Link to="/" className="brand">
            OneEHR
          </Link>
          <p className="brand-tagline">
            Shared run contracts for EHR modeling, analysis, and evaluation.
          </p>
        </div>
        <nav className="topnav">
          <Link to="/" activeProps={{ className: 'topnav-link active' }} className="topnav-link">
            Runs
          </Link>
        </nav>
      </header>
      <main className="viewport">
        <Outlet />
      </main>
    </div>
  )
}

function RunLayout() {
  const { runName } = useParams({ from: '/runs/$runName' })
  const runConsoleQuery = useQuery({
    queryKey: ['run-console', runName],
    queryFn: () => fetchRunConsole(runName),
  })

  if (runConsoleQuery.isLoading) {
    return <LoadingPanel label="Loading run console" />
  }

  if (runConsoleQuery.isError || !runConsoleQuery.data) {
    return (
      <EmptyState
        title="Run console unavailable"
        description={runConsoleQuery.error instanceof Error ? runConsoleQuery.error.message : 'Unable to load run details.'}
      />
    )
  }

  const run = runConsoleQuery.data.run
  const orderedModules = sortModulesByPriority(run.analysis.modules)

  return (
    <div className="run-shell">
      <aside className="run-rail">
        <div className="rail-card">
          <p className="eyebrow">Navigation</p>
          <div className="stack-list">
            <Link
              to="/runs/$runName"
              params={{ runName }}
              activeProps={{ className: 'rail-link active' }}
              className="rail-link"
            >
              <span>Overview</span>
            </Link>
            <Link
              to="/runs/$runName/eval"
              params={{ runName }}
              activeProps={{ className: 'rail-link active' }}
              className="rail-link"
            >
              <span>Evaluation</span>
              <strong>{run.eval.system_count}</strong>
            </Link>
            <Link
              to="/runs/$runName/comparison"
              params={{ runName }}
              activeProps={{ className: 'rail-link active' }}
              className="rail-link secondary"
            >
              Comparison
            </Link>
          </div>
        </div>

        <div className="rail-card">
          <p className="eyebrow">Run</p>
          <h1 className="rail-title identifier-text">{formatIdentifierDisplay(run.run_name)}</h1>
          <div className="detail-grid compact">
            <div>
              <span>Task</span>
              <strong>{titleCase(String(run.manifest.task?.kind ?? 'unknown'))}</strong>
            </div>
            <div>
              <span>Mode</span>
              <strong>{titleCase(String(run.manifest.task?.prediction_mode ?? 'unknown'))}</strong>
            </div>
            <div>
              <span>Models</span>
              <strong>{run.training.models.length}</strong>
            </div>
            <div>
              <span>Splits</span>
              <strong>{run.training.splits.length}</strong>
            </div>
            <div>
              <span>Test rows</span>
              <strong>{run.testing.record_count}</strong>
            </div>
            <div>
              <span>Best test</span>
              <strong>{run.testing.best_model?.model ?? '—'}</strong>
            </div>
          </div>
        </div>

        <div className="rail-card">
          <p className="eyebrow">Modules</p>
          <div className="stack-list">
            {orderedModules.map((module) => (
              <Link
                key={module.name}
                to="/runs/$runName/analysis/$moduleName"
                params={{ runName, moduleName: module.name }}
                activeProps={{ className: 'rail-link active' }}
                className="rail-link"
              >
                <span>{module.title ?? titleCase(module.name)}</span>
                <StatusBadge status={module.status} />
              </Link>
            ))}
          </div>
        </div>
      </aside>

      <section className="run-content">
        <Outlet />
      </section>
    </div>
  )
}

const rootRoute = createRootRoute({
  component: RootLayout,
})

const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: RunsPage,
})

const runRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: 'runs/$runName',
  component: RunLayout,
})

const runOverviewRoute = createRoute({
  getParentRoute: () => runRoute,
  path: '/',
  component: RunOverviewPage,
})

const moduleRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'analysis/$moduleName',
  component: ModuleDashboardPage,
})

const evalRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'eval',
  component: EvalPage,
})

const comparisonRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'comparison',
  component: ComparisonPage,
})

const routeTree = rootRoute.addChildren([
  runsRoute,
  runRoute.addChildren([
    runOverviewRoute,
    moduleRoute,
    evalRoute,
    comparisonRoute,
  ]),
])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
