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
import { fetchRunDetail } from './lib/api'
import { EmptyState } from './ui/EmptyState'
import { LoadingPanel } from './ui/LoadingPanel'
import { StatusBadge } from './ui/StatusBadge'
import { titleCase } from './lib/format'

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

const CasesPage = createLazyPage(
  () => import('./routes/CasesPage').then((module) => ({ default: module.CasesPage })),
  'Loading case workspace',
)

const CaseDetailPage = createLazyPage(
  () => import('./routes/CaseDetailPage').then((module) => ({ default: module.CaseDetailPage })),
  'Loading case detail',
)

const AgentsPage = createLazyPage(
  () => import('./routes/AgentsPage').then((module) => ({ default: module.AgentsPage })),
  'Loading agent workspace',
)

const ComparisonPage = createLazyPage(
  () => import('./routes/ComparisonPage').then((module) => ({ default: module.ComparisonPage })),
  'Loading comparison workspace',
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
            Longitudinal EHR observability, analysis, and agent-era model operations.
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
  const runQuery = useQuery({
    queryKey: ['run', runName],
    queryFn: () => fetchRunDetail(runName),
  })

  if (runQuery.isLoading) {
    return <LoadingPanel label="Loading run workspace" />
  }

  if (runQuery.isError || !runQuery.data) {
    return (
      <EmptyState
        title="Run workspace unavailable"
        description={runQuery.error instanceof Error ? runQuery.error.message : 'Unable to load run details.'}
      />
    )
  }

  const run = runQuery.data

  return (
    <div className="workspace-shell">
      <aside className="workspace-rail">
        <div className="workspace-card">
          <p className="eyebrow">Workspace</p>
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
              to="/runs/$runName/cases"
              params={{ runName }}
              activeProps={{ className: 'rail-link active' }}
              className="rail-link"
            >
              <span>Cases</span>
              <strong>{run.cases.case_count}</strong>
            </Link>
            <Link
              to="/runs/$runName/agents"
              params={{ runName }}
              activeProps={{ className: 'rail-link active' }}
              className="rail-link"
            >
              <span>Agents</span>
              <strong>{run.agent_predict.record_count + run.agent_review.record_count}</strong>
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

        <div className="workspace-card">
          <p className="eyebrow">Run</p>
          <h1>{run.run_name}</h1>
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
          </div>
        </div>

        <div className="workspace-card">
          <p className="eyebrow">Modules</p>
          <div className="stack-list">
            {run.analysis.modules.map((module) => (
              <Link
                key={module.name}
                to="/runs/$runName/analysis/$moduleName"
                params={{ runName, moduleName: module.name }}
                activeProps={{ className: 'rail-link active' }}
                className="rail-link"
              >
                <span>{titleCase(module.name)}</span>
                <StatusBadge status={module.status} />
              </Link>
            ))}
          </div>
        </div>
      </aside>

      <section className="workspace-content">
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

const casesRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'cases',
  component: CasesPage,
})

const caseDetailRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'cases/$caseId',
  component: CaseDetailPage,
})

const agentsRoute = createRoute({
  getParentRoute: () => runRoute,
  path: 'agents',
  component: AgentsPage,
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
    casesRoute,
    caseDetailRoute,
    agentsRoute,
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
