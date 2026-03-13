import { Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchModuleDashboard, fetchRuns, fetchRunWorkspace } from '../lib/api'
import { formatDate, titleCase } from '../lib/format'
import { ChartPanel } from '../ui/ChartPanel'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'

export function RunsPage() {
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
  })
  const runs = runsQuery.data ?? []
  const spotlightRun = runs.find((run) => run.has_analysis_index)
  const spotlightWorkspaceQuery = useQuery({
    queryKey: ['run-workspace', spotlightRun?.run_name],
    queryFn: () => fetchRunWorkspace(String(spotlightRun?.run_name)),
    enabled: spotlightRun != null,
  })
  const spotlightModule = spotlightWorkspaceQuery.data?.analysis.modules.find((module) => module.status === 'ok') ?? null
  const spotlightDashboardQuery = useQuery({
    queryKey: ['module-dashboard-spotlight', spotlightRun?.run_name, spotlightModule?.name],
    queryFn: () => fetchModuleDashboard(String(spotlightRun?.run_name), String(spotlightModule?.name)),
    enabled: spotlightRun != null && spotlightModule != null,
  })
  const spotlightCharts = (spotlightDashboardQuery.data?.charts ?? []).slice(0, 2)
  const spotlightTables = (spotlightDashboardQuery.data?.tables ?? []).slice(0, 1)

  if (runsQuery.isLoading) {
    return <LoadingPanel label="Loading experiment runs" />
  }

  if (runsQuery.isError) {
    return (
      <EmptyState
        title="Run catalog unavailable"
        description={runsQuery.error instanceof Error ? runsQuery.error.message : 'Unable to load runs.'}
      />
    )
  }

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Operations Console</p>
          <h1>Longitudinal EHR run explorer</h1>
          <p className="hero-copy">
            Review modeling outputs, structured analysis, and agent-era audit signals from one workspace.
          </p>
        </div>
        <div className="hero-stats">
          <div>
            <span>Total runs</span>
            <strong>{runs.length}</strong>
          </div>
          <div>
            <span>Analysis-ready</span>
            <strong>{runs.filter((run) => run.has_analysis_index).length}</strong>
          </div>
          <div>
            <span>Agent-enabled</span>
            <strong>{runs.filter((run) => run.has_agent_predict_summary).length}</strong>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Visual Spotlight</p>
            <h2>First look at the latest analysis-ready run</h2>
            <p className="panel-copy">
              The console now surfaces live charts immediately instead of making you hunt through module pages first.
            </p>
          </div>
          {spotlightRun ? (
            <Link to="/runs/$runName" params={{ runName: spotlightRun.run_name }} className="button-link">
              Open {spotlightRun.run_name}
            </Link>
          ) : null}
        </div>

        {!spotlightRun ? (
          <EmptyState
            title="No analysis-ready runs yet"
            description="Run `oneehr analyze` after training to populate dashboard charts on the landing page."
          />
        ) : spotlightWorkspaceQuery.isLoading || spotlightDashboardQuery.isLoading ? (
          <LoadingPanel label="Loading visual spotlight" />
        ) : spotlightDashboardQuery.isError ? (
          <EmptyState
            title="Visual spotlight unavailable"
            description={
              spotlightDashboardQuery.error instanceof Error
                ? spotlightDashboardQuery.error.message
                : 'Unable to load spotlight charts.'
            }
          />
        ) : spotlightDashboardQuery.data == null ? (
          <EmptyState
            title="No spotlight dashboard available"
            description="This run has an analysis index, but no ready dashboard module produced a preview yet."
          />
        ) : spotlightCharts.length > 0 ? (
          <div className="chart-grid">
            {spotlightCharts.map((chart) => (
              <ChartPanel
                key={`spotlight-${chart.key}`}
                chart={{
                  ...chart,
                  key: `spotlight-${chart.key}`,
                  title: `${spotlightDashboardQuery.data?.title}: ${chart.title}`,
                }}
              />
            ))}
          </div>
        ) : spotlightTables.length > 0 ? (
          <DataTable table={spotlightTables[0]} />
        ) : (
          <EmptyState
            title="No visual preview available"
            description="This run has analysis metadata, but no charts or preview tables were emitted for the first ready module."
          />
        )}
      </section>

      {runs.length === 0 ? (
        <EmptyState
          title="No runs found"
          description="Point the backend at a logs root with OneEHR run manifests to populate the dashboard."
        />
      ) : (
        <section className="card-grid runs-grid">
          {runs.map((run) => (
            <Link
              key={run.run_name}
              to="/runs/$runName"
              params={{ runName: run.run_name }}
              className="run-card"
            >
              <div className="run-card-header">
                <div>
                  <p className="eyebrow">Run</p>
                  <h2>{run.run_name}</h2>
                </div>
                <StatusBadge status={run.has_analysis_index ? 'ok' : 'warning'} />
              </div>
              <div className="detail-grid">
                <div>
                  <span>Task</span>
                  <strong>{titleCase(String(run.task?.kind ?? 'unknown'))}</strong>
                </div>
                <div>
                  <span>Mode</span>
                  <strong>{titleCase(String(run.task?.prediction_mode ?? 'unknown'))}</strong>
                </div>
                <div>
                  <span>Split</span>
                  <strong>{titleCase(String(run.split?.kind ?? 'unknown'))}</strong>
                </div>
                <div>
                  <span>Updated</span>
                  <strong>{formatDate(run.mtime_unix)}</strong>
                </div>
              </div>
              <div className="capability-row">
                <span>{run.has_cases_index ? 'Cases ready' : 'No cases'}</span>
                <span>{run.has_agent_predict_summary ? 'Agent predict' : 'Model only'}</span>
              </div>
            </Link>
          ))}
        </section>
      )}
    </div>
  )
}
