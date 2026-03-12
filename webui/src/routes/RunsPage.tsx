import { Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchRuns } from '../lib/api'
import { formatDate, titleCase } from '../lib/format'
import { EmptyState } from '../ui/EmptyState'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'

export function RunsPage() {
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
  })

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

  const runs = runsQuery.data ?? []

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
