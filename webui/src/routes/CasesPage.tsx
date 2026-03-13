import { Link, useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchCasesIndex } from '../lib/api'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { DataTable } from '../ui/DataTable'

export function CasesPage() {
  const { runName } = useParams({ from: '/runs/$runName/cases' })
  const casesQuery = useQuery({
    queryKey: ['cases', runName],
    queryFn: () => fetchCasesIndex(runName),
  })

  if (casesQuery.isLoading) {
    return <LoadingPanel label="Loading cases" />
  }

  if (casesQuery.isError || !casesQuery.data) {
    return (
      <EmptyState
        title="Cases unavailable"
        description={casesQuery.error instanceof Error ? casesQuery.error.message : 'Unable to load cases.'}
      />
    )
  }

  const payload = casesQuery.data
  const table = {
    key: 'cases',
    title: 'Case Inventory',
    description: `${payload.total_rows} cases in this run`,
    row_count: payload.total_rows,
    columns: payload.columns,
    records: payload.records,
  }

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Cases</p>
          <h1>Durable case bundles</h1>
          <p className="hero-copy">
            Patient-level evidence packages for review, audit, and agent-grounded workflows.
          </p>
        </div>
        <div className="header-actions">
          <Link to="/runs/$runName/agents" params={{ runName }} className="button-link">
            Open agents
          </Link>
        </div>
      </section>

      <section className="card-grid kpi-grid">
        <KpiCard key="cases" title="Cases" value={payload.case_count} subtitle="Indexed bundles" />
        <KpiCard key="visible-rows" title="Visible rows" value={payload.row_count} subtitle={`of ${payload.total_rows}`} />
        <KpiCard key="splits" title="Splits" value={payload.splits.length} subtitle={payload.splits.join(', ') || '—'} />
        <KpiCard key="status" title="Status" value={payload.status} subtitle="Run coverage" />
      </section>

      {payload.status !== 'ok' ? (
        <EmptyState
          title="No case bundles yet"
          description="Run `oneehr cases build` for this experiment to materialize durable case evidence."
        />
      ) : (
        <>
          <section className="two-column-grid">
            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Navigator</p>
                  <h2>Open a case</h2>
                </div>
              </div>
              <div className="artifact-list">
                {payload.records.slice(0, 12).map((record) => {
                  const caseId = String(record.case_id ?? '')
                  return (
                    <Link
                      key={caseId}
                      to="/runs/$runName/cases/$caseId"
                      params={{ runName, caseId }}
                      className="artifact-button"
                    >
                      <div>
                        <strong>{caseId}</strong>
                        <div className="artifact-meta">
                          Patient {String(record.patient_id ?? '—')} | Split {String(record.split ?? '—')}
                        </div>
                      </div>
                      <span>{String(record.prediction_mode ?? '—')}</span>
                    </Link>
                  )
                })}
              </div>
            </article>

            <article className="panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">Coverage</p>
                  <h2>Case coverage</h2>
                </div>
              </div>
              <div className="detail-grid">
                <div>
                  <span>Indexed cases</span>
                  <strong>{payload.case_count}</strong>
                </div>
                <div>
                  <span>Active splits</span>
                  <strong>{payload.splits.length}</strong>
                </div>
                <div>
                  <span>Prediction modes</span>
                  <strong>
                    {Array.from(new Set(payload.records.map((record) => String(record.prediction_mode ?? '—')))).join(', ')}
                  </strong>
                </div>
                <div>
                  <span>Selection</span>
                  <strong>{payload.row_count}</strong>
                </div>
              </div>
            </article>
          </section>

          <DataTable table={table} />
        </>
      )}
    </div>
  )
}
