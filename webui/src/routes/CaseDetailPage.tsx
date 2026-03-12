import { Link, useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchCaseDetail } from '../lib/api'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { DataTable } from '../ui/DataTable'

export function CaseDetailPage() {
  const { runName, caseId } = useParams({ from: '/runs/$runName/cases/$caseId' })
  const caseQuery = useQuery({
    queryKey: ['case-detail', runName, caseId],
    queryFn: () => fetchCaseDetail(runName, caseId),
  })

  if (caseQuery.isLoading) {
    return <LoadingPanel label="Loading case detail" />
  }

  if (caseQuery.isError || !caseQuery.data) {
    return (
      <EmptyState
        title="Case unavailable"
        description={caseQuery.error instanceof Error ? caseQuery.error.message : 'Unable to load case detail.'}
      />
    )
  }

  const payload = caseQuery.data
  const patientId = String(payload.case.patient_id ?? '—')
  const split = String(payload.case.split ?? '—')
  const groundTruth = payload.case.ground_truth == null ? '—' : String(payload.case.ground_truth)

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Case Detail</p>
          <h1>{String(payload.case.case_id ?? caseId)}</h1>
          <p className="hero-copy">
            Patient {patientId} | Split {split} | Ground truth {groundTruth}
          </p>
        </div>
        <div className="header-actions">
          <Link to="/runs/$runName/cases" params={{ runName }} className="button-link">
            Back to cases
          </Link>
          <Link to="/runs/$runName/agents" params={{ runName }} className="button-link">
            Open agents
          </Link>
        </div>
      </section>

      <section className="card-grid kpi-grid">
        <KpiCard key="timeline" title="Timeline rows" value={payload.timeline.row_count} subtitle="Observed events" />
        <KpiCard key="predictions" title="Predictions" value={payload.predictions.row_count} subtitle="Model and agent outputs" />
        <KpiCard key="static" title="Static features" value={payload.static.feature_count} subtitle="Patient covariates" />
        <KpiCard
          key="analysis-refs"
          title="Analysis refs"
          value={payload.analysis_refs.module_count}
          subtitle={`${payload.analysis_refs.patient_case_match_count} patient matches`}
        />
      </section>

      <section className="two-column-grid">
        <DataTable table={payload.static.table} />
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Case contract</p>
              <h2>Metadata</h2>
            </div>
          </div>
          <div className="detail-grid">
            {Object.entries(payload.case).map(([key, value]) => (
              <div key={key}>
                <span>{key.replace(/_/g, ' ')}</span>
                <strong>{value == null || value === '' ? '—' : String(value)}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      <DataTable table={payload.timeline} />
      <DataTable table={payload.predictions} />

      {payload.analysis_refs.modules.row_count > 0 ? <DataTable table={payload.analysis_refs.modules} /> : null}
      {payload.analysis_refs.patient_case_matches.row_count > 0 ? (
        <DataTable table={payload.analysis_refs.patient_case_matches} />
      ) : null}
    </div>
  )
}
