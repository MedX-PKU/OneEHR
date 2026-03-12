import { useMemo, useState } from 'react'
import { Link, useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import {
  fetchCaseArtifacts,
  fetchFailureCases,
  fetchModuleDashboard,
  fetchPatientCase,
} from '../lib/api'
import { titleCase } from '../lib/format'
import { ChartPanel } from '../ui/ChartPanel'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'

export function ModuleDashboardPage() {
  const { runName, moduleName } = useParams({ from: '/runs/$runName/analysis/$moduleName' })
  const [selectedArtifact, setSelectedArtifact] = useState<string | null>(null)
  const [patientId, setPatientId] = useState('')

  const dashboardQuery = useQuery({
    queryKey: ['dashboard', runName, moduleName],
    queryFn: () => fetchModuleDashboard(runName, moduleName),
  })

  const caseArtifactsQuery = useQuery({
    queryKey: ['case-artifacts', runName, moduleName],
    queryFn: () => fetchCaseArtifacts(runName, moduleName),
    enabled: moduleName === 'prediction_audit' || moduleName === 'agent_audit',
  })

  const caseRowsQuery = useQuery({
    queryKey: ['case-rows', runName, moduleName, selectedArtifact],
    queryFn: () => fetchFailureCases(runName, moduleName, selectedArtifact ?? undefined),
    enabled: Boolean(selectedArtifact),
  })

  const patientQuery = useQuery({
    queryKey: ['patient-case', runName, moduleName, patientId],
    queryFn: () => fetchPatientCase(runName, moduleName, patientId),
    enabled: patientId.trim().length > 0,
  })

  const caseTable = useMemo(() => {
    if (!caseRowsQuery.data) {
      return null
    }
    return {
      key: `case-${caseRowsQuery.data.name ?? 'rows'}`,
      title: `${titleCase(moduleName)} case drill-down`,
      description: caseRowsQuery.data.name ?? undefined,
      row_count: caseRowsQuery.data.row_count,
      columns: caseRowsQuery.data.columns,
      records: caseRowsQuery.data.records,
    }
  }, [caseRowsQuery.data, moduleName])

  const patientTable = useMemo(() => {
    if (!patientQuery.data) {
      return null
    }
    const columns = patientQuery.data.matches[0]
      ? Object.keys(patientQuery.data.matches[0])
      : ['patient_id']
    return {
      key: `patient-${patientQuery.data.patient_id}`,
      title: `Patient ${patientQuery.data.patient_id}`,
      description: `${patientQuery.data.n_matches} matched rows`,
      row_count: patientQuery.data.n_matches,
      columns,
      records: patientQuery.data.matches,
    }
  }, [patientQuery.data])

  if (dashboardQuery.isLoading) {
    return <LoadingPanel label={`Loading ${moduleName} dashboard`} />
  }

  if (dashboardQuery.isError || !dashboardQuery.data) {
    return (
      <EmptyState
        title="Dashboard unavailable"
        description={dashboardQuery.error instanceof Error ? dashboardQuery.error.message : 'Unable to load module.'}
      />
    )
  }

  const dashboard = dashboardQuery.data

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Analysis Module</p>
          <h1>{dashboard.title}</h1>
          <p className="hero-copy">
            {dashboard.notes?.[0] ??
              'Structured artifacts translated into dashboard cards, rich tables, and visual drill-downs.'}
          </p>
        </div>
        <div className="header-actions">
          <StatusBadge status={dashboard.status} />
          {dashboard.comparison_available ? (
            <Link to="/runs/$runName/comparison" params={{ runName }} className="button-link">
              View comparison
            </Link>
          ) : null}
        </div>
      </section>

      <section className="card-grid kpi-grid">
        {dashboard.cards.map((card) => (
          <KpiCard
            key={card.key}
            title={card.title}
            value={card.value}
            subtitle={card.subtitle}
            tone={card.tone}
          />
        ))}
      </section>

      <section className="card-grid chart-grid">
        {dashboard.charts.map((chart) => (
          <ChartPanel key={chart.key} chart={chart} />
        ))}
      </section>

      <section className="page-stack">
        {dashboard.tables.map((table) => (
          <DataTable key={table.key} table={table} />
        ))}
      </section>

      {caseArtifactsQuery.data && caseArtifactsQuery.data.length > 0 ? (
        <section className="two-column-grid">
          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Drill-down</p>
                <h2>Failure artifacts</h2>
              </div>
            </div>
            <div className="artifact-list">
              {caseArtifactsQuery.data.map((artifact) => (
                <button
                  key={artifact.name}
                  type="button"
                  className={`artifact-button ${selectedArtifact === artifact.name ? 'active' : ''}`}
                  onClick={() => setSelectedArtifact(artifact.name)}
                >
                  <strong>{artifact.name}</strong>
                  <span>{artifact.row_count} rows</span>
                </button>
              ))}
            </div>
          </article>

          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Patient lookup</p>
                <h2>Audit matches</h2>
              </div>
            </div>
            <label className="table-filter wide">
              <span>Patient ID</span>
              <input
                value={patientId}
                onChange={(event) => setPatientId(event.target.value)}
                placeholder="e.g. p0001"
              />
            </label>
            <p className="panel-copy">
              Search patient-level matches inside analysis case artifacts without leaving the current module view.
            </p>
          </article>
        </section>
      ) : null}

      {caseRowsQuery.isLoading ? <LoadingPanel label="Loading case rows" /> : null}
      {caseTable ? <DataTable table={caseTable} /> : null}
      {patientQuery.isLoading ? <LoadingPanel label="Searching patient matches" /> : null}
      {patientTable ? <DataTable table={patientTable} /> : null}
    </div>
  )
}
