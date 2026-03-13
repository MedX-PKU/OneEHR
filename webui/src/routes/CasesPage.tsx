import { useDeferredValue, useEffect, useState } from 'react'
import { Link, useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { formatNumber } from '../lib/format'
import { fetchCasesIndex } from '../lib/api'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { DataTable } from '../ui/DataTable'

const DEFAULT_CASE_PAGE_SIZE = 25

export function CasesPage() {
  const { runName } = useParams({ from: '/runs/$runName/cases' })
  const [split, setSplit] = useState('')
  const [search, setSearch] = useState('')
  const [limit, setLimit] = useState(DEFAULT_CASE_PAGE_SIZE)
  const [offset, setOffset] = useState(0)
  const deferredSearch = useDeferredValue(search)
  const casesQuery = useQuery({
    queryKey: ['cases', runName, split, deferredSearch, limit, offset],
    queryFn: () =>
      fetchCasesIndex(runName, {
        split: split || null,
        search: deferredSearch || null,
        limit,
        offset,
      }),
  })

  useEffect(() => {
    setOffset(0)
  }, [split, search, limit])

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
  const pageStart = payload.total_rows === 0 ? 0 : payload.offset + 1
  const pageEnd = payload.total_rows === 0 ? 0 : payload.offset + payload.row_count
  const canGoBack = payload.offset > 0
  const canGoForward = payload.offset + payload.limit < payload.total_rows
  const table = {
    key: 'cases',
    title: 'Case Inventory',
    description: `Showing ${pageStart}-${pageEnd} of ${formatNumber(payload.total_rows)} indexed cases`,
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
          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Inventory Controls</p>
                <h2>Filter the indexed cases</h2>
                <p className="panel-copy">
                  Query saved case bundles by split or patient/case search without loading the full inventory into the browser.
                </p>
              </div>
            </div>
            <div className="filter-grid">
              <label className="table-filter">
                <span>Split</span>
                <select value={split} onChange={(event) => setSplit(event.target.value)}>
                  <option value="">All splits</option>
                  {payload.splits.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>

              <label className="table-filter wide">
                <span>Search</span>
                <input
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search case ID or patient ID"
                />
              </label>

              <label className="table-filter">
                <span>Page size</span>
                <select value={String(limit)} onChange={(event) => setLimit(Number(event.target.value))}>
                  <option value="25">25 rows</option>
                  <option value="50">50 rows</option>
                  <option value="100">100 rows</option>
                </select>
              </label>
            </div>
          </article>

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

          <DataTable
            table={table}
            searchable={false}
            emptyMessage="No indexed cases matched the current filters."
            toolbarActions={
              <div className="table-toolbar-actions">
                <span className="table-page-summary">
                  Page {Math.floor(payload.offset / payload.limit) + 1}
                </span>
                <button
                  type="button"
                  className="button-link"
                  onClick={() => setOffset(Math.max(0, payload.offset - payload.limit))}
                  disabled={!canGoBack}
                >
                  Previous
                </button>
                <button
                  type="button"
                  className="button-link"
                  onClick={() => setOffset(payload.offset + payload.limit)}
                  disabled={!canGoForward}
                >
                  Next
                </button>
              </div>
            }
          />
        </>
      )}
    </div>
  )
}
