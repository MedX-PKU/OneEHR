import { useDeferredValue, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { fetchAnalysisTable } from '../lib/api'
import type { DashboardTable } from '../lib/types'
import { formatNumber } from '../lib/format'
import { DataTable } from './DataTable'
import { EmptyState } from './EmptyState'
import { LoadingPanel } from './LoadingPanel'

interface AnalysisTableExplorerProps {
  runName: string
  moduleName: string
  tables: DashboardTable[]
}

const DEFAULT_PAGE_SIZE = 25

export function AnalysisTableExplorer({
  runName,
  moduleName,
  tables,
}: AnalysisTableExplorerProps) {
  const [selectedTable, setSelectedTable] = useState<string>(tables[0]?.key ?? '')
  const [sortBy, setSortBy] = useState('')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filterCol, setFilterCol] = useState('')
  const [filterValue, setFilterValue] = useState('')
  const [limit, setLimit] = useState(DEFAULT_PAGE_SIZE)
  const [offset, setOffset] = useState(0)
  const deferredFilterValue = useDeferredValue(filterValue)

  useEffect(() => {
    if (tables.length === 0) {
      if (selectedTable !== '') {
        setSelectedTable('')
      }
      return
    }
    if (!tables.some((table) => table.key === selectedTable)) {
      setSelectedTable(tables[0]?.key ?? '')
    }
  }, [selectedTable, tables])

  useEffect(() => {
    setSortBy('')
    setSortDir('desc')
    setFilterCol('')
    setFilterValue('')
    setLimit(DEFAULT_PAGE_SIZE)
    setOffset(0)
  }, [selectedTable])

  useEffect(() => {
    setOffset(0)
  }, [sortBy, sortDir, filterCol, filterValue, limit])

  const selectedPreview = useMemo(
    () => tables.find((table) => table.key === selectedTable) ?? null,
    [selectedTable, tables],
  )
  const availableColumns = selectedPreview?.columns ?? []

  const tableQuery = useQuery({
    queryKey: [
      'analysis-table',
      runName,
      moduleName,
      selectedTable,
      limit,
      offset,
      sortBy,
      sortDir,
      filterCol,
      deferredFilterValue,
    ],
    queryFn: () =>
      fetchAnalysisTable(runName, moduleName, selectedTable, {
        limit,
        offset,
        sortBy: sortBy || null,
        sortDir,
        filterCol: filterCol || null,
        filterValue: deferredFilterValue || null,
      }),
    enabled: selectedTable.length > 0,
  })

  if (tables.length === 0) {
    return (
      <EmptyState
        title="No structured tables"
        description="This module produced charts and highlights, but no saved tabular artifacts were available to browse."
      />
    )
  }

  const totalRows = tableQuery.data?.total_rows ?? selectedPreview?.row_count ?? 0
  const pageRowCount = tableQuery.data?.row_count ?? 0
  const pageStart = totalRows === 0 ? 0 : offset + 1
  const pageEnd = totalRows === 0 ? 0 : offset + pageRowCount
  const canGoBack = offset > 0
  const canGoForward = offset + limit < totalRows

  return (
    <section className="page-stack">
      <section className="two-column-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Table Explorer</p>
              <h2>Choose a saved analysis table</h2>
              <p className="panel-copy">
                Switch across saved artifacts without leaving the current analysis module.
              </p>
            </div>
          </div>
          <div className="artifact-list">
            {tables.map((table) => (
              <button
                key={table.key}
                type="button"
                className={`artifact-button ${selectedTable === table.key ? 'active' : ''}`}
                onClick={() => setSelectedTable(table.key)}
              >
                <div>
                  <strong>{table.title}</strong>
                  <div className="artifact-meta">{table.description ?? `${formatNumber(table.row_count)} rows`}</div>
                </div>
                <span>{formatNumber(table.row_count)}</span>
              </button>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Server Controls</p>
              <h2>Sort, filter, and paginate the full table</h2>
              <p className="panel-copy">
                These controls query the backend instead of only searching the visible preview rows.
              </p>
            </div>
          </div>
          <div className="filter-grid">
            <label className="table-filter">
              <span>Sort by</span>
              <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                <option value="">Default order</option>
                {availableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>

            <label className="table-filter">
              <span>Direction</span>
              <select value={sortDir} onChange={(event) => setSortDir(event.target.value as 'asc' | 'desc')}>
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </label>

            <label className="table-filter">
              <span>Filter column</span>
              <select value={filterCol} onChange={(event) => setFilterCol(event.target.value)}>
                <option value="">No column filter</option>
                {availableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>

            <label className="table-filter wide">
              <span>Filter value</span>
              <input
                type="search"
                value={filterValue}
                onChange={(event) => setFilterValue(event.target.value)}
                disabled={!filterCol}
                placeholder={filterCol ? `Contains value in ${filterCol}` : 'Choose a filter column first'}
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
          <div className="detail-grid compact">
            <div>
              <span>Selected table</span>
              <strong>{selectedPreview?.title ?? '—'}</strong>
            </div>
            <div>
              <span>Total rows</span>
              <strong>{formatNumber(totalRows)}</strong>
            </div>
            <div>
              <span>Current window</span>
              <strong>
                {pageStart}-{pageEnd}
              </strong>
            </div>
            <div>
              <span>Column filter</span>
              <strong>{filterCol || 'None'}</strong>
            </div>
          </div>
        </article>
      </section>

      {tableQuery.isLoading ? (
        <LoadingPanel label="Loading analysis table" />
      ) : tableQuery.isError || !tableQuery.data ? (
        <EmptyState
          title="Table browser unavailable"
          description={tableQuery.error instanceof Error ? tableQuery.error.message : 'Unable to load the table.'}
        />
      ) : (
        <DataTable
          table={{
            key: tableQuery.data.key,
            title: tableQuery.data.title,
            description: `Showing ${pageStart}-${pageEnd} of ${formatNumber(tableQuery.data.total_rows)} rows`,
            row_count: tableQuery.data.total_rows,
            columns: tableQuery.data.columns,
            records: tableQuery.data.records,
          }}
          searchable={false}
          emptyMessage="No rows matched the current server-side controls."
          toolbarActions={
            <div className="table-toolbar-actions">
              <span className="table-page-summary">
                Page {Math.floor(offset / limit) + 1}
              </span>
              <button
                type="button"
                className="button-link"
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={!canGoBack}
              >
                Previous
              </button>
              <button
                type="button"
                className="button-link"
                onClick={() => setOffset(offset + limit)}
                disabled={!canGoForward}
              >
                Next
              </button>
            </div>
          }
        />
      )}
    </section>
  )
}
