import { useDeferredValue, useEffect, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import type { DashboardTable, ExplorerItem, TablePage } from '../lib/types'
import { formatNumber } from '../lib/format'
import { DataTable } from './DataTable'
import { EmptyState } from './EmptyState'
import { LoadingPanel } from './LoadingPanel'
import { TablePaginationControls } from './TablePaginationControls'

interface ServerTableExplorerProps {
  items: ExplorerItem[]
  queryKeyPrefix: readonly unknown[]
  fetchPage: (
    itemKey: string,
    options: {
      limit: number
      offset: number
      sortBy: string | null
      sortDir: 'asc' | 'desc'
      filterCol: string | null
      filterValue: string | null
    },
  ) => Promise<TablePage>
  pickerEyebrow: string
  pickerTitle: string
  pickerDescription: string
  controlsEyebrow: string
  controlsTitle: string
  controlsDescription: string
  selectionLabel: string
  noItemsTitle: string
  noItemsDescription: string
  noMatchesMessage: string
  buildTable?: (page: TablePage, item: ExplorerItem) => DashboardTable
}

const DEFAULT_PAGE_SIZE = 25

function defaultBuildTable(page: TablePage): DashboardTable {
  return {
    key: page.key,
    title: page.title,
    description: `Showing ${page.total_rows === 0 ? 0 : page.offset + 1}-${page.total_rows === 0 ? 0 : page.offset + page.row_count} of ${formatNumber(page.total_rows)} rows`,
    row_count: page.total_rows,
    columns: page.columns,
    records: page.records,
  }
}

export function ServerTableExplorer({
  items,
  queryKeyPrefix,
  fetchPage,
  pickerEyebrow,
  pickerTitle,
  pickerDescription,
  controlsEyebrow,
  controlsTitle,
  controlsDescription,
  selectionLabel,
  noItemsTitle,
  noItemsDescription,
  noMatchesMessage,
  buildTable = defaultBuildTable,
}: ServerTableExplorerProps) {
  const [selectedItemKey, setSelectedItemKey] = useState<string>(items[0]?.key ?? '')
  const [sortBy, setSortBy] = useState('')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [filterCol, setFilterCol] = useState('')
  const [filterValue, setFilterValue] = useState('')
  const [limit, setLimit] = useState(DEFAULT_PAGE_SIZE)
  const [offset, setOffset] = useState(0)
  const deferredFilterValue = useDeferredValue(filterValue)

  useEffect(() => {
    if (items.length === 0) {
      if (selectedItemKey !== '') {
        setSelectedItemKey('')
      }
      return
    }
    if (!items.some((item) => item.key === selectedItemKey)) {
      setSelectedItemKey(items[0]?.key ?? '')
    }
  }, [items, selectedItemKey])

  useEffect(() => {
    setSortBy('')
    setSortDir('desc')
    setFilterCol('')
    setFilterValue('')
    setLimit(DEFAULT_PAGE_SIZE)
    setOffset(0)
  }, [selectedItemKey])

  useEffect(() => {
    setOffset(0)
  }, [sortBy, sortDir, filterCol, filterValue, limit])

  const selectedItem = useMemo(
    () => items.find((item) => item.key === selectedItemKey) ?? null,
    [items, selectedItemKey],
  )
  const availableColumns = selectedItem?.columns ?? []

  const pageQuery = useQuery({
    queryKey: [
      ...queryKeyPrefix,
      selectedItemKey,
      limit,
      offset,
      sortBy,
      sortDir,
      filterCol,
      deferredFilterValue,
    ],
    queryFn: () =>
      fetchPage(selectedItemKey, {
        limit,
        offset,
        sortBy: sortBy || null,
        sortDir,
        filterCol: filterCol || null,
        filterValue: deferredFilterValue || null,
      }),
    enabled: selectedItemKey.length > 0,
  })

  if (items.length === 0) {
    return <EmptyState title={noItemsTitle} description={noItemsDescription} />
  }

  const totalRows = pageQuery.data?.total_rows ?? selectedItem?.row_count ?? 0
  const pageRowCount = pageQuery.data?.row_count ?? 0
  const pageStart = totalRows === 0 ? 0 : offset + 1
  const pageEnd = totalRows === 0 ? 0 : offset + pageRowCount

  return (
    <section className="page-stack">
      <section className="two-column-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">{pickerEyebrow}</p>
              <h2>{pickerTitle}</h2>
              <p className="panel-copy">{pickerDescription}</p>
            </div>
          </div>
          <div className="artifact-list">
            {items.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`artifact-button ${selectedItemKey === item.key ? 'active' : ''}`}
                onClick={() => setSelectedItemKey(item.key)}
              >
                <div>
                  <strong>{item.title}</strong>
                  <div className="artifact-meta">{item.description ?? `${formatNumber(item.row_count)} rows`}</div>
                </div>
                <span>{item.badge ?? formatNumber(item.row_count)}</span>
              </button>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">{controlsEyebrow}</p>
              <h2>{controlsTitle}</h2>
              <p className="panel-copy">{controlsDescription}</p>
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
              <span>{selectionLabel}</span>
              <strong>{selectedItem?.title ?? '—'}</strong>
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

      {pageQuery.isLoading ? (
        <LoadingPanel label="Loading table rows" />
      ) : pageQuery.isError || !pageQuery.data || !selectedItem ? (
        <EmptyState
          title="Table browser unavailable"
          description={pageQuery.error instanceof Error ? pageQuery.error.message : 'Unable to load rows.'}
        />
      ) : (
        <DataTable
          table={buildTable(pageQuery.data, selectedItem)}
          searchable={false}
          emptyMessage={noMatchesMessage}
          toolbarActions={
            <div className="table-toolbar-actions">
              <TablePaginationControls
                offset={offset}
                limit={limit}
                totalRows={totalRows}
                onPrevious={() => setOffset(Math.max(0, offset - limit))}
                onNext={() => setOffset(offset + limit)}
              />
            </div>
          }
        />
      )}
    </section>
  )
}
