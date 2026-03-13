import { fetchAnalysisTable } from '../lib/api'
import type { DashboardTable, ExplorerItem, TablePage } from '../lib/types'
import { formatNumber } from '../lib/format'
import { ServerTableExplorer } from './ServerTableExplorer'

interface AnalysisTableExplorerProps {
  runName: string
  moduleName: string
  tables: DashboardTable[]
}

function buildExplorerItems(tables: DashboardTable[]): ExplorerItem[] {
  return tables.map((table) => ({
    key: table.key,
    title: table.title,
    description: table.description,
    row_count: table.row_count,
    columns: table.columns,
    badge: formatNumber(table.row_count),
  }))
}

function buildTable(page: TablePage, item: ExplorerItem): DashboardTable {
  return {
    key: page.key,
    title: page.title,
    description: `Showing ${page.total_rows === 0 ? 0 : page.offset + 1}-${page.total_rows === 0 ? 0 : page.offset + page.row_count} of ${formatNumber(page.total_rows)} rows from ${item.title}`,
    row_count: page.total_rows,
    columns: page.columns,
    records: page.records,
  }
}

export function AnalysisTableExplorer({
  runName,
  moduleName,
  tables,
}: AnalysisTableExplorerProps) {
  const items = buildExplorerItems(tables)

  return (
    <ServerTableExplorer
      items={items}
      queryKeyPrefix={['analysis-table', runName, moduleName]}
      fetchPage={(itemKey, options) => fetchAnalysisTable(runName, moduleName, itemKey, options)}
      pickerEyebrow="Table Explorer"
      pickerTitle="Choose a saved analysis table"
      pickerDescription="Switch across saved artifacts without leaving the current analysis module."
      controlsEyebrow="Server Controls"
      controlsTitle="Sort, filter, and paginate the full table"
      controlsDescription="These controls query the backend instead of only searching the visible preview rows."
      selectionLabel="Selected table"
      noItemsTitle="No structured tables"
      noItemsDescription="This module produced charts and highlights, but no saved tabular artifacts were available to browse."
      noMatchesMessage="No rows matched the current server-side controls."
      buildTable={buildTable}
    />
  )
}
