import { fetchComparisonTable } from '../lib/api'
import type { DashboardTable, ExplorerItem, TablePage } from '../lib/types'
import { formatNumber } from '../lib/format'
import { ServerTableExplorer } from './ServerTableExplorer'

interface ComparisonTableExplorerProps {
  runName: string
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

export function ComparisonTableExplorer({ runName, tables }: ComparisonTableExplorerProps) {
  const items = buildExplorerItems(tables)

  return (
    <ServerTableExplorer
      items={items}
      queryKeyPrefix={['comparison-table', runName]}
      fetchPage={(itemKey, options) => fetchComparisonTable(runName, itemKey, options)}
      pickerEyebrow="Compare Tables"
      pickerTitle="Choose a saved comparison artifact"
      pickerDescription="Switch across compare-run tables without leaving the comparison console."
      controlsEyebrow="Server Controls"
      controlsTitle="Sort, filter, and paginate the full delta table"
      controlsDescription="These controls query the backend comparison artifacts instead of only searching the preview rows."
      selectionLabel="Selected table"
      noItemsTitle="No compare-run tables"
      noItemsDescription="This comparison snapshot produced summary cards or charts, but no saved tabular deltas were available to browse."
      noMatchesMessage="No comparison rows matched the current server-side controls."
      buildTable={buildTable}
    />
  )
}
