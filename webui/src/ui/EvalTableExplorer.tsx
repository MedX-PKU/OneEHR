import { fetchEvalTable } from '../lib/api'
import type { DashboardTable, ExplorerItem, TablePage } from '../lib/types'
import { formatNumber } from '../lib/format'
import { ServerTableExplorer } from './ServerTableExplorer'

interface EvalTableExplorerProps {
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

export function EvalTableExplorer({ runName, tables }: EvalTableExplorerProps) {
  const items = buildExplorerItems(tables)

  return (
    <ServerTableExplorer
      items={items}
      queryKeyPrefix={['eval-table', runName]}
      fetchPage={(itemKey, options) => fetchEvalTable(runName, itemKey, options)}
      pickerEyebrow="Eval Tables"
      pickerTitle="Choose a saved evaluation table"
      pickerDescription="Switch across leaderboard, split metrics, and pairwise deltas without leaving the evaluation console."
      controlsEyebrow="Server Controls"
      controlsTitle="Sort, filter, and paginate the full table"
      controlsDescription="These controls query the backend evaluation artifacts instead of only searching the preview rows."
      selectionLabel="Selected table"
      noItemsTitle="No evaluation tables"
      noItemsDescription="This run has an eval payload, but no saved report tables were available to browse."
      noMatchesMessage="No evaluation rows matched the current server-side controls."
      buildTable={buildTable}
    />
  )
}
