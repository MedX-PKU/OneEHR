import { fetchFailureCaseRows } from '../lib/api'
import type { DashboardTable, ExplorerItem, FailureCaseArtifact, TablePage } from '../lib/types'
import { formatNumber, titleCase } from '../lib/format'
import { ServerTableExplorer } from './ServerTableExplorer'

interface FailureCaseExplorerProps {
  runName: string
  moduleName: string
  artifacts: FailureCaseArtifact[]
}

function buildExplorerItems(artifacts: FailureCaseArtifact[]): ExplorerItem[] {
  return artifacts.map((artifact) => ({
    key: artifact.name,
    title: titleCase(artifact.name),
    description: `${formatNumber(artifact.row_count)} rows across ${formatNumber(artifact.patient_count)} patients`,
    row_count: artifact.row_count,
    columns: artifact.columns,
    badge: formatNumber(artifact.patient_count),
  }))
}

function buildTable(page: TablePage, item: ExplorerItem, moduleName: string): DashboardTable {
  return {
    key: `case-${page.key}`,
    title: `${titleCase(moduleName)} failure case rows`,
    description: `Showing ${page.total_rows === 0 ? 0 : page.offset + 1}-${page.total_rows === 0 ? 0 : page.offset + page.row_count} of ${formatNumber(page.total_rows)} rows from ${item.title}`,
    row_count: page.total_rows,
    columns: page.columns,
    records: page.records,
  }
}

export function FailureCaseExplorer({
  runName,
  moduleName,
  artifacts,
}: FailureCaseExplorerProps) {
  const items = buildExplorerItems(artifacts)

  return (
    <ServerTableExplorer
      items={items}
      queryKeyPrefix={['failure-case-rows', runName, moduleName]}
      fetchPage={(itemKey, options) => fetchFailureCaseRows(runName, moduleName, itemKey, options)}
      pickerEyebrow="Failure Explorer"
      pickerTitle="Choose a saved failure artifact"
      pickerDescription="Browse saved prediction slices and error bundles without leaving the current module."
      controlsEyebrow="Artifact Controls"
      controlsTitle="Sort, filter, and paginate failure rows"
      controlsDescription="These controls query saved case artifact rows directly from the backend."
      selectionLabel="Selected artifact"
      noItemsTitle="No failure artifacts"
      noItemsDescription="This module did not emit saved failure case artifacts for drill-down browsing."
      noMatchesMessage="No artifact rows matched the current server-side controls."
      buildTable={(page, item) => buildTable(page, item, moduleName)}
    />
  )
}
