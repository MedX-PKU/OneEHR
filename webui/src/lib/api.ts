import type {
  AgentRowsPayload,
  AgentTaskPayload,
  AgentsPayload,
  RunConsolePayload,
  CaseDetailPayload,
  CasesIndexPayload,
  DashboardCard,
  DashboardChart,
  DashboardTable,
  TablePage,
  FailureCasePage,
  ComparisonPayload,
  CohortComparison,
  FailureCaseArtifact,
  ModuleDashboard,
  PatientCasePayload,
  RunRecord,
} from './types'

const API_BASE = (import.meta.env.VITE_ONEEHR_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}/api/v1${path}`, {
    headers: {
      Accept: 'application/json',
    },
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with status ${response.status}`)
  }
  return (await response.json()) as T
}

function buildQueryString(params: Record<string, string | number | boolean | null | undefined>): string {
  const searchParams = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value == null || value === '') {
      continue
    }
    searchParams.set(key, String(value))
  }
  const query = searchParams.toString()
  return query ? `?${query}` : ''
}

export async function fetchRuns(): Promise<RunRecord[]> {
  const payload = await request<{ runs: RunRecord[] }>('/runs')
  return payload.runs
}

export async function fetchRunConsole(runName: string): Promise<RunConsolePayload> {
  return request<RunConsolePayload>(`/runs/${encodeURIComponent(runName)}`)
}

interface RawKpiCard {
  id: string
  label: string
  value: string | number | null
  format?: string
}

interface RawDashboard {
  run_name: string
  module: {
    name: string
    title: string
    status: string
    summary: Record<string, unknown>
    kpis: RawKpiCard[]
  }
  charts: Array<{
    id: string
    kind: DashboardChart['kind']
    title: string
    x_key?: string | null
    y_key?: string | null
    group_key?: string | null
    data?: Array<Record<string, unknown>>
  }>
  tables: Array<{
    name: string
    title: string
    row_count: number
    columns: Array<{ name: string }>
    preview: Array<Record<string, unknown>>
  }>
  highlights: Array<{ title: string; body: string }>
  drilldowns: {
    case_artifacts: FailureCaseArtifact[]
  }
}

interface RawTablePayload {
  name: string
  title: string
  row_count: number
  columns: Array<{ name: string }>
  preview: Array<Record<string, unknown>>
}

function mapCards(cards: RawKpiCard[]): DashboardCard[] {
  return cards.map((card) => ({
    key: card.id,
    title: card.label,
    value: card.value,
    subtitle: card.format === 'text' ? String(card.value ?? '—') : undefined,
    tone: 'neutral',
  }))
}

function mapCharts(
  charts: RawDashboard['charts'],
  highlights: Array<{ title: string; body: string }>,
): DashboardChart[] {
  return charts.map((chart) => ({
    key: chart.id,
    kind: chart.kind,
    title: chart.title,
    description: highlights.find((item) => item.title === chart.title)?.body ?? undefined,
    x: chart.x_key,
    y: chart.y_key,
    group: chart.group_key,
    data: chart.data,
  }))
}

function mapTables(tables: RawDashboard['tables']): DashboardTable[] {
  return tables.map((table) => ({
    key: table.name,
    title: table.title,
    description: `${table.row_count} rows`,
    row_count: table.row_count,
    columns: table.columns.map((column) => column.name),
    records: table.preview,
  }))
}

function mapTablePayload(table: RawTablePayload | null | undefined): DashboardTable | null {
  if (!table) {
    return null
  }
  return {
    key: table.name,
    title: table.title,
    description: `${table.row_count} rows`,
    row_count: table.row_count,
    columns: table.columns.map((column) => column.name),
    records: table.preview,
  }
}

function mapTablePagePayload(payload: {
  table: string
  title: string
  offset: number
  limit: number
  row_count: number
  total_rows: number
  columns: Array<{ name: string }>
  records: Array<Record<string, unknown>>
}): TablePage {
  return {
    key: payload.table,
    title: payload.title,
    offset: payload.offset,
    limit: payload.limit,
    row_count: payload.row_count,
    total_rows: payload.total_rows,
    columns: payload.columns.map((column) => column.name),
    records: payload.records,
  }
}

export async function fetchModuleDashboard(
  runName: string,
  moduleName: string,
): Promise<ModuleDashboard> {
  const payload = await request<RawDashboard>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/dashboard`,
  )
  return {
    run_name: payload.run_name,
    module: payload.module.name,
    title: payload.module.title,
    status: payload.module.status,
    summary: payload.module.summary,
    cards: mapCards(payload.module.kpis),
    charts: mapCharts(payload.charts, payload.highlights),
    tables: mapTables(payload.tables),
    case_artifacts: payload.drilldowns.case_artifacts,
    comparison_available: false,
    notes: payload.highlights.map((item) => `${item.title}: ${item.body}`),
  }
}

export async function fetchAnalysisTable(
  runName: string,
  moduleName: string,
  tableName: string,
  options: {
    limit?: number
    offset?: number
    sortBy?: string | null
    sortDir?: 'asc' | 'desc'
    filterCol?: string | null
    filterValue?: string | null
  } = {},
): Promise<TablePage> {
  const query = buildQueryString({
    limit: options.limit ?? 25,
    offset: options.offset ?? 0,
    sort_by: options.sortBy,
    sort_dir: options.sortDir ?? 'desc',
    filter_col: options.filterCol,
    filter_value: options.filterValue,
  })
  const payload = await request<{
    table: string
    title: string
    offset: number
    limit: number
    row_count: number
    total_rows: number
    columns: Array<{ name: string }>
    records: Array<Record<string, unknown>>
  }>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/tables/${encodeURIComponent(tableName)}${query}`,
  )
  return mapTablePagePayload(payload)
}

export async function fetchFailureCaseRows(
  runName: string,
  moduleName: string,
  artifactName: string,
  options: {
    limit?: number
    offset?: number
    sortBy?: string | null
    sortDir?: 'asc' | 'desc'
    filterCol?: string | null
    filterValue?: string | null
  } = {},
): Promise<FailureCasePage> {
  const query = buildQueryString({
    limit: options.limit ?? 25,
    offset: options.offset ?? 0,
    sort_by: options.sortBy,
    sort_dir: options.sortDir ?? 'desc',
    filter_col: options.filterCol,
    filter_value: options.filterValue,
  })
  const payload = await request<{
    module: string
    name: string
    offset: number
    limit: number
    row_count: number
    total_rows: number
    columns: Array<{ name: string }>
    records: Array<Record<string, unknown>>
  }>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/cases/${encodeURIComponent(artifactName)}${query}`,
  )
  return {
    module: payload.module,
    name: payload.name,
    key: payload.name,
    title: payload.name,
    offset: payload.offset,
    limit: payload.limit,
    row_count: payload.row_count,
    total_rows: payload.total_rows,
    columns: payload.columns.map((column) => column.name),
    records: payload.records,
  }
}

export async function fetchCaseArtifacts(
  runName: string,
  moduleName: string,
): Promise<FailureCaseArtifact[]> {
  const payload = await request<{ case_artifacts: FailureCaseArtifact[] }>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/cases`,
  )
  return payload.case_artifacts
}

export async function fetchPatientCase(
  runName: string,
  moduleName: string,
  patientId: string,
): Promise<PatientCasePayload> {
  const payload = await request<{ patient: PatientCasePayload }>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/patient-case/${encodeURIComponent(patientId)}`,
  )
  return payload.patient
}

export async function fetchComparison(runName: string): Promise<ComparisonPayload> {
  const payload = await request<{
    run_name: string
    status: string
    summary: Record<string, unknown> | null
    tables: Array<{
      name: string
      title: string
      row_count: number
      columns: Array<{ name: string }>
      preview: Array<Record<string, unknown>>
    }>
    charts: Array<{
      id: string
      kind: DashboardChart['kind']
      title: string
      x_key?: string | null
      y_key?: string | null
      group_key?: string | null
      data?: Array<Record<string, unknown>>
    }>
    highlights: Array<{ title: string; body: string }>
  }>(`/runs/${encodeURIComponent(runName)}/comparison`)
  const tableMap = new Map(payload.tables.map((table) => [table.name, table] as const))
  const summaryCards: DashboardCard[] =
    payload.summary == null
      ? []
      : Object.entries(payload.summary)
          .filter((entry): entry is [string, string | number] => {
            const value = entry[1]
            return typeof value === 'string' || typeof value === 'number'
          })
          .map(([key, value]) => ({
            key,
            title: key.replace(/_/g, ' '),
            value,
            tone: 'neutral',
          }))

  const mapTable = (name: string): DashboardTable | null => {
    const table = tableMap.get(name)
    return mapTablePayload(table)
  }

  return {
    status: payload.status,
    run_name: payload.run_name,
    summary: payload.summary,
    train_metrics: mapTable('train_metrics'),
    agent_predict_metrics: mapTable('agent_predict_metrics'),
    cards: summaryCards,
    charts: payload.charts.map((chart) => ({
      key: chart.id,
      kind: chart.kind,
      title: chart.title,
      x: chart.x_key,
      y: chart.y_key,
      group: chart.group_key,
      data: chart.data,
    })),
    highlights: payload.highlights.map((item) => `${item.title}: ${item.body}`),
  }
}

export async function fetchCohortComparison(
  runName: string,
  options: {
    split?: string | null
    leftRole?: 'train' | 'val' | 'test'
    rightRole?: 'train' | 'val' | 'test'
    topK?: number
  } = {},
): Promise<CohortComparison> {
  const query = buildQueryString({
    split: options.split,
    left_role: options.leftRole ?? 'train',
    right_role: options.rightRole ?? 'test',
    top_k: options.topK ?? 10,
  })
  const payload = await request<{
    run_name: string
    comparison: {
      split: string
      left_role: string
      right_role: string
      left: Record<string, unknown>
      right: Record<string, unknown>
      deltas: Record<string, number | null>
      feature_drift_available: boolean
      top_feature_drift: Array<Record<string, unknown>>
    }
  }>(`/runs/${encodeURIComponent(runName)}/cohorts/compare${query}`)
  return payload.comparison
}

export async function fetchCasesIndex(
  runName: string,
  options: {
    split?: string | null
    search?: string | null
    limit?: number
    offset?: number
  } = {},
): Promise<CasesIndexPayload> {
  const query = buildQueryString({
    split: options.split,
    search: options.search,
    limit: options.limit ?? 25,
    offset: options.offset ?? 0,
  })
  const payload = await request<{
    run_name: string
    status: string
    case_count: number
    offset: number
    limit: number
    row_count: number
    total_rows: number
    splits: string[]
    columns: Array<{ name: string }>
    records: Array<Record<string, unknown>>
  }>(`/runs/${encodeURIComponent(runName)}/cases${query}`)

  return {
    run_name: payload.run_name,
    status: payload.status,
    case_count: payload.case_count,
    offset: payload.offset,
    limit: payload.limit,
    row_count: payload.row_count,
    total_rows: payload.total_rows,
    splits: payload.splits,
    columns: payload.columns.map((column) => column.name),
    records: payload.records,
  }
}

export async function fetchCaseDetail(runName: string, caseId: string): Promise<CaseDetailPayload> {
  const payload = await request<{
    run_name: string
    case: Record<string, unknown>
    timeline: RawTablePayload
    predictions: RawTablePayload
    static: {
      feature_count: number
      table: RawTablePayload
    }
    analysis_refs: {
      module_count: number
      patient_case_match_count: number
      modules: RawTablePayload
      patient_case_matches: RawTablePayload
    }
  }>(`/runs/${encodeURIComponent(runName)}/cases/${encodeURIComponent(caseId)}`)

  return {
    run_name: payload.run_name,
    case: payload.case,
    timeline: mapTablePayload(payload.timeline) ?? {
      key: 'timeline',
      title: 'Case Timeline',
      row_count: 0,
      columns: [],
      records: [],
    },
    predictions: mapTablePayload(payload.predictions) ?? {
      key: 'predictions',
      title: 'Case Predictions',
      row_count: 0,
      columns: [],
      records: [],
    },
    static: {
      feature_count: payload.static.feature_count,
      table: mapTablePayload(payload.static.table) ?? {
        key: 'static_features',
        title: 'Static Features',
        row_count: 0,
        columns: [],
        records: [],
      },
    },
    analysis_refs: {
      module_count: payload.analysis_refs.module_count,
      patient_case_match_count: payload.analysis_refs.patient_case_match_count,
      modules: mapTablePayload(payload.analysis_refs.modules) ?? {
        key: 'analysis_modules',
        title: 'Analysis References',
        row_count: 0,
        columns: [],
        records: [],
      },
      patient_case_matches: mapTablePayload(payload.analysis_refs.patient_case_matches) ?? {
        key: 'patient_case_matches',
        title: 'Patient Case Matches',
        row_count: 0,
        columns: [],
        records: [],
      },
    },
  }
}

function mapAgentTaskPayload(task: {
  status: string
  summary: Record<string, unknown> | null
  cards: RawKpiCard[]
  table: RawTablePayload | null
  actors: string[]
  splits: string[]
  detail_available: boolean
}): AgentTaskPayload {
  return {
    status: task.status,
    summary: task.summary,
    cards: mapCards(task.cards),
    table: mapTablePayload(task.table),
    actors: task.actors,
    splits: task.splits,
    detail_available: task.detail_available,
  }
}

function mapAgentRowsPayload(payload: {
  run_name: string
  task_name: string
  kind: string
  status: string
  actors: string[]
  splits: string[]
  detail_available: boolean
  offset: number
  limit: number
  row_count: number
  total_rows: number
  columns: Array<{ name: string }>
  records: Array<Record<string, unknown>>
}): AgentRowsPayload {
  return {
    run_name: payload.run_name,
    task_name: payload.task_name,
    kind: payload.kind,
    status: payload.status,
    actors: payload.actors,
    splits: payload.splits,
    detail_available: payload.detail_available,
    offset: payload.offset,
    limit: payload.limit,
    row_count: payload.row_count,
    total_rows: payload.total_rows,
    table: {
      key: `agent_${payload.task_name}_${payload.kind}`,
      title: payload.kind === 'failures' ? 'Failure Rows' : 'Detailed Rows',
      description: `${payload.row_count} of ${payload.total_rows} rows`,
      row_count: payload.total_rows,
      columns: payload.columns.map((column) => column.name),
      records: payload.records,
    },
  }
}

export async function fetchAgents(runName: string): Promise<AgentsPayload> {
  const payload = await request<{
    run_name: string
    predict: {
      status: string
      summary: Record<string, unknown> | null
      cards: RawKpiCard[]
      table: RawTablePayload | null
      actors: string[]
      splits: string[]
      detail_available: boolean
    }
    review: {
      status: string
      summary: Record<string, unknown> | null
      cards: RawKpiCard[]
      table: RawTablePayload | null
      actors: string[]
      splits: string[]
      detail_available: boolean
    }
  }>(`/runs/${encodeURIComponent(runName)}/agents`)

  return {
    run_name: payload.run_name,
    predict: mapAgentTaskPayload(payload.predict),
    review: mapAgentTaskPayload(payload.review),
  }
}

export async function fetchAgentTaskRecords(
  runName: string,
  taskName: string,
  options: {
    actor?: string | null
    split?: string | null
    parsedOk?: boolean | null
    search?: string | null
    limit?: number
    offset?: number
  } = {},
): Promise<AgentRowsPayload> {
  const query = buildQueryString({
    actor: options.actor,
    split: options.split,
    parsed_ok: options.parsedOk,
    search: options.search,
    limit: options.limit ?? 100,
    offset: options.offset ?? 0,
  })
  const payload = await request<{
    run_name: string
    task_name: string
    kind: string
    status: string
    actors: string[]
    splits: string[]
    detail_available: boolean
    offset: number
    limit: number
    row_count: number
    total_rows: number
    columns: Array<{ name: string }>
    records: Array<Record<string, unknown>>
  }>(`/runs/${encodeURIComponent(runName)}/agents/${encodeURIComponent(taskName)}/records${query}`)
  return mapAgentRowsPayload(payload)
}

export async function fetchAgentTaskFailures(
  runName: string,
  taskName: string,
  options: {
    actor?: string | null
    split?: string | null
    search?: string | null
    limit?: number
    offset?: number
  } = {},
): Promise<AgentRowsPayload> {
  const query = buildQueryString({
    actor: options.actor,
    split: options.split,
    search: options.search,
    limit: options.limit ?? 100,
    offset: options.offset ?? 0,
  })
  const payload = await request<{
    run_name: string
    task_name: string
    kind: string
    status: string
    actors: string[]
    splits: string[]
    detail_available: boolean
    offset: number
    limit: number
    row_count: number
    total_rows: number
    columns: Array<{ name: string }>
    records: Array<Record<string, unknown>>
  }>(`/runs/${encodeURIComponent(runName)}/agents/${encodeURIComponent(taskName)}/failures${query}`)
  return mapAgentRowsPayload(payload)
}
