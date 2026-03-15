import type {
  AnalysisModuleMeta,
  CohortComparison,
  ComparisonPayload,
  DashboardCard,
  DashboardChart,
  DashboardTable,
  EvalIndexRecord,
  EvalInstancePayload,
  EvalPayload,
  EvalSystemSummary,
  FailureCaseArtifact,
  FailureCasePage,
  ModuleDashboard,
  PatientCasePayload,
  RunConsolePayload,
  RunDetail,
  RunRecord,
  TablePage,
  TestBestModel,
  TestingSummary,
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

const EMPTY_TESTING_SUMMARY: TestingSummary = {
  record_count: 0,
  models: [],
  splits: [],
  primary_metric: '—',
  best_model: null,
  best_score: null,
  summary_path: null,
}

function asObject(value: unknown): Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string' && item.length > 0)
}

function asNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : fallback
}

function normalizeTestBestModel(value: unknown): TestBestModel | null {
  const record = asObject(value)
  if (typeof record.model !== 'string' || typeof record.metric !== 'string') {
    return null
  }
  return {
    model: record.model,
    metric: record.metric,
    value: asNumber(record.value, 0),
  }
}

function normalizeTestingSummary(value: unknown): TestingSummary {
  const record = asObject(value)
  return {
    record_count: asNumber(record.record_count, 0),
    models: asStringArray(record.models),
    splits: asStringArray(record.splits),
    primary_metric:
      typeof record.primary_metric === 'string' && record.primary_metric.length > 0
        ? record.primary_metric
        : EMPTY_TESTING_SUMMARY.primary_metric,
    best_model: normalizeTestBestModel(record.best_model),
    best_score: record.best_score == null ? null : asNumber(record.best_score, 0),
    summary_path: typeof record.summary_path === 'string' ? record.summary_path : null,
  }
}

function normalizeModule(module: unknown): AnalysisModuleMeta {
  const record = asObject(module)
  return {
    name: typeof record.name === 'string' ? record.name : 'unknown',
    status: typeof record.status === 'string' ? record.status : 'unknown',
    title: typeof record.title === 'string' ? record.title : undefined,
    description: typeof record.description === 'string' ? record.description : undefined,
    summary_path: typeof record.summary_path === 'string' ? record.summary_path : null,
    table_names: asStringArray(record.table_names),
    plot_names: asStringArray(record.plot_names),
    case_names: asStringArray(record.case_names),
    route: typeof record.route === 'string' ? record.route : undefined,
  }
}

function normalizeRunRecord(run: unknown): RunRecord {
  const record = asObject(run)
  const testing = normalizeTestingSummary(record.testing)
  const hasTestSummary =
    typeof record.has_test_summary === 'boolean'
      ? record.has_test_summary
      : Boolean(testing.summary_path || testing.record_count > 0)

  return {
    run_name: typeof record.run_name === 'string' ? record.run_name : 'unknown',
    run_dir: typeof record.run_dir === 'string' ? record.run_dir : '',
    schema_version: asNumber(record.schema_version, 0),
    task: asObject(record.task) as RunRecord['task'],
    split: asObject(record.split) as RunRecord['split'],
    has_train_summary: Boolean(record.has_train_summary),
    has_test_summary: hasTestSummary,
    has_analysis_index: Boolean(record.has_analysis_index),
    has_eval_index: Boolean(record.has_eval_index),
    has_eval_summary: Boolean(record.has_eval_summary),
    has_eval_report_summary: Boolean(record.has_eval_report_summary),
    analysis_status: typeof record.analysis_status === 'string' ? record.analysis_status : undefined,
    testing_status:
      typeof record.testing_status === 'string' ? record.testing_status : hasTestSummary ? 'ready' : 'pending',
    eval_status: typeof record.eval_status === 'string' ? record.eval_status : undefined,
    task_label: typeof record.task_label === 'string' ? record.task_label : undefined,
    route: typeof record.route === 'string' ? record.route : undefined,
    testing,
    mtime_unix: record.mtime_unix == null ? undefined : asNumber(record.mtime_unix, 0),
  }
}

function normalizeRunDetail(run: unknown): RunDetail {
  const record = asObject(run)
  const analysisModules = asObject(record.analysis).modules
  return {
    run_name: typeof record.run_name === 'string' ? record.run_name : 'unknown',
    run_dir: typeof record.run_dir === 'string' ? record.run_dir : '',
    manifest: {
      schema_version: asNumber(asObject(record.manifest).schema_version, 0),
      task: asObject(asObject(record.manifest).task),
      split: asObject(asObject(record.manifest).split),
    },
    training: {
      record_count: asNumber(asObject(record.training).record_count, 0),
      models: asStringArray(asObject(record.training).models),
      splits: asStringArray(asObject(record.training).splits),
      summary_path:
        typeof asObject(record.training).summary_path === 'string' ? String(asObject(record.training).summary_path) : null,
    },
    testing: normalizeTestingSummary(record.testing),
    analysis: {
      has_index: Boolean(asObject(record.analysis).has_index),
      modules: Array.isArray(analysisModules) ? analysisModules.map(normalizeModule) : [],
      index_path:
        typeof asObject(record.analysis).index_path === 'string' ? String(asObject(record.analysis).index_path) : null,
    },
    eval: {
      instance_count: asNumber(asObject(record.eval).instance_count, 0),
      system_count: asNumber(asObject(record.eval).system_count, 0),
      leaderboard_rows: asNumber(asObject(record.eval).leaderboard_rows, 0),
      primary_metric:
        typeof asObject(record.eval).primary_metric === 'string' ? String(asObject(record.eval).primary_metric) : null,
      index_path: typeof asObject(record.eval).index_path === 'string' ? String(asObject(record.eval).index_path) : null,
      summary_path:
        typeof asObject(record.eval).summary_path === 'string' ? String(asObject(record.eval).summary_path) : null,
      report_summary_path:
        typeof asObject(record.eval).report_summary_path === 'string'
          ? String(asObject(record.eval).report_summary_path)
          : null,
    },
    artifacts: asObject(record.artifacts) as Record<string, boolean>,
  }
}

function normalizeRunConsole(payload: unknown): RunConsolePayload {
  const record = asObject(payload)
  const analysisRecord = asObject(record.analysis)
  const run = normalizeRunDetail(record.run)
  const analysisModules = Array.isArray(analysisRecord.modules)
    ? analysisRecord.modules.map(normalizeModule)
    : run.analysis.modules

  return {
    run: {
      ...run,
      analysis: {
        ...run.analysis,
        modules: analysisModules,
      },
    },
    hero: {
      run_name: typeof asObject(record.hero).run_name === 'string' ? String(asObject(record.hero).run_name) : run.run_name,
      task_label:
        typeof asObject(record.hero).task_label === 'string' ? String(asObject(record.hero).task_label) : 'Unknown task',
      model_count: asNumber(asObject(record.hero).model_count, run.training.models.length),
      test_model_count: asNumber(asObject(record.hero).test_model_count, run.testing.models.length),
      test_record_count: asNumber(asObject(record.hero).test_record_count, run.testing.record_count),
      analysis_module_count: asNumber(asObject(record.hero).analysis_module_count, analysisModules.length),
      eval_instance_count: asNumber(asObject(record.hero).eval_instance_count, run.eval.instance_count),
      eval_system_count: asNumber(asObject(record.hero).eval_system_count, run.eval.system_count),
    },
    analysis: {
      run_name: typeof analysisRecord.run_name === 'string' ? analysisRecord.run_name : run.run_name,
      status: typeof analysisRecord.status === 'string' ? analysisRecord.status : run.analysis.has_index ? 'ok' : 'missing',
      task: asObject(analysisRecord.task),
      modules: analysisModules,
      comparison: analysisRecord.comparison != null ? asObject(analysisRecord.comparison) : null,
    },
    navigation: {
      overview_route:
        typeof asObject(record.navigation).overview_route === 'string'
          ? String(asObject(record.navigation).overview_route)
          : `/runs/${encodeURIComponent(run.run_name)}`,
      eval_route:
        typeof asObject(record.navigation).eval_route === 'string'
          ? String(asObject(record.navigation).eval_route)
          : `/runs/${encodeURIComponent(run.run_name)}/eval`,
      comparison_route:
        typeof asObject(record.navigation).comparison_route === 'string'
          ? String(asObject(record.navigation).comparison_route)
          : `/runs/${encodeURIComponent(run.run_name)}/comparison`,
    },
  }
}

interface RawKpiCard {
  id: string
  label: string
  value: string | number | null
  format?: string
}

interface RawDashboard {
  run_name: string
  comparison_available?: boolean
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

function mapTables(
  tables: Array<{
    name: string
    title: string
    row_count: number
    columns: Array<{ name: string }>
    preview: Array<Record<string, unknown>>
  }>,
): DashboardTable[] {
  return tables.map((table) => ({
    key: table.name,
    title: table.title,
    description: `${table.row_count} rows`,
    row_count: table.row_count,
    columns: table.columns.map((column) => column.name),
    records: table.preview,
  }))
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

function normalizeEvalIndexRecord(value: unknown): EvalIndexRecord {
  const record = asObject(value)
  return {
    instance_id: typeof record.instance_id === 'string' ? record.instance_id : 'unknown',
    patient_id: typeof record.patient_id === 'string' ? record.patient_id : 'unknown',
    split: typeof record.split === 'string' ? record.split : 'unknown',
    split_role: typeof record.split_role === 'string' ? record.split_role : null,
    prediction_mode: typeof record.prediction_mode === 'string' ? record.prediction_mode : null,
    bin_time: typeof record.bin_time === 'string' ? record.bin_time : null,
    ground_truth: record.ground_truth,
    event_count: record.event_count == null ? undefined : asNumber(record.event_count, 0),
    static_feature_count:
      record.static_feature_count == null ? undefined : asNumber(record.static_feature_count, 0),
    evidence_path: typeof record.evidence_path === 'string' ? record.evidence_path : null,
  }
}

function normalizeEvalSystemSummary(value: unknown): EvalSystemSummary {
  const record = asObject(value)
  return {
    system_name: typeof record.system_name === 'string' ? record.system_name : 'unknown',
    system_kind: typeof record.system_kind === 'string' ? record.system_kind : null,
    framework_type: typeof record.framework_type === 'string' ? record.framework_type : null,
    row_count: asNumber(record.row_count, 0),
    parsed_ok_rows: asNumber(record.parsed_ok_rows, 0),
    coverage: asNumber(record.coverage, 0),
    mean_latency_ms: record.mean_latency_ms == null ? null : asNumber(record.mean_latency_ms, 0),
    total_tokens: record.total_tokens == null ? null : asNumber(record.total_tokens, 0),
    total_cost_usd: record.total_cost_usd == null ? null : asNumber(record.total_cost_usd, 0),
  }
}

export async function fetchRuns(): Promise<RunRecord[]> {
  const payload = await request<{ runs: RunRecord[] }>('/runs')
  return (payload.runs ?? []).map(normalizeRunRecord)
}

export async function fetchRunConsole(runName: string): Promise<RunConsolePayload> {
  return normalizeRunConsole(await request<RunConsolePayload>(`/runs/${encodeURIComponent(runName)}`))
}

export async function fetchModuleDashboard(runName: string, moduleName: string): Promise<ModuleDashboard> {
  const payload = await request<RawDashboard>(
    `/runs/${encodeURIComponent(runName)}/analysis/${encodeURIComponent(moduleName)}/dashboard`,
  )
  return {
    run_name: payload.run_name,
    module: payload.module.name,
    title: payload.module.title,
    status: payload.module.status,
    summary: payload.module.summary,
    cards: mapCards(payload.module.kpis ?? []),
    charts: mapCharts(payload.charts ?? [], payload.highlights ?? []),
    tables: mapTables(payload.tables ?? []),
    case_artifacts: payload.drilldowns?.case_artifacts ?? [],
    comparison_available: payload.comparison_available ?? false,
    notes: (payload.highlights ?? []).map((item) => `${item.title}: ${item.body}`),
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

export async function fetchCaseArtifacts(runName: string, moduleName: string): Promise<FailureCaseArtifact[]> {
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
  const summaryCards: DashboardCard[] =
    payload.summary == null
      ? []
      : Object.entries(payload.summary)
          .filter((entry): entry is [string, number] => typeof entry[1] === 'number')
          .map(([key, value]) => ({
            key,
            title: key.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase()),
            value,
            tone: 'neutral',
          }))

  return {
    status: payload.status,
    run_name: payload.run_name,
    summary: payload.summary,
    tables: mapTables(payload.tables ?? []),
    cards: summaryCards,
    charts: (payload.charts ?? []).map((chart) => ({
      key: chart.id,
      kind: chart.kind,
      title: chart.title,
      x: chart.x_key,
      y: chart.y_key,
      group: chart.group_key,
      data: chart.data,
    })),
    highlights: (payload.highlights ?? []).map((item) => `${item.title}: ${item.body}`),
  }
}

export async function fetchComparisonTable(
  runName: string,
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
  }>(`/runs/${encodeURIComponent(runName)}/comparison/tables/${encodeURIComponent(tableName)}${query}`)
  return mapTablePagePayload(payload)
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
  const payload = await request<{ comparison: CohortComparison }>(
    `/runs/${encodeURIComponent(runName)}/cohorts/compare${query}`,
  )
  return payload.comparison
}

export async function fetchEval(runName: string): Promise<EvalPayload> {
  const payload = await request<{
    run_name: string
    status: string
    index: Record<string, unknown> | null
    systems: Record<string, unknown> | null
    report: Record<string, unknown> | null
    tables: Array<{
      name: string
      title: string
      row_count: number
      columns: Array<{ name: string }>
      preview: Array<Record<string, unknown>>
    }>
    highlights: Array<{ title: string; body: string }>
  }>(`/runs/${encodeURIComponent(runName)}/eval`)

  const index = payload.index == null ? null : {
    instance_count: asNumber(asObject(payload.index).instance_count, 0),
    records: Array.isArray(asObject(payload.index).records)
      ? (asObject(payload.index).records as unknown[]).map(normalizeEvalIndexRecord)
      : [],
  }
  const systems = payload.systems == null ? null : {
    records: Array.isArray(asObject(payload.systems).records)
      ? (asObject(payload.systems).records as unknown[]).map(normalizeEvalSystemSummary)
      : [],
  }
  const report = payload.report == null ? null : {
    primary_metric:
      typeof asObject(payload.report).primary_metric === 'string'
        ? String(asObject(payload.report).primary_metric)
        : null,
    leaderboard_rows: asNumber(asObject(payload.report).leaderboard_rows, 0),
    pairwise_rows: asNumber(asObject(payload.report).pairwise_rows, 0),
  }

  return {
    run_name: payload.run_name,
    status: payload.status,
    index,
    systems,
    report,
    tables: mapTables(payload.tables ?? []),
    highlights: (payload.highlights ?? []).map((item) => `${item.title}: ${item.body}`),
  }
}

export async function fetchEvalTable(
  runName: string,
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
  }>(`/runs/${encodeURIComponent(runName)}/eval/tables/${encodeURIComponent(tableName)}${query}`)
  return mapTablePagePayload(payload)
}

export async function fetchEvalInstance(runName: string, instanceId: string): Promise<EvalInstancePayload> {
  const payload = await request<{ instance: EvalInstancePayload }>(
    `/runs/${encodeURIComponent(runName)}/eval/instances/${encodeURIComponent(instanceId)}`,
  )
  return payload.instance
}

export async function fetchEvalTrace(
  runName: string,
  systemName: string,
  options: {
    limit?: number
    offset?: number
    stage?: string | null
    role?: string | null
    roundIndex?: number | null
  } = {},
): Promise<TablePage> {
  const query = buildQueryString({
    limit: options.limit ?? 25,
    offset: options.offset ?? 0,
    stage: options.stage,
    role: options.role,
    round_index: options.roundIndex,
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
  }>(`/runs/${encodeURIComponent(runName)}/eval/traces/${encodeURIComponent(systemName)}${query}`)
  return mapTablePagePayload(payload)
}
