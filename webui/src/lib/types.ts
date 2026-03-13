export type StatusTone = 'ok' | 'warning' | 'error' | 'neutral'

export interface RunRecord {
  run_name: string
  run_dir: string
  schema_version: number
  task?: {
    kind?: string
    prediction_mode?: string
  }
  split?: {
    kind?: string
  }
  has_train_summary?: boolean
  has_analysis_index?: boolean
  has_cases_index?: boolean
  has_agent_predict_summary?: boolean
  has_agent_review_summary?: boolean
  mtime_unix?: number
}

export interface AnalysisModuleMeta {
  name: string
  status: string
  title?: string
  description?: string
  summary_path?: string | null
  table_names?: string[]
  plot_names?: string[]
  case_names?: string[]
  route?: string
}

export interface RunDetail {
  run_name: string
  run_dir: string
  manifest: {
    schema_version: number
    task?: Record<string, unknown>
    split?: Record<string, unknown>
    cases?: Record<string, unknown>
    agent?: Record<string, unknown>
  }
  training: {
    record_count: number
    models: string[]
    splits: string[]
    summary_path?: string | null
  }
  analysis: {
    has_index: boolean
    modules: AnalysisModuleMeta[]
    index_path?: string | null
  }
  cases: {
    case_count: number
    index_path?: string | null
  }
  agent_predict: {
    record_count: number
    predictors: string[]
    summary_path?: string | null
  }
  agent_review: {
    record_count: number
    reviewers: string[]
    summary_path?: string | null
  }
  artifacts: Record<string, boolean>
}

export interface RunConsolePayload {
  run: RunDetail
  hero: {
    run_name: string
    task_label: string
    model_count: number
    analysis_module_count: number
    case_count: number
    agent_predict_record_count: number
    agent_review_record_count: number
  }
  analysis: {
    run_name: string
    status: string
    task?: Record<string, unknown>
    modules: AnalysisModuleMeta[]
    comparison?: Record<string, unknown> | null
  }
  navigation: {
    overview_route: string
    cases_route: string
    agents_route: string
    comparison_route: string
  }
}

export interface DashboardCard {
  key: string
  title: string
  value: string | number | null
  subtitle?: string | null
  tone?: StatusTone
}

export interface DashboardChart {
  key: string
  kind: 'metric_card' | 'bar' | 'grouped_bar' | 'line' | 'heatmap' | 'ranked_table'
  title: string
  description?: string | null
  x?: string | null
  y?: string | null
  group?: string | null
  data?: Array<Record<string, unknown>>
  rows?: Array<Record<string, unknown>>
}

export interface DashboardTable {
  key: string
  title: string
  description?: string | null
  row_count: number
  columns: string[]
  records: Array<Record<string, unknown>>
}

export interface TablePage {
  key: string
  title: string
  offset: number
  limit: number
  row_count: number
  total_rows: number
  columns: string[]
  records: Array<Record<string, unknown>>
}

export interface ExplorerItem {
  key: string
  title: string
  description?: string | null
  row_count: number
  columns: string[]
  badge?: string | number | null
}

export interface FailureCaseArtifact {
  module: string
  name: string
  path: string
  row_count: number
  patient_count: number
  columns: string[]
}

export interface FailureCasePage extends TablePage {
  module: string
  name?: string | null
}

export interface PatientCasePayload {
  module: string
  patient_id: string
  n_matches: number
  matches: Array<Record<string, unknown>>
}

export interface ModuleDashboard {
  run_name: string
  module: string
  title: string
  status: string
  summary: Record<string, unknown>
  cards: DashboardCard[]
  charts: DashboardChart[]
  tables: DashboardTable[]
  case_artifacts?: FailureCaseArtifact[]
  comparison_available?: boolean
  notes?: string[]
}

export interface ComparisonPayload {
  status?: string
  run_name: string
  summary: Record<string, unknown> | null
  train_metrics?: DashboardTable | null
  agent_predict_metrics?: DashboardTable | null
  cards?: DashboardCard[]
  charts?: DashboardChart[]
  highlights?: string[]
}

export interface CohortComparison {
  split: string
  left_role: string
  right_role: string
  left: Record<string, unknown>
  right: Record<string, unknown>
  deltas: Record<string, number | null>
  feature_drift_available: boolean
  top_feature_drift: Array<Record<string, unknown>>
}

export interface CasesIndexPayload {
  run_name: string
  status: string
  case_count: number
  offset: number
  limit: number
  row_count: number
  total_rows: number
  splits: string[]
  columns: string[]
  records: Array<Record<string, unknown>>
}

export interface CaseDetailPayload {
  run_name: string
  case: Record<string, unknown>
  timeline: DashboardTable
  predictions: DashboardTable
  static: {
    feature_count: number
    table: DashboardTable
  }
  analysis_refs: {
    module_count: number
    patient_case_match_count: number
    modules: DashboardTable
    patient_case_matches: DashboardTable
  }
}

export interface AgentTaskPayload {
  status: string
  summary: Record<string, unknown> | null
  cards: DashboardCard[]
  table: DashboardTable | null
  actors: string[]
  splits: string[]
  detail_available: boolean
}

export interface AgentRowsPayload {
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
  table: DashboardTable | null
}

export interface AgentsPayload {
  run_name: string
  predict: AgentTaskPayload
  review: AgentTaskPayload
}
