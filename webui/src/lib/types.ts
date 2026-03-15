export type StatusTone = 'ok' | 'warning' | 'error' | 'neutral'

export interface TestBestModel {
  model: string
  metric: string
  value: number
}

export interface TestingSummary {
  record_count: number
  models: string[]
  splits: string[]
  primary_metric: string
  best_model?: TestBestModel | null
  best_score?: number | null
  summary_path?: string | null
}

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
  has_test_summary?: boolean
  has_analysis_index?: boolean
  has_eval_index?: boolean
  has_eval_summary?: boolean
  has_eval_report_summary?: boolean
  analysis_status?: string
  testing_status?: string
  eval_status?: string
  task_label?: string
  route?: string
  testing?: TestingSummary
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
  }
  training: {
    record_count: number
    models: string[]
    splits: string[]
    summary_path?: string | null
  }
  testing: TestingSummary
  analysis: {
    has_index: boolean
    modules: AnalysisModuleMeta[]
    index_path?: string | null
  }
  eval: {
    instance_count: number
    system_count: number
    leaderboard_rows: number
    primary_metric?: string | null
    index_path?: string | null
    summary_path?: string | null
    report_summary_path?: string | null
  }
  artifacts: Record<string, boolean>
}

export interface RunConsolePayload {
  run: RunDetail
  hero: {
    run_name: string
    task_label: string
    model_count: number
    test_model_count: number
    test_record_count: number
    analysis_module_count: number
    eval_instance_count: number
    eval_system_count: number
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
    eval_route: string
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
  tables: DashboardTable[]
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

export interface EvalSystemSummary {
  system_name: string
  system_kind?: string | null
  framework_type?: string | null
  row_count: number
  parsed_ok_rows: number
  coverage: number
  mean_latency_ms?: number | null
  total_tokens?: number | null
  total_cost_usd?: number | null
}

export interface EvalIndexRecord {
  instance_id: string
  patient_id: string
  split: string
  split_role?: string | null
  prediction_mode?: string | null
  bin_time?: string | null
  ground_truth?: unknown
  event_count?: number
  static_feature_count?: number
  evidence_path?: string | null
}

export interface EvalPayload {
  run_name: string
  status: string
  index: {
    instance_count: number
    records: EvalIndexRecord[]
  } | null
  systems: {
    records: EvalSystemSummary[]
  } | null
  report: {
    primary_metric?: string | null
    leaderboard_rows?: number
    pairwise_rows?: number
  } | null
  tables: DashboardTable[]
  highlights: string[]
}

export interface EvalInstancePayload {
  instance_id: string
  evidence: {
    patient_id?: string
    split?: string
    split_role?: string
    prediction_mode?: string
    bin_time?: string | null
    ground_truth?: unknown
    event_count?: number
    static_feature_count?: number
    events: Array<Record<string, unknown>>
    static: {
      patient_id?: string
      features: Record<string, unknown>
    }
    analysis_refs: {
      modules: Array<Record<string, unknown>>
      patient_case_matches: Array<Record<string, unknown>>
    }
  }
  outputs: Array<Record<string, unknown>>
}
