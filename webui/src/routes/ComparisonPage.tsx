import { useEffect, useState } from 'react'
import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchComparison, fetchCohortComparison, fetchRunConsole } from '../lib/api'
import type { DashboardCard, DashboardChart, DashboardTable } from '../lib/types'
import { titleCase } from '../lib/format'
import { ChartPanel } from '../ui/ChartPanel'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'

type CohortRole = 'train' | 'val' | 'test'

const DEFAULT_TOP_K = 10

function buildCohortCards(comparison: {
  split: string
  left_role: string
  right_role: string
  deltas: Record<string, number | null>
  top_feature_drift: Array<Record<string, unknown>>
}): DashboardCard[] {
  return [
    {
      key: 'cohort_split',
      title: 'Split',
      value: comparison.split,
      tone: 'neutral',
    },
    {
      key: 'cohort_pair',
      title: 'Role Pair',
      value: `${titleCase(comparison.left_role)} -> ${titleCase(comparison.right_role)}`,
      tone: 'neutral',
    },
    {
      key: 'label_rate_delta',
      title: 'Label Rate Delta',
      value: comparison.deltas.label_rate_delta ?? '—',
      tone: 'neutral',
    },
    {
      key: 'patient_delta',
      title: 'Patient Delta',
      value: comparison.deltas.n_patients_delta ?? '—',
      tone: 'neutral',
    },
    {
      key: 'sample_delta',
      title: 'Sample Delta',
      value: comparison.deltas.n_samples_delta ?? '—',
      tone: 'neutral',
    },
    {
      key: 'feature_drift_rows',
      title: 'Top Drift Rows',
      value: comparison.top_feature_drift.length,
      tone: 'neutral',
    },
  ]
}

function buildCohortMetricTable(comparison: {
  left: Record<string, unknown>
  right: Record<string, unknown>
  deltas: Record<string, number | null>
}): DashboardTable {
  const records = Object.entries(comparison.deltas).map(([key, delta]) => {
    const metric = key.replace(/_delta$/, '')
    return {
      metric,
      left: comparison.left[metric] ?? null,
      right: comparison.right[metric] ?? null,
      delta,
    }
  })
  return {
    key: 'cohort_metric_deltas',
    title: 'Cohort Metric Deltas',
    description: `${records.length} aligned metrics`,
    row_count: records.length,
    columns: ['metric', 'left', 'right', 'delta'],
    records,
  }
}

function buildCohortInfoTable(comparison: {
  split: string
  left_role: string
  right_role: string
  left: Record<string, unknown>
  right: Record<string, unknown>
}): DashboardTable {
  return {
    key: 'cohort_role_rows',
    title: 'Selected Cohort Rows',
    description: 'Raw cohort rows from split_roles.csv',
    row_count: 2,
    columns: ['role', 'split', 'n_patients', 'n_samples', 'n_labeled_samples', 'label_rate', 'mean_events_per_patient'],
    records: [
      {
        role: comparison.left_role,
        split: comparison.split,
        n_patients: comparison.left.n_patients ?? null,
        n_samples: comparison.left.n_samples ?? null,
        n_labeled_samples: comparison.left.n_labeled_samples ?? null,
        label_rate: comparison.left.label_rate ?? null,
        mean_events_per_patient: comparison.left.mean_events_per_patient ?? null,
      },
      {
        role: comparison.right_role,
        split: comparison.split,
        n_patients: comparison.right.n_patients ?? null,
        n_samples: comparison.right.n_samples ?? null,
        n_labeled_samples: comparison.right.n_labeled_samples ?? null,
        label_rate: comparison.right.label_rate ?? null,
        mean_events_per_patient: comparison.right.mean_events_per_patient ?? null,
      },
    ],
  }
}

function buildCohortDriftChart(comparison: {
  top_feature_drift: Array<Record<string, unknown>>
}): DashboardChart | null {
  if (comparison.top_feature_drift.length === 0) {
    return null
  }
  return {
    key: 'top_feature_drift',
    kind: 'bar',
    title: 'Top Feature Drift',
    description: 'Largest absolute train-to-holdout drift rows from cohort_analysis.',
    x: 'feature',
    y: 'abs_delta',
    data: comparison.top_feature_drift,
  }
}

export function ComparisonPage() {
  const { runName } = useParams({ from: '/runs/$runName/comparison' })
  const [selectedSplit, setSelectedSplit] = useState('')
  const [leftRole, setLeftRole] = useState<CohortRole>('train')
  const [rightRole, setRightRole] = useState<CohortRole>('test')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)

  const runConsoleQuery = useQuery({
    queryKey: ['run-console', runName],
    queryFn: () => fetchRunConsole(runName),
  })
  const comparisonQuery = useQuery({
    queryKey: ['comparison', runName],
    queryFn: () => fetchComparison(runName),
  })

  const splitOptions = runConsoleQuery.data?.run.training.splits ?? []

  useEffect(() => {
    if (splitOptions.length === 0) {
      if (selectedSplit !== '') {
        setSelectedSplit('')
      }
      return
    }
    if (!selectedSplit || !splitOptions.includes(selectedSplit)) {
      setSelectedSplit(splitOptions[0] ?? '')
    }
  }, [selectedSplit, splitOptions])

  const cohortQuery = useQuery({
    queryKey: ['cohort-compare', runName, selectedSplit, leftRole, rightRole, topK],
    queryFn: () =>
      fetchCohortComparison(runName, {
        split: selectedSplit || null,
        leftRole,
        rightRole,
        topK,
      }),
    enabled: runConsoleQuery.data != null && (splitOptions.length === 0 || selectedSplit.length > 0),
  })

  if (runConsoleQuery.isLoading || comparisonQuery.isLoading) {
    return <LoadingPanel label="Loading comparison console" />
  }

  if (runConsoleQuery.isError || !runConsoleQuery.data) {
    return (
      <EmptyState
        title="Comparison console unavailable"
        description={runConsoleQuery.error instanceof Error ? runConsoleQuery.error.message : 'Unable to load the run.'}
      />
    )
  }

  const comparison = comparisonQuery.data
  const cohortComparison = cohortQuery.data
  const compareRunAvailable = comparison?.status === 'ok'
  const cohortCards = cohortComparison ? buildCohortCards(cohortComparison) : []
  const cohortMetricTable = cohortComparison ? buildCohortMetricTable(cohortComparison) : null
  const cohortInfoTable = cohortComparison ? buildCohortInfoTable(cohortComparison) : null
  const cohortDriftTable: DashboardTable | null = cohortComparison
    ? {
        key: 'cohort_feature_drift',
        title: 'Top Feature Drift Rows',
        description: `${cohortComparison.top_feature_drift.length} rows`,
        row_count: cohortComparison.top_feature_drift.length,
        columns: cohortComparison.top_feature_drift[0] ? Object.keys(cohortComparison.top_feature_drift[0]) : [],
        records: cohortComparison.top_feature_drift,
      }
    : null
  const cohortDriftChart = cohortComparison ? buildCohortDriftChart(cohortComparison) : null

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Comparison</p>
          <h1>{runName}</h1>
          <p className="hero-copy">
            Compare saved train deltas when they exist, and inspect cohort integrity gaps directly from cohort analysis artifacts.
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Compare Run</p>
            <h2>Artifact deltas</h2>
            <p className="panel-copy">
              Compare-run outputs are optional. When they exist, this section shows saved training and agent metric deltas.
            </p>
          </div>
        </div>

        {comparisonQuery.isError ? (
          <EmptyState
            title="Comparison artifacts unavailable"
            description={comparisonQuery.error instanceof Error ? comparisonQuery.error.message : 'Unable to load comparison artifacts.'}
          />
        ) : !compareRunAvailable || !comparison ? (
          <EmptyState
            title="No compare-run artifacts yet"
            description="Generate comparison artifacts with `oneehr analyze --compare-run` for this run."
          />
        ) : (
          <div className="page-stack">
            {comparison.cards?.length ? (
              <section className="card-grid kpi-grid">
                {comparison.cards.map((card) => (
                  <KpiCard
                    key={card.key}
                    title={card.title}
                    value={card.value}
                    subtitle={card.subtitle}
                    tone={card.tone}
                  />
                ))}
              </section>
            ) : null}

            {comparison.charts?.length ? (
              <section className="card-grid chart-grid">
                {comparison.charts.map((chart) => (
                  <ChartPanel key={chart.key} chart={chart} />
                ))}
              </section>
            ) : null}

            {comparison.train_metrics ? <DataTable table={comparison.train_metrics} /> : null}
            {comparison.agent_predict_metrics ? <DataTable table={comparison.agent_predict_metrics} /> : null}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Cohort Compare</p>
            <h2>Split and role integrity</h2>
            <p className="panel-copy">
              This view queries `cohort_analysis` artifacts directly so you can compare train, val, and test cohorts even without compare-run outputs.
            </p>
          </div>
        </div>

        <div className="filter-grid">
          <label className="table-filter">
            <span>Split</span>
            <select
              value={selectedSplit}
              onChange={(event) => setSelectedSplit(event.target.value)}
              disabled={splitOptions.length === 0}
            >
              {splitOptions.length === 0 ? (
                <option value="">No split artifacts</option>
              ) : (
                splitOptions.map((split) => (
                  <option key={split} value={split}>
                    {split}
                  </option>
                ))
              )}
            </select>
          </label>

          <label className="table-filter">
            <span>Left role</span>
            <select value={leftRole} onChange={(event) => setLeftRole(event.target.value as CohortRole)}>
              <option value="train">Train</option>
              <option value="val">Val</option>
              <option value="test">Test</option>
            </select>
          </label>

          <label className="table-filter">
            <span>Right role</span>
            <select value={rightRole} onChange={(event) => setRightRole(event.target.value as CohortRole)}>
              <option value="train">Train</option>
              <option value="val">Val</option>
              <option value="test">Test</option>
            </select>
          </label>

          <label className="table-filter">
            <span>Top K drift rows</span>
            <select value={String(topK)} onChange={(event) => setTopK(Number(event.target.value))}>
              <option value="5">5 rows</option>
              <option value="10">10 rows</option>
              <option value="20">20 rows</option>
            </select>
          </label>
        </div>

        {cohortQuery.isLoading ? (
          <LoadingPanel label="Loading cohort comparison" />
        ) : cohortQuery.isError ? (
          <EmptyState
            title="Cohort comparison unavailable"
            description={cohortQuery.error instanceof Error ? cohortQuery.error.message : 'Unable to compare cohorts for this run.'}
          />
        ) : !cohortComparison ? (
          <EmptyState
            title="No cohort analysis yet"
            description="Run `oneehr analyze --module cohort_analysis` after training to populate split-role comparisons."
          />
        ) : (
          <div className="page-stack">
            <section className="card-grid kpi-grid">
              {cohortCards.map((card) => (
                <KpiCard
                  key={card.key}
                  title={card.title}
                  value={card.value}
                  subtitle={card.subtitle}
                  tone={card.tone}
                />
              ))}
            </section>

            <section className="two-column-grid">
              {cohortMetricTable ? <DataTable table={cohortMetricTable} /> : null}
              {cohortInfoTable ? <DataTable table={cohortInfoTable} /> : null}
            </section>

            {cohortComparison.feature_drift_available && cohortDriftChart ? (
              <section className="card-grid chart-grid">
                <ChartPanel chart={cohortDriftChart} />
              </section>
            ) : (
              <EmptyState
                title="Feature drift not available for this pairing"
                description="Top feature drift is emitted for train-to-val or train-to-test comparisons when cohort drift rows were saved."
              />
            )}

            {cohortComparison.feature_drift_available && cohortDriftTable ? <DataTable table={cohortDriftTable} /> : null}
          </div>
        )}
      </section>
    </div>
  )
}
