import { useMemo, useState } from 'react'
import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchEval, fetchEvalInstance, fetchEvalTrace, fetchRunConsole } from '../lib/api'
import { formatIdentifierDisplay, formatMetricName, formatNameList, formatNumber, titleCase } from '../lib/format'
import type { DashboardTable } from '../lib/types'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { EvalTableExplorer } from '../ui/EvalTableExplorer'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'

function buildPreviewTable(
  key: string,
  title: string,
  records: Array<Record<string, unknown>>,
): DashboardTable {
  return {
    key,
    title,
    description: `${records.length} rows`,
    row_count: records.length,
    columns: records[0] ? Object.keys(records[0]) : [],
    records,
  }
}

export function EvalPage() {
  const { runName } = useParams({ from: '/runs/$runName/eval' })
  const [selectedInstanceId, setSelectedInstanceId] = useState('')
  const [selectedSystemName, setSelectedSystemName] = useState('')

  const runConsoleQuery = useQuery({
    queryKey: ['run-console', runName],
    queryFn: () => fetchRunConsole(runName),
  })
  const evalQuery = useQuery({
    queryKey: ['eval', runName],
    queryFn: () => fetchEval(runName),
  })

  const instanceOptions = evalQuery.data?.index?.records ?? []
  const systemOptions = evalQuery.data?.systems?.records ?? []
  const activeInstanceId =
    selectedInstanceId && instanceOptions.some((record) => record.instance_id === selectedInstanceId)
      ? selectedInstanceId
      : (instanceOptions[0]?.instance_id ?? '')
  const activeSystemName =
    selectedSystemName && systemOptions.some((record) => record.system_name === selectedSystemName)
      ? selectedSystemName
      : (systemOptions[0]?.system_name ?? '')

  const instanceQuery = useQuery({
    queryKey: ['eval-instance', runName, activeInstanceId],
    queryFn: () => fetchEvalInstance(runName, activeInstanceId),
    enabled: activeInstanceId.length > 0,
  })
  const traceQuery = useQuery({
    queryKey: ['eval-trace', runName, activeSystemName],
    queryFn: () => fetchEvalTrace(runName, activeSystemName, { limit: 25 }),
    enabled: activeSystemName.length > 0,
  })

  const outputTable = useMemo(() => {
    const outputs = instanceQuery.data?.outputs ?? []
    return buildPreviewTable('eval_outputs', 'System Outputs', outputs)
  }, [instanceQuery.data])
  const eventTable = useMemo(() => {
    const events = instanceQuery.data?.evidence.events ?? []
    return buildPreviewTable('eval_events', 'Evidence Events', events)
  }, [instanceQuery.data])
  const staticFeatureTable = useMemo(() => {
    const features = Object.entries(instanceQuery.data?.evidence.static.features ?? {}).map(([feature, value]) => ({
      feature,
      value,
    }))
    return buildPreviewTable('eval_static', 'Static Features', features)
  }, [instanceQuery.data])
  const traceTable = useMemo(() => {
    if (!traceQuery.data) {
      return null
    }
    return {
      key: traceQuery.data.key,
      title: traceQuery.data.title,
      description: `Showing ${traceQuery.data.total_rows === 0 ? 0 : traceQuery.data.offset + 1}-${traceQuery.data.total_rows === 0 ? 0 : traceQuery.data.offset + traceQuery.data.row_count} of ${formatNumber(traceQuery.data.total_rows)} trace rows`,
      row_count: traceQuery.data.total_rows,
      columns: traceQuery.data.columns,
      records: traceQuery.data.records,
    }
  }, [traceQuery.data])

  if (runConsoleQuery.isLoading || evalQuery.isLoading) {
    return <LoadingPanel label="Loading evaluation console" />
  }

  if (runConsoleQuery.isError || !runConsoleQuery.data) {
    return (
      <EmptyState
        title="Evaluation console unavailable"
        description={runConsoleQuery.error instanceof Error ? runConsoleQuery.error.message : 'Unable to load the run.'}
      />
    )
  }

  if (evalQuery.isError) {
    return (
      <EmptyState
        title="Evaluation payload unavailable"
        description={evalQuery.error instanceof Error ? evalQuery.error.message : 'Unable to load evaluation artifacts.'}
      />
    )
  }

  const evalPayload = evalQuery.data
  if (!evalPayload || evalPayload.status !== 'ok' || !evalPayload.index || !evalPayload.systems || !evalPayload.report) {
    return (
      <EmptyState
        title="No unified evaluation artifacts yet"
        description="Run `oneehr eval build`, `oneehr eval run`, and `oneehr eval report` to populate reproducible evaluation artifacts for models and frameworks."
      />
    )
  }

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Evaluation</p>
          <h1 className="entity-title identifier-text">{formatIdentifierDisplay(runName)}</h1>
          <p className="hero-copy">
            Unified, reproducible evaluation across trained models, single-LLM systems, and multi-agent frameworks.
          </p>
        </div>
      </section>

      <section className="card-grid kpi-grid">
        <KpiCard
          key="eval_instances"
          title="Eval instances"
          value={evalPayload.index.instance_count}
          subtitle={`${formatNameList(runConsoleQuery.data.run.training.splits)} splits`}
        />
        <KpiCard
          key="eval_systems"
          title="Systems"
          value={evalPayload.systems.records.length}
          subtitle={`${runConsoleQuery.data.run.eval.system_count} executed`}
        />
        <KpiCard
          key="leaderboard_rows"
          title="Leaderboard rows"
          value={evalPayload.report.leaderboard_rows ?? 0}
          subtitle={formatMetricName(evalPayload.report.primary_metric)}
        />
        <KpiCard
          key="system_kinds"
          title="System kinds"
          value={new Set(evalPayload.systems.records.map((record) => record.system_kind ?? 'unknown')).size}
          subtitle="Trained models and frameworks"
        />
      </section>

      {evalPayload.highlights.length > 0 ? (
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Highlights</p>
              <h2>Evaluation summary</h2>
              <p className="panel-copy">
                Saved leaderboard and pairwise artifacts promoted into a quick readout.
              </p>
            </div>
          </div>
          <div className="artifact-list">
            {evalPayload.highlights.map((highlight) => (
              <div key={highlight} className="artifact-button">
                <div className="artifact-meta">{highlight}</div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Systems</p>
            <h2>Execution coverage</h2>
            <p className="panel-copy">
              Each row represents one evaluated system under the frozen instance index.
            </p>
          </div>
        </div>
        <DataTable
          table={{
            key: 'eval_systems',
            title: 'Executed Systems',
            description: `${evalPayload.systems.records.length} systems`,
            row_count: evalPayload.systems.records.length,
            columns: evalPayload.systems.records[0] ? Object.keys(evalPayload.systems.records[0]) : [],
            records: evalPayload.systems.records.map(
              (record): Record<string, unknown> => ({ ...record }),
            ),
          }}
        />
      </section>

      <EvalTableExplorer runName={runName} tables={evalPayload.tables} />

      <section className="two-column-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Instance Inspector</p>
              <h2>Evidence and outputs</h2>
              <p className="panel-copy">
                Inspect the frozen evidence bundle and aligned outputs for one evaluation instance.
              </p>
            </div>
          </div>

          <label className="table-filter wide">
            <span>Instance</span>
            <select value={activeInstanceId} onChange={(event) => setSelectedInstanceId(event.target.value)}>
              {instanceOptions.map((record) => (
                <option key={record.instance_id} value={record.instance_id}>
                  {record.instance_id} · {record.patient_id} · {record.split}
                </option>
              ))}
            </select>
          </label>

          {instanceQuery.isLoading ? (
            <LoadingPanel label="Loading instance payload" />
          ) : instanceQuery.isError || !instanceQuery.data ? (
            <EmptyState
              title="Instance payload unavailable"
              description={instanceQuery.error instanceof Error ? instanceQuery.error.message : 'Unable to load the selected instance.'}
            />
          ) : (
            <div className="page-stack">
              <div className="detail-grid">
                <div>
                  <span>Patient</span>
                  <strong>{instanceQuery.data.evidence.patient_id ?? '—'}</strong>
                </div>
                <div>
                  <span>Split</span>
                  <strong>{instanceQuery.data.evidence.split ?? '—'}</strong>
                </div>
                <div>
                  <span>Mode</span>
                  <strong>{titleCase(String(instanceQuery.data.evidence.prediction_mode ?? 'unknown'))}</strong>
                </div>
                <div>
                  <span>Ground truth</span>
                  <strong>{String(instanceQuery.data.evidence.ground_truth ?? '—')}</strong>
                </div>
              </div>
              <DataTable table={outputTable} />
              <DataTable table={eventTable} />
              <DataTable table={staticFeatureTable} />
            </div>
          )}
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Trace Inspector</p>
              <h2>System trace rows</h2>
              <p className="panel-copy">
                Read saved stage traces for one framework or LLM system without leaving the run.
              </p>
            </div>
          </div>

          <label className="table-filter wide">
            <span>System</span>
            <select value={activeSystemName} onChange={(event) => setSelectedSystemName(event.target.value)}>
              {systemOptions.map((record) => (
                <option key={record.system_name} value={record.system_name}>
                  {record.system_name} · {titleCase(String(record.framework_type ?? record.system_kind ?? 'unknown'))}
                </option>
              ))}
            </select>
          </label>

          {traceQuery.isLoading ? (
            <LoadingPanel label="Loading system trace" />
          ) : traceQuery.isError ? (
            <EmptyState
              title="Trace payload unavailable"
              description={traceQuery.error instanceof Error ? traceQuery.error.message : 'Unable to load trace rows.'}
            />
          ) : traceTable == null || traceTable.records.length === 0 ? (
            <EmptyState
              title="No trace rows for this system"
              description="This system did not emit saved traces, or traces were disabled for the run."
            />
          ) : (
            <DataTable table={traceTable} />
          )}
        </article>
      </section>
    </div>
  )
}
