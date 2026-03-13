import { useDeferredValue, useEffect, useState } from 'react'
import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchAgentTaskFailures, fetchAgentTaskRecords, fetchAgents } from '../lib/api'
import { formatNumber } from '../lib/format'
import type { AgentRowsPayload, AgentTaskPayload } from '../lib/types'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'
import { TablePaginationControls } from '../ui/TablePaginationControls'

const DEFAULT_AGENT_PAGE_SIZE = 25

interface AgentSectionProps {
  runName: string
  taskName: string
  title: string
  description: string
  task: AgentTaskPayload
}

interface AgentRowsBrowserProps {
  title: string
  emptyTitle: string
  emptyDescription: string
  loadingLabel: string
  queryKeyPrefix: readonly unknown[]
  enabled: boolean
  resetKey: string
  fetchRows: (options: { limit: number; offset: number }) => Promise<AgentRowsPayload>
}

function describeAgentWindow(payload: AgentRowsPayload): string {
  const pageStart = payload.total_rows === 0 ? 0 : payload.offset + 1
  const pageEnd = payload.total_rows === 0 ? 0 : payload.offset + payload.row_count
  return `Showing ${pageStart}-${pageEnd} of ${formatNumber(payload.total_rows)} rows`
}

function AgentRowsBrowser({
  title,
  emptyTitle,
  emptyDescription,
  loadingLabel,
  queryKeyPrefix,
  enabled,
  resetKey,
  fetchRows,
}: AgentRowsBrowserProps) {
  const [limit, setLimit] = useState(DEFAULT_AGENT_PAGE_SIZE)
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    setOffset(0)
  }, [resetKey])

  const rowsQuery = useQuery({
    queryKey: [...queryKeyPrefix, limit, offset],
    queryFn: () => fetchRows({ limit, offset }),
    enabled,
  })

  if (rowsQuery.isLoading) {
    return <LoadingPanel label={loadingLabel} />
  }

  if (rowsQuery.isError) {
    return (
      <EmptyState
        title={`${title} unavailable`}
        description={rowsQuery.error instanceof Error ? rowsQuery.error.message : `Unable to load ${title.toLowerCase()}.`}
      />
    )
  }

  const payload = rowsQuery.data
  if (!payload?.table || payload.total_rows === 0) {
    return <EmptyState title={emptyTitle} description={emptyDescription} />
  }

  const table = {
    ...payload.table,
    description: describeAgentWindow(payload),
  }

  return (
    <DataTable
      table={table}
      searchable={false}
      emptyMessage={emptyDescription}
      toolbarActions={
        <div className="table-toolbar-actions">
          <label className="table-filter">
            <span>Page size</span>
            <select
              value={String(limit)}
              onChange={(event) => {
                setLimit(Number(event.target.value))
                setOffset(0)
              }}
            >
              <option value="25">25 rows</option>
              <option value="50">50 rows</option>
              <option value="100">100 rows</option>
            </select>
          </label>
          <TablePaginationControls
            offset={payload.offset}
            limit={payload.limit}
            totalRows={payload.total_rows}
            onPrevious={() => setOffset(Math.max(0, payload.offset - payload.limit))}
            onNext={() => setOffset(payload.offset + payload.limit)}
          />
        </div>
      }
    />
  )
}

function AgentSection({ runName, taskName, title, description, task }: AgentSectionProps) {
  const [actor, setActor] = useState('')
  const [split, setSplit] = useState('')
  const [search, setSearch] = useState('')
  const [parsedOk, setParsedOk] = useState('all')
  const deferredSearch = useDeferredValue(search)
  const parsedOkValue = parsedOk === 'all' ? null : parsedOk === 'true'
  const sharedResetKey = `${actor}|${split}|${parsedOk}|${deferredSearch}`

  return (
    <section className="page-stack">
      <article className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Agent Console</p>
            <h2>{title}</h2>
            <p className="panel-copy">{description}</p>
          </div>
          <StatusBadge status={task.status} />
        </div>
      </article>

      {task.status !== 'ok' ? (
        <EmptyState
          title={`${title} unavailable`}
          description="This run does not have saved artifacts for this agent workflow."
        />
      ) : (
        <>
          <section className="card-grid kpi-grid">
            {task.cards.map((card) => (
              <KpiCard
                key={card.key}
                title={card.title}
                value={card.value}
                subtitle={card.subtitle}
                tone={card.tone}
              />
            ))}
          </section>

          {task.table ? <DataTable table={task.table} /> : null}

          <article className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Drilldown</p>
                <h2>Browse detailed agent outputs</h2>
                <p className="panel-copy">
                  Filter saved agent rows by backend, split, parse status, or free-text search. Detailed rows and
                  failures page independently from the server.
                </p>
              </div>
            </div>

            <div className="filter-grid">
              <label className="table-filter">
                <span>Actor</span>
                <select value={actor} onChange={(event) => setActor(event.target.value)}>
                  <option value="">All actors</option>
                  {task.actors.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>

              <label className="table-filter">
                <span>Split</span>
                <select value={split} onChange={(event) => setSplit(event.target.value)}>
                  <option value="">All splits</option>
                  {task.splits.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </label>

              <label className="table-filter">
                <span>Parse status</span>
                <select value={parsedOk} onChange={(event) => setParsedOk(event.target.value)}>
                  <option value="all">All rows</option>
                  <option value="true">Parsed OK</option>
                  <option value="false">Parse failed</option>
                </select>
              </label>

              <label className="table-filter wide">
                <span>Search</span>
                <input
                  type="search"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  placeholder="Search patient, case, predictor, or message"
                />
              </label>
            </div>
          </article>

          {!task.detail_available ? (
            <EmptyState
              title="Detailed rows unavailable"
              description="This run only saved summary metrics for this agent workflow."
            />
          ) : (
            <AgentRowsBrowser
              title={`${title} rows`}
              emptyTitle="No matching detailed rows"
              emptyDescription="No saved rows matched the current filters."
              loadingLabel="Loading detailed agent rows"
              queryKeyPrefix={['agent-records', runName, taskName, actor, split, parsedOk, deferredSearch]}
              enabled={task.detail_available}
              resetKey={sharedResetKey}
              fetchRows={({ limit, offset }) =>
                fetchAgentTaskRecords(runName, taskName, {
                  actor: actor || null,
                  split: split || null,
                  parsedOk: parsedOkValue,
                  search: deferredSearch || null,
                  limit,
                  offset,
                })
              }
            />
          )}

          <AgentRowsBrowser
            title={`${title} failures`}
            emptyTitle="No failure rows"
            emptyDescription="No agent failures matched the current filters."
            loadingLabel="Loading agent failures"
            queryKeyPrefix={['agent-failures', runName, taskName, actor, split, deferredSearch]}
            enabled={task.status === 'ok'}
            resetKey={`${actor}|${split}|${deferredSearch}`}
            fetchRows={({ limit, offset }) =>
              fetchAgentTaskFailures(runName, taskName, {
                actor: actor || null,
                split: split || null,
                search: deferredSearch || null,
                limit,
                offset,
              })
            }
          />
        </>
      )}
    </section>
  )
}

export function AgentsPage() {
  const { runName } = useParams({ from: '/runs/$runName/agents' })
  const agentsQuery = useQuery({
    queryKey: ['agents', runName],
    queryFn: () => fetchAgents(runName),
  })

  if (agentsQuery.isLoading) {
    return <LoadingPanel label="Loading agents" />
  }

  if (agentsQuery.isError || !agentsQuery.data) {
    return (
      <EmptyState
        title="Agents unavailable"
        description={agentsQuery.error instanceof Error ? agentsQuery.error.message : 'Unable to load agent outputs.'}
      />
    )
  }

  const payload = agentsQuery.data
  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Agents</p>
          <h1>Agent operations console</h1>
          <p className="hero-copy">
            Inspect saved agent prediction and review outputs alongside analysis artifacts and case bundles.
          </p>
        </div>
      </section>

      <AgentSection
        runName={runName}
        taskName="predict"
        title="Agent Predict"
        description="Structured prediction outputs from configured agent backends."
        task={payload.predict}
      />
      <AgentSection
        runName={runName}
        taskName="review"
        title="Agent Review"
        description="Evidence-grounded review outputs over saved case bundles and predictions."
        task={payload.review}
      />
    </div>
  )
}
