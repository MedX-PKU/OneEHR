import { useDeferredValue, useState } from 'react'
import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchAgentTaskFailures, fetchAgentTaskRecords, fetchAgents } from '../lib/api'
import type { AgentRowsPayload, AgentTaskPayload } from '../lib/types'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'
import { DataTable } from '../ui/DataTable'

interface AgentSectionProps {
  runName: string
  taskName: string
  title: string
  description: string
  task: AgentTaskPayload
}

function AgentRowsSection({
  title,
  emptyTitle,
  emptyDescription,
  payload,
  isLoading,
  isError,
  error,
}: {
  title: string
  emptyTitle: string
  emptyDescription: string
  payload: AgentRowsPayload | undefined
  isLoading: boolean
  isError: boolean
  error: Error | null
}) {
  if (isLoading) {
    return <LoadingPanel label={`Loading ${title.toLowerCase()}`} />
  }

  if (isError) {
    return (
      <EmptyState
        title={`${title} unavailable`}
        description={error?.message ?? `Unable to load ${title.toLowerCase()}.`}
      />
    )
  }

  if (!payload?.table || payload.total_rows === 0) {
    return <EmptyState title={emptyTitle} description={emptyDescription} />
  }

  return <DataTable table={payload.table} />
}

function AgentSection({ runName, taskName, title, description, task }: AgentSectionProps) {
  const [actor, setActor] = useState('')
  const [split, setSplit] = useState('')
  const [search, setSearch] = useState('')
  const [parsedOk, setParsedOk] = useState('all')
  const deferredSearch = useDeferredValue(search)

  const recordsQuery = useQuery({
    queryKey: ['agent-records', runName, taskName, actor, split, parsedOk, deferredSearch],
    queryFn: () =>
      fetchAgentTaskRecords(runName, taskName, {
        actor: actor || null,
        split: split || null,
        parsedOk: parsedOk === 'all' ? null : parsedOk === 'true',
        search: deferredSearch || null,
        limit: 150,
      }),
    enabled: task.status === 'ok' && task.detail_available,
  })

  const failuresQuery = useQuery({
    queryKey: ['agent-failures', runName, taskName, actor, split, deferredSearch],
    queryFn: () =>
      fetchAgentTaskFailures(runName, taskName, {
        actor: actor || null,
        split: split || null,
        search: deferredSearch || null,
        limit: 150,
      }),
    enabled: task.status === 'ok',
  })

  return (
    <section className="page-stack">
      <article className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Agent Workspace</p>
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
                <h2>Detailed agent rows</h2>
                <p className="panel-copy">
                  Filter saved agent outputs by backend, split, parse status, or free-text search. The table previews
                  the first 150 matching rows.
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
            <AgentRowsSection
              title={`${title} rows`}
              emptyTitle="No matching detailed rows"
              emptyDescription="No saved rows matched the current filters."
              payload={recordsQuery.data}
              isLoading={recordsQuery.isLoading}
              isError={recordsQuery.isError}
              error={recordsQuery.error instanceof Error ? recordsQuery.error : null}
            />
          )}

          <AgentRowsSection
            title={`${title} failures`}
            emptyTitle="No failure rows"
            emptyDescription="No agent failures matched the current filters."
            payload={failuresQuery.data}
            isLoading={failuresQuery.isLoading}
            isError={failuresQuery.isError}
            error={failuresQuery.error instanceof Error ? failuresQuery.error : null}
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
    return <LoadingPanel label="Loading agent workspace" />
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
          <h1>Agent operations workspace</h1>
          <p className="hero-copy">
            Inspect agent prediction and review outputs using the same artifact-backed workspace as the analysis UI.
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
