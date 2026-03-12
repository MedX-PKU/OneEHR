import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchAgents } from '../lib/api'
import type { AgentTaskPayload } from '../lib/types'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { StatusBadge } from '../ui/StatusBadge'
import { DataTable } from '../ui/DataTable'

interface AgentSectionProps {
  title: string
  description: string
  task: AgentTaskPayload
}

function AgentSection({ title, description, task }: AgentSectionProps) {
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
        title="Agent Predict"
        description="Structured prediction outputs from configured agent backends."
        task={payload.predict}
      />
      <AgentSection
        title="Agent Review"
        description="Evidence-grounded review outputs over saved case bundles and predictions."
        task={payload.review}
      />
    </div>
  )
}
