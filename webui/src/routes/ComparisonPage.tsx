import { useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchComparison } from '../lib/api'
import { ChartPanel } from '../ui/ChartPanel'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { DataTable } from '../ui/DataTable'

export function ComparisonPage() {
  const { runName } = useParams({ from: '/runs/$runName/comparison' })
  const comparisonQuery = useQuery({
    queryKey: ['comparison', runName],
    queryFn: () => fetchComparison(runName),
  })

  if (comparisonQuery.isLoading) {
    return <LoadingPanel label="Loading comparison dashboard" />
  }

  if (comparisonQuery.isError || !comparisonQuery.data) {
    return (
      <EmptyState
        title="Comparison not available"
        description={
          comparisonQuery.error instanceof Error
            ? comparisonQuery.error.message
            : 'This run does not have compare-run outputs yet.'
        }
      />
    )
  }

  const comparison = comparisonQuery.data

  if (comparison.status !== 'ok') {
    return (
      <EmptyState
        title="Comparison not available"
        description="Generate comparison artifacts with `oneehr analyze --compare-run` for this run."
      />
    )
  }

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Compare Run</p>
          <h1>{runName}</h1>
          <p className="hero-copy">
            Delta views across training metrics and agent prediction summaries when comparison artifacts exist.
          </p>
        </div>
      </section>

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
  )
}
