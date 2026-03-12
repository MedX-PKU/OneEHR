import { Link, useParams } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchRunDetail } from '../lib/api'
import { titleCase } from '../lib/format'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { ModuleCard } from '../ui/ModuleCard'

export function RunOverviewPage() {
  const { runName } = useParams({ from: '/runs/$runName/' })
  const runQuery = useQuery({
    queryKey: ['run', runName],
    queryFn: () => fetchRunDetail(runName),
  })

  if (runQuery.isLoading) {
    return <LoadingPanel label="Loading run overview" />
  }

  if (runQuery.isError || !runQuery.data) {
    return (
      <EmptyState
        title="Run overview unavailable"
        description={runQuery.error instanceof Error ? runQuery.error.message : 'Unable to load the run.'}
      />
    )
  }

  const run = runQuery.data

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Run Overview</p>
          <h1>{run.run_name}</h1>
          <p className="hero-copy">
            Unified control room for modeling outputs, analysis modules, and future case-level investigations.
          </p>
        </div>
        <Link to="/runs/$runName/comparison" params={{ runName }} className="button-link">
          Open comparison
        </Link>
      </section>

      <section className="card-grid kpi-grid">
        <KpiCard key="models" title="Models" value={run.training.models.length} subtitle={run.training.models.join(', ') || '—'} />
        <KpiCard key="splits" title="Splits" value={run.training.splits.length} subtitle={run.training.splits.join(', ') || '—'} />
        <KpiCard key="cases" title="Cases" value={run.cases.case_count} subtitle="Durable case bundles" />
        <KpiCard
          key="modules"
          title="Analysis modules"
          value={run.analysis.modules.length}
          subtitle={`${run.analysis.modules.filter((module) => module.status === 'ok').length} ready`}
        />
      </section>

      <section className="two-column-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Manifest</p>
              <h2>Experiment contract</h2>
            </div>
          </div>
          <div className="detail-grid">
            <div>
              <span>Task kind</span>
              <strong>{titleCase(String(run.manifest.task?.kind ?? 'unknown'))}</strong>
            </div>
            <div>
              <span>Prediction mode</span>
              <strong>{titleCase(String(run.manifest.task?.prediction_mode ?? 'unknown'))}</strong>
            </div>
            <div>
              <span>Split kind</span>
              <strong>{titleCase(String(run.manifest.split?.kind ?? 'unknown'))}</strong>
            </div>
            <div>
              <span>Schema</span>
              <strong>v{run.manifest.schema_version}</strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Connected systems</p>
              <h2>Agent and artifact coverage</h2>
            </div>
          </div>
          <div className="detail-grid">
            <div>
              <span>Agent predictors</span>
              <strong>{run.agent_predict.predictors.length}</strong>
            </div>
            <div>
              <span>Reviewers</span>
              <strong>{run.agent_review.reviewers.length}</strong>
            </div>
            <div>
              <span>Cases index</span>
              <strong>{run.cases.index_path ? 'Present' : 'Missing'}</strong>
            </div>
            <div>
              <span>Analysis index</span>
              <strong>{run.analysis.index_path ? 'Present' : 'Missing'}</strong>
            </div>
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Analysis modules</p>
            <h2>Visual dashboards</h2>
          </div>
        </div>
        {run.analysis.modules.length === 0 ? (
          <EmptyState
            title="No analysis modules found"
            description="Run `oneehr analyze` for this experiment to populate the dashboard."
          />
        ) : (
          <div className="card-grid modules-grid">
            {run.analysis.modules.map((module) => (
              <ModuleCard key={module.name} runName={runName} module={module} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
