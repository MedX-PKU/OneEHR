import { Link, useParams } from '@tanstack/react-router'
import { useQueries, useQuery } from '@tanstack/react-query'
import { fetchModuleDashboard, fetchRunWorkspace } from '../lib/api'
import { titleCase } from '../lib/format'
import { ChartPanel } from '../ui/ChartPanel'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { ModuleCard } from '../ui/ModuleCard'

export function RunOverviewPage() {
  const { runName } = useParams({ from: '/runs/$runName/' })
  const workspaceQuery = useQuery({
    queryKey: ['run-workspace', runName],
    queryFn: () => fetchRunWorkspace(runName),
  })
  const readyModules = (workspaceQuery.data?.analysis.modules ?? []).filter((module) => module.status === 'ok').slice(0, 3)
  const skippedModules = (workspaceQuery.data?.analysis.modules ?? []).filter((module) => module.status === 'skipped')
  const moduleDashboardQueries = useQueries({
    queries: readyModules.map((module) => ({
      queryKey: ['module-dashboard-overview', runName, module.name],
      queryFn: () => fetchModuleDashboard(runName, module.name),
    })),
  })
  const previewCharts = moduleDashboardQueries.flatMap((query, index) => {
    if (!query.data) {
      return []
    }
    return query.data.charts.slice(0, 2).map((chart) => ({
      ...chart,
      key: `${readyModules[index]?.name ?? 'module'}-${chart.key}`,
      title: `${query.data?.title}: ${chart.title}`,
    }))
  })
  const previewTables = moduleDashboardQueries.flatMap((query) => (query.data?.tables ?? []).slice(0, 1))
  const previewIsLoading = moduleDashboardQueries.some((query) => query.isLoading)
  const previewError = moduleDashboardQueries.find((query) => query.isError)?.error

  if (workspaceQuery.isLoading) {
    return <LoadingPanel label="Loading run overview" />
  }

  if (workspaceQuery.isError || !workspaceQuery.data) {
    return (
      <EmptyState
        title="Run overview unavailable"
        description={workspaceQuery.error instanceof Error ? workspaceQuery.error.message : 'Unable to load the run.'}
      />
    )
  }

  const workspace = workspaceQuery.data
  const run = workspace.run

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

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Visual Preview</p>
            <h2>Charts visible from the run overview</h2>
            <p className="panel-copy">
              Ready module dashboards are previewed here so the first run screen already contains actual visual output.
            </p>
          </div>
        </div>

        {readyModules.length === 0 ? (
          <EmptyState
            title="No ready dashboards yet"
            description="Run `oneehr train` and then `oneehr analyze` to unlock model-backed visual modules."
          />
        ) : previewIsLoading ? (
          <LoadingPanel label="Loading module previews" />
        ) : previewError ? (
          <EmptyState
            title="Module previews unavailable"
            description={previewError instanceof Error ? previewError.message : 'Unable to load module previews.'}
          />
        ) : previewCharts.length > 0 ? (
          <div className="chart-grid">
            {previewCharts.map((chart) => (
              <ChartPanel key={chart.key} chart={chart} />
            ))}
          </div>
        ) : previewTables.length > 0 ? (
          <div className="page-stack">
            {previewTables.map((table) => (
              <DataTable key={table.key} table={table} />
            ))}
          </div>
        ) : (
          <EmptyState
            title="No preview artifacts available"
            description="The ready modules did not emit charts or preview tables for this run."
          />
        )}
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

      {(skippedModules.length > 0 || run.training.record_count === 0 || run.cases.case_count === 0) && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Readiness</p>
              <h2>What is still missing</h2>
            </div>
          </div>
          <div className="detail-grid">
            <div>
              <span>Skipped modules</span>
              <strong>{skippedModules.length}</strong>
            </div>
            <div>
              <span>Training summary rows</span>
              <strong>{run.training.record_count}</strong>
            </div>
            <div>
              <span>Case bundles</span>
              <strong>{run.cases.case_count}</strong>
            </div>
            <div>
              <span>Recommended next step</span>
              <strong>
                {run.training.record_count === 0
                  ? 'Train the run'
                  : run.cases.case_count === 0
                    ? 'Build cases'
                    : 'Re-run analysis'}
              </strong>
            </div>
          </div>
          <p className="panel-copy">
            Model-backed modules stay skipped until the run has training outputs. Case drilldowns stay sparse until
            `oneehr cases build` has materialized durable bundles.
          </p>
        </section>
      )}

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
            {workspace.analysis.modules.map((module) => (
              <ModuleCard key={module.name} runName={runName} module={module} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
