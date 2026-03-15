import { Link, useParams } from '@tanstack/react-router'
import { useQueries, useQuery } from '@tanstack/react-query'
import { fetchModuleDashboard, fetchRunConsole } from '../lib/api'
import {
  formatIdentifierDisplay,
  formatMetricName,
  formatNameList,
  formatNumber,
  formatTestingBestModel,
  titleCase,
} from '../lib/format'
import { sortModulesByPriority } from '../lib/modules'
import { ChartPanel } from '../ui/ChartPanel'
import { DataTable } from '../ui/DataTable'
import { EmptyState } from '../ui/EmptyState'
import { KpiCard } from '../ui/KpiCard'
import { LoadingPanel } from '../ui/LoadingPanel'
import { ModuleCard } from '../ui/ModuleCard'

export function RunOverviewPage() {
  const { runName } = useParams({ from: '/runs/$runName/' })
  const runConsoleQuery = useQuery({
    queryKey: ['run-console', runName],
    queryFn: () => fetchRunConsole(runName),
  })
  const orderedModules = sortModulesByPriority(runConsoleQuery.data?.analysis.modules ?? [])
  const readyModules = orderedModules.filter((module) => module.status === 'ok').slice(0, 3)
  const skippedModules = orderedModules.filter((module) => module.status === 'skipped')
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

  if (runConsoleQuery.isLoading) {
    return <LoadingPanel label="Loading run overview" />
  }

  if (runConsoleQuery.isError || !runConsoleQuery.data) {
    return (
      <EmptyState
        title="Run overview unavailable"
        description={runConsoleQuery.error instanceof Error ? runConsoleQuery.error.message : 'Unable to load the run.'}
      />
    )
  }

  const runConsole = runConsoleQuery.data
  const run = runConsole.run
  const testAuditModule = orderedModules.find((module) => module.name === 'test_audit' && module.status === 'ok') ?? null

  return (
    <div className="page-stack">
      <section className="page-header">
        <div>
          <p className="eyebrow">Run Overview</p>
          <h1 className="entity-title identifier-text">{formatIdentifierDisplay(run.run_name)}</h1>
          <p className="hero-copy">
            Unified control room for training outputs, analysis modules, and reproducible evaluation artifacts.
          </p>
        </div>
        <div className="header-actions">
          {run.eval.report_summary_path ? (
            <Link to="/runs/$runName/eval" params={{ runName }} className="button-link">
              Open evaluation
            </Link>
          ) : null}
          {testAuditModule ? (
            <Link to="/runs/$runName/analysis/$moduleName" params={{ runName, moduleName: 'test_audit' }} className="button-link">
              Open Test Audit
            </Link>
          ) : null}
          <Link to="/runs/$runName/comparison" params={{ runName }} className="button-link">
            Open comparison
          </Link>
        </div>
      </section>

      <section className="card-grid kpi-grid">
        <KpiCard key="models" title="Models" value={run.training.models.length} subtitle={run.training.models.join(', ') || '—'} />
        <KpiCard key="splits" title="Splits" value={run.training.splits.length} subtitle={run.training.splits.join(', ') || '—'} />
        <KpiCard
          key="testing"
          title="External test"
          value={run.testing.record_count}
          subtitle={formatTestingBestModel(run.testing)}
        />
        <KpiCard
          key="eval"
          title="Eval systems"
          value={run.eval.system_count}
          subtitle={`${run.eval.instance_count} frozen instances`}
        />
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
              Ready module dashboards are previewed here so the first run screen already contains actual structured output.
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

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">External Testing</p>
            <h2>Held-out evaluation snapshot</h2>
            <p className="panel-copy">
              Run-level testing metadata comes directly from saved `test_summary.json`, even when compare-run artifacts are absent.
            </p>
          </div>
          {testAuditModule ? (
            <Link to="/runs/$runName/analysis/$moduleName" params={{ runName, moduleName: 'test_audit' }} className="button-link">
              Inspect test audit
            </Link>
          ) : null}
        </div>

        {run.testing.record_count === 0 ? (
          <EmptyState
            title="No external test summary yet"
            description="Run `oneehr test` for this experiment to populate held-out metrics and unlock the test audit module."
          />
        ) : (
          <div className="detail-grid">
            <div>
              <span>Primary metric</span>
              <strong>{formatMetricName(run.testing.primary_metric)}</strong>
            </div>
            <div>
              <span>Best model</span>
              <strong>{run.testing.best_model?.model ?? '—'}</strong>
            </div>
            <div>
              <span>Best score</span>
              <strong>{run.testing.best_score == null ? '—' : formatNumber(run.testing.best_score)}</strong>
            </div>
            <div>
              <span>Test models</span>
              <strong>{formatNameList(run.testing.models)}</strong>
            </div>
            <div>
              <span>Test splits</span>
              <strong>{formatNameList(run.testing.splits)}</strong>
            </div>
            <div>
              <span>Summary artifact</span>
              <strong>{run.testing.summary_path ?? 'Missing'}</strong>
            </div>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Unified Eval</p>
            <h2>Frozen cross-system comparison</h2>
            <p className="panel-copy">
              The eval pipeline keeps model and framework comparisons on one reproducible instance index.
            </p>
          </div>
          {run.eval.report_summary_path ? (
            <Link to="/runs/$runName/eval" params={{ runName }} className="button-link">
              Inspect evaluation
            </Link>
          ) : null}
        </div>

        {run.eval.report_summary_path == null ? (
          <EmptyState
            title="No unified evaluation report yet"
            description="Run `oneehr eval build`, `oneehr eval run`, and `oneehr eval report` to compare trained models, LLM systems, and multi-agent frameworks."
          />
        ) : (
          <div className="detail-grid">
            <div>
              <span>Eval instances</span>
              <strong>{run.eval.instance_count}</strong>
            </div>
            <div>
              <span>Systems</span>
              <strong>{run.eval.system_count}</strong>
            </div>
            <div>
              <span>Leaderboard rows</span>
              <strong>{run.eval.leaderboard_rows}</strong>
            </div>
            <div>
              <span>Primary metric</span>
              <strong>{formatMetricName(run.eval.primary_metric)}</strong>
            </div>
            <div>
              <span>Index artifact</span>
              <strong>{run.eval.index_path ?? 'Missing'}</strong>
            </div>
            <div>
              <span>Report artifact</span>
              <strong>{run.eval.report_summary_path ?? 'Missing'}</strong>
            </div>
          </div>
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
            <div>
              <span>Held-out test</span>
              <strong>{run.testing.summary_path ? 'Present' : 'Missing'}</strong>
            </div>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Artifacts</p>
              <h2>Run surface coverage</h2>
            </div>
          </div>
          <div className="detail-grid">
            <div>
              <span>Analysis index</span>
              <strong>{run.analysis.index_path ? 'Present' : 'Missing'}</strong>
            </div>
            <div>
              <span>Eval index</span>
              <strong>{run.eval.index_path ? 'Present' : 'Missing'}</strong>
            </div>
            <div>
              <span>Eval report</span>
              <strong>{run.eval.report_summary_path ? 'Present' : 'Missing'}</strong>
            </div>
            <div>
              <span>Test summary</span>
              <strong>{run.testing.summary_path ? 'Present' : 'Missing'}</strong>
            </div>
            <div>
              <span>Best test model</span>
              <strong>{run.testing.best_model?.model ?? '—'}</strong>
            </div>
          </div>
        </article>
      </section>

      {(skippedModules.length > 0 || run.training.record_count === 0 || run.testing.record_count === 0 || run.eval.report_summary_path == null) && (
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
              <span>Eval systems</span>
              <strong>{run.eval.system_count}</strong>
            </div>
            <div>
              <span>Test rows</span>
              <strong>{run.testing.record_count}</strong>
            </div>
            <div>
              <span>Recommended next step</span>
              <strong>
                {run.training.record_count === 0
                  ? 'Train the run'
                  : run.testing.record_count === 0
                    ? 'Run held-out test'
                    : run.eval.report_summary_path == null
                      ? 'Run unified eval'
                      : 'Re-run analysis'}
              </strong>
            </div>
          </div>
          <p className="panel-copy">
            Model-backed modules stay skipped until the run has training outputs. Cross-system comparison stays empty until the `oneehr eval` pipeline has produced the frozen evaluation index, predictions, and report.
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
            {orderedModules.map((module) => (
              <ModuleCard key={module.name} runName={runName} module={module} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
