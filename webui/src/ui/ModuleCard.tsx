import { Link } from '@tanstack/react-router'
import type { AnalysisModuleMeta } from '../lib/types'
import { titleCase } from '../lib/format'
import { StatusBadge } from './StatusBadge'

interface ModuleCardProps {
  runName: string
  module: AnalysisModuleMeta
}

export function ModuleCard({ runName, module }: ModuleCardProps) {
  const title = module.title ?? titleCase(module.name)
  const description =
    module.description ?? `Structured summary, tables, and drill-down views for ${title.toLowerCase()}.`
  const metadata = [
    `${module.plot_names?.length ?? 0} chart${module.plot_names?.length === 1 ? '' : 's'}`,
    `${module.table_names?.length ?? 0} table${module.table_names?.length === 1 ? '' : 's'}`,
    `${module.case_names?.length ?? 0} case artifact${module.case_names?.length === 1 ? '' : 's'}`,
  ]

  return (
    <Link
      to="/runs/$runName/analysis/$moduleName"
      params={{ runName, moduleName: module.name }}
      className="module-card"
    >
      <div className="module-card-header">
        <div>
          <p className="eyebrow">Analysis Module</p>
          <h3>{title}</h3>
        </div>
        <StatusBadge status={module.status} />
      </div>
      <p>{description}</p>
      <div className="capability-row">
        {metadata.map((item) => (
          <span key={`${module.name}-${item}`}>{item}</span>
        ))}
      </div>
    </Link>
  )
}
