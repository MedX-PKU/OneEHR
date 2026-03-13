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
    </Link>
  )
}
