import { Link } from '@tanstack/react-router'
import type { AnalysisModuleMeta } from '../lib/types'
import { titleCase } from '../lib/format'
import { StatusBadge } from './StatusBadge'

interface ModuleCardProps {
  runName: string
  module: AnalysisModuleMeta
}

export function ModuleCard({ runName, module }: ModuleCardProps) {
  return (
    <Link
      to="/runs/$runName/analysis/$moduleName"
      params={{ runName, moduleName: module.name }}
      className="module-card"
    >
      <div className="module-card-header">
        <div>
          <p className="eyebrow">Analysis Module</p>
          <h3>{titleCase(module.name)}</h3>
        </div>
        <StatusBadge status={module.status} />
      </div>
      <p>
        Structured summary, tables, and drill-down views for {titleCase(module.name).toLowerCase()}.
      </p>
    </Link>
  )
}
