import type { DashboardCard } from '../lib/types'
import { stringifyValue } from '../lib/format'

export function KpiCard({ title, value, subtitle, tone = 'neutral' }: DashboardCard) {
  return (
    <article className={`kpi-card tone-${tone}`}>
      <p className="eyebrow">{title}</p>
      <h3>{stringifyValue(value)}</h3>
      {subtitle ? <p className="kpi-subtitle">{subtitle}</p> : null}
    </article>
  )
}
