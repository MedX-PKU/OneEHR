import type { DashboardCard } from '../lib/types'

export function KpiCard({ title, value, subtitle, tone = 'neutral' }: DashboardCard) {
  return (
    <article className={`kpi-card tone-${tone}`}>
      <p className="eyebrow">{title}</p>
      <h3>{value ?? '—'}</h3>
      {subtitle ? <p className="kpi-subtitle">{subtitle}</p> : null}
    </article>
  )
}
