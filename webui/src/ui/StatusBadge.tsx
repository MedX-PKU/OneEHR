interface StatusBadgeProps {
  status?: string | null
}

const STATUS_LABELS: Record<string, string> = {
  ok: 'OK',
  skipped: 'Skipped',
  error: 'Error',
  warning: 'Warning',
  neutral: 'Unknown',
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const raw = (status ?? 'neutral').toLowerCase()
  const tone =
    raw === 'ok' || raw === 'ready'
      ? 'ok'
      : raw === 'skipped' || raw === 'pending'
        ? 'skipped'
        : raw === 'missing' || raw === 'warning'
          ? 'warning'
          : raw === 'error'
            ? 'error'
            : 'neutral'
  return (
    <span className={`status-badge tone-${tone}`}>
      {STATUS_LABELS[tone] ?? status ?? 'Unknown'}
    </span>
  )
}
