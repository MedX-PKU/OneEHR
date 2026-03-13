export function formatNumber(value: unknown): string {
  if (value == null || value === '') {
    return '—'
  }
  const number = Number(value)
  if (Number.isNaN(number)) {
    return String(value)
  }
  if (Math.abs(number) >= 1000) {
    return new Intl.NumberFormat('en-US', {
      maximumFractionDigits: 1,
      notation: 'compact',
    }).format(number)
  }
  if (Math.abs(number) >= 1) {
    return new Intl.NumberFormat('en-US', {
      maximumFractionDigits: 3,
    }).format(number)
  }
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 4,
  }).format(number)
}

export function formatDate(value: number | string | undefined): string {
  if (!value) {
    return '—'
  }
  const date = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  if (Number.isNaN(date.getTime())) {
    return '—'
  }
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

export function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

export function formatNameList(values: Array<string | null | undefined> | null | undefined): string {
  const items = (values ?? []).filter((value): value is string => typeof value === 'string' && value.length > 0)
  return items.length > 0 ? items.join(', ') : '—'
}

export function formatMetricName(value: unknown): string {
  if (value == null || value === '') {
    return '—'
  }
  const metric = String(value)
  return metric.length <= 6 ? metric.toUpperCase() : titleCase(metric)
}

export function formatIdentifierDisplay(value: unknown): string {
  if (value == null || value === '') {
    return '—'
  }
  return String(value).replace(/([/_:-])/g, '$1\u200b')
}

export function formatTestingBestModel(
  testing:
    | {
        best_model?: {
          model?: unknown
          metric?: unknown
          value?: unknown
        } | null
      }
    | null
    | undefined,
): string {
  const bestModel = testing?.best_model
  if (!bestModel || bestModel.model == null) {
    return 'No test summary'
  }
  const metricValue = bestModel.value == null ? '—' : formatNumber(bestModel.value)
  return `${String(bestModel.model)} ${formatMetricName(bestModel.metric)} ${metricValue}`
}

export function stringifyValue(value: unknown): string {
  if (value == null || value === '') {
    return '—'
  }
  if (typeof value === 'number') {
    return formatNumber(value)
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }
  if (Array.isArray(value)) {
    return value.join(', ')
  }
  if (typeof value === 'object') {
    return JSON.stringify(value)
  }
  return String(value)
}
