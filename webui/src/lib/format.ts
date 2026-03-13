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
