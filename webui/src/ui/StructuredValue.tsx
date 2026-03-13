import { stringifyValue, titleCase } from '../lib/format'

interface StructuredValueProps {
  value: unknown
  maxItems?: number
  depth?: number
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null && !Array.isArray(value)
}

function isPrimitiveArray(value: unknown[]): boolean {
  return value.every((item) => item == null || ['string', 'number', 'boolean'].includes(typeof item))
}

export function StructuredValue({ value, maxItems = 4, depth = 0 }: StructuredValueProps) {
  if (value == null || value === '') {
    return <span className="structured-value structured-empty">—</span>
  }

  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') {
    return <span className="structured-value structured-text">{stringifyValue(value)}</span>
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="structured-value structured-empty">—</span>
    }

    const items = value.slice(0, maxItems)
    const overflowCount = value.length - items.length

    if (isPrimitiveArray(value)) {
      return (
        <span className="structured-value structured-items">
          {items.map((item, index) => (
            <span key={`${String(item)}-${index}`} className="structured-pill">
              {stringifyValue(item)}
            </span>
          ))}
          {overflowCount > 0 ? <span className="structured-more">+{overflowCount} more</span> : null}
        </span>
      )
    }

    if (depth >= 2) {
      return <span className="structured-value structured-text">{stringifyValue(value)}</span>
    }

    return (
      <span className="structured-value structured-stack">
        {items.map((item, index) => (
          <span key={index} className="structured-block">
            <StructuredValue value={item} maxItems={maxItems} depth={depth + 1} />
          </span>
        ))}
        {overflowCount > 0 ? <span className="structured-more">+{overflowCount} more</span> : null}
      </span>
    )
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value)
    if (entries.length === 0) {
      return <span className="structured-value structured-empty">—</span>
    }

    if (depth >= 2) {
      return <span className="structured-value structured-text">{JSON.stringify(value)}</span>
    }

    const items = entries.slice(0, maxItems)
    const overflowCount = entries.length - items.length

    return (
      <span className="structured-value structured-fields">
        {items.map(([key, itemValue]) => (
          <span key={key} className="structured-field">
            <span className="structured-key">{titleCase(key)}</span>
            <span className="structured-content">
              <StructuredValue value={itemValue} maxItems={maxItems} depth={depth + 1} />
            </span>
          </span>
        ))}
        {overflowCount > 0 ? <span className="structured-more">+{overflowCount} more</span> : null}
      </span>
    )
  }

  return <span className="structured-value structured-text">{stringifyValue(value)}</span>
}
