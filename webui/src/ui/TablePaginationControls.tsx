import { formatNumber } from '../lib/format'

interface TablePaginationControlsProps {
  offset: number
  limit: number
  totalRows: number
  onPrevious: () => void
  onNext: () => void
}

export function TablePaginationControls({
  offset,
  limit,
  totalRows,
  onPrevious,
  onNext,
}: TablePaginationControlsProps) {
  const pageStart = totalRows === 0 ? 0 : offset + 1
  const pageEnd = totalRows === 0 ? 0 : Math.min(offset + limit, totalRows)
  const canGoBack = offset > 0
  const canGoForward = offset + limit < totalRows

  return (
    <>
      <span className="table-page-summary">Page {Math.floor(offset / limit) + 1}</span>
      <span className="table-page-summary">
        {pageStart}-{pageEnd} of {formatNumber(totalRows)}
      </span>
      <button type="button" className="button-link" onClick={onPrevious} disabled={!canGoBack}>
        Previous
      </button>
      <button type="button" className="button-link" onClick={onNext} disabled={!canGoForward}>
        Next
      </button>
    </>
  )
}
