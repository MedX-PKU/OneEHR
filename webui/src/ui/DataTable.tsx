import { useDeferredValue, useMemo, useState } from 'react'
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  useReactTable,
} from '@tanstack/react-table'
import type { DashboardTable } from '../lib/types'
import { stringifyValue, titleCase } from '../lib/format'

interface DataTableProps {
  table: DashboardTable
}

export function DataTable({ table }: DataTableProps) {
  const [filter, setFilter] = useState('')
  const deferredFilter = useDeferredValue(filter)

  const columns = useMemo(
    () =>
      table.columns.map((column) => ({
        accessorKey: column,
        header: titleCase(column),
        cell: ({ getValue }: { getValue: () => unknown }) => stringifyValue(getValue()),
      })),
    [table.columns],
  )

  const data = useMemo(() => table.records, [table.records])

  const instance = useReactTable({
    data,
    columns,
    state: {
      globalFilter: deferredFilter,
    },
    globalFilterFn: (row, _columnId, filterValue) => {
      const query = String(filterValue).trim().toLowerCase()
      if (!query) {
        return true
      }
      return row
        .getAllCells()
        .some((cell) => stringifyValue(cell.getValue()).toLowerCase().includes(query))
    },
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  })

  return (
    <article className="table-card">
      <div className="table-toolbar">
        <div>
          <h3>{table.title}</h3>
          <p>{table.description ?? `${table.row_count} rows available`}</p>
        </div>
        <label className="table-filter">
          <span>Search</span>
          <input
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Filter rows"
          />
        </label>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            {instance.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id}>
                    {header.isPlaceholder
                      ? null
                      : flexRender(header.column.columnDef.header, header.getContext())}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {instance.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  )
}
