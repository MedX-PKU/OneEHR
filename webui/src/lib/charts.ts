import type { DashboardChart } from './types'

const AXIS_COLOR = 'rgba(32, 68, 60, 0.78)'
const GRID_LINE = 'rgba(32, 68, 60, 0.08)'
const PALETTE = ['#12756d', '#d68641', '#c05621', '#4c7cc1', '#8b6fc8', '#d4a017']

function asNumber(value: unknown): number | null {
  const number = Number(value)
  return Number.isNaN(number) ? null : number
}

export function buildChartOption(spec: DashboardChart) {
  const rows = spec.data ?? spec.rows ?? []
  if (spec.kind === 'bar') {
    return {
      color: PALETTE,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 24, right: 12, top: 40, bottom: 36, containLabel: true },
      xAxis: {
        type: 'category',
        axisLine: { lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR },
        data: rows.map((row) => String(row[spec.x ?? 'x'] ?? '—')),
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: GRID_LINE } },
        axisLine: { show: false },
        axisLabel: { color: AXIS_COLOR },
      },
      series: [
        {
          type: 'bar',
          barWidth: '42%',
          itemStyle: { borderRadius: [10, 10, 0, 0] },
          data: rows.map((row) => asNumber(row[spec.y ?? 'y'])),
        },
      ],
    }
  }

  if (spec.kind === 'grouped_bar') {
    const xKey = spec.x ?? 'x'
    const yKey = spec.y ?? 'y'
    const groupKey = spec.group ?? 'group'
    const categories = [...new Set(rows.map((row) => String(row[xKey] ?? '—')))]
    const groups = [...new Set(rows.map((row) => String(row[groupKey] ?? '—')))]
    const lookup = new Map<string, number>()
    rows.forEach((row) => {
      lookup.set(
        `${String(row[groupKey] ?? '—')}::${String(row[xKey] ?? '—')}`,
        asNumber(row[yKey]) ?? 0,
      )
    })
    return {
      color: PALETTE,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: {
        top: 4,
        textStyle: { color: AXIS_COLOR },
      },
      grid: { left: 24, right: 12, top: 52, bottom: 36, containLabel: true },
      xAxis: {
        type: 'category',
        axisLine: { lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR },
        data: categories,
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: GRID_LINE } },
        axisLine: { show: false },
        axisLabel: { color: AXIS_COLOR },
      },
      series: groups.map((group) => ({
        name: group,
        type: 'bar',
        emphasis: { focus: 'series' },
        itemStyle: { borderRadius: [10, 10, 0, 0] },
        data: categories.map((category) => lookup.get(`${group}::${category}`) ?? 0),
      })),
    }
  }

  if (spec.kind === 'line') {
    const xKey = spec.x ?? 'x'
    const yKey = spec.y ?? 'y'
    const groupKey = spec.group ?? 'group'
    const groups = [...new Set(rows.map((row) => String(row[groupKey] ?? 'Series')))]
    return {
      color: PALETTE,
      tooltip: { trigger: 'axis' },
      legend: { top: 4, textStyle: { color: AXIS_COLOR } },
      grid: { left: 24, right: 12, top: 52, bottom: 36, containLabel: true },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        axisLine: { lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR },
        data: [...new Set(rows.map((row) => String(row[xKey] ?? '—')))],
      },
      yAxis: {
        type: 'value',
        splitLine: { lineStyle: { color: GRID_LINE } },
        axisLine: { show: false },
        axisLabel: { color: AXIS_COLOR },
      },
      series: groups.map((group) => ({
        name: group,
        type: 'line',
        smooth: true,
        symbolSize: 8,
        data: rows
          .filter((row) => String(row[groupKey] ?? 'Series') === group)
          .map((row) => [String(row[xKey] ?? '—'), asNumber(row[yKey])]),
      })),
    }
  }

  if (spec.kind === 'heatmap') {
    const xKey = spec.x ?? 'x'
    const yKey = spec.y ?? 'y'
    const valueKey = spec.group ?? 'value'
    const xs = [...new Set(rows.map((row) => String(row[xKey] ?? '—')))]
    const ys = [...new Set(rows.map((row) => String(row[yKey] ?? '—')))]
    return {
      tooltip: { position: 'top' },
      grid: { left: 24, right: 12, top: 36, bottom: 36, containLabel: true },
      xAxis: {
        type: 'category',
        data: xs,
        splitArea: { show: true },
        axisLine: { lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR },
      },
      yAxis: {
        type: 'category',
        data: ys,
        splitArea: { show: true },
        axisLine: { lineStyle: { color: AXIS_COLOR } },
        axisLabel: { color: AXIS_COLOR },
      },
      visualMap: {
        min: 0,
        max: Math.max(...rows.map((row) => asNumber(row[valueKey]) ?? 0), 1),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: { color: AXIS_COLOR },
      },
      series: [
        {
          type: 'heatmap',
          data: rows.map((row) => [
            xs.indexOf(String(row[xKey] ?? '—')),
            ys.indexOf(String(row[yKey] ?? '—')),
            asNumber(row[valueKey]) ?? 0,
          ]),
          label: { show: false },
        },
      ],
    }
  }

  return null
}
