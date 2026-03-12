import { useEffect, useRef } from 'react'
import { BarChart, HeatmapChart, LineChart } from 'echarts/charts'
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { init, use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { buildChartOption } from '../lib/charts'
import { formatNumber, stringifyValue } from '../lib/format'
import type { DashboardChart } from '../lib/types'

use([
  BarChart,
  HeatmapChart,
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

interface ChartPanelProps {
  chart: DashboardChart
}

export function ChartPanel({ chart }: ChartPanelProps) {
  if (chart.kind === 'metric_card') {
    const firstRow = chart.data?.[0] ?? {}
    const firstValue = chart.y ? firstRow[chart.y] : Object.values(firstRow)[0]
    return (
      <article className="chart-card metric-card">
        <p className="eyebrow">{chart.title}</p>
        <h3>{formatNumber(firstValue)}</h3>
        {chart.description ? <p>{chart.description}</p> : null}
      </article>
    )
  }

  if (chart.kind === 'ranked_table') {
    const rows = chart.data ?? []
    return (
      <article className="chart-card ranked-chart">
        <div className="chart-header">
          <div>
            <p className="eyebrow">Ranked view</p>
            <h3>{chart.title}</h3>
          </div>
          {chart.description ? <p>{chart.description}</p> : null}
        </div>
        <div className="ranked-list">
          {rows.map((row, index) => (
            <div key={`${chart.key}-${index}`} className="ranked-row">
              <span>{index + 1}</span>
              <strong>{stringifyValue(row[chart.x ?? 'label'])}</strong>
              <em>{stringifyValue(row[chart.y ?? 'value'])}</em>
            </div>
          ))}
        </div>
      </article>
    )
  }

  return <EChartsPanel chart={chart} />
}

function EChartsPanel({ chart }: ChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!containerRef.current) {
      return
    }

    const instance = init(containerRef.current, undefined, {
      renderer: 'canvas',
    })
    const option = buildChartOption(chart)
    if (option) {
      instance.setOption(option)
    }

    const handleResize = () => instance.resize()
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      instance.dispose()
    }
  }, [chart])

  return (
    <article className="chart-card">
      <div className="chart-header">
        <div>
          <p className="eyebrow">Visualization</p>
          <h3>{chart.title}</h3>
        </div>
        {chart.description ? <p>{chart.description}</p> : null}
      </div>
      <div ref={containerRef} className="chart-surface" />
    </article>
  )
}
