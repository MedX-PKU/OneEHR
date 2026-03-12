interface LoadingPanelProps {
  label: string
}

export function LoadingPanel({ label }: LoadingPanelProps) {
  return (
    <section className="loading-panel">
      <span className="loading-pulse" />
      <p>{label}</p>
    </section>
  )
}
