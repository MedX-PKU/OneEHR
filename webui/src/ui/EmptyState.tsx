interface EmptyStateProps {
  title: string
  description: string
}

export function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <section className="empty-state">
      <p className="eyebrow">No content</p>
      <h2>{title}</h2>
      <p>{description}</p>
    </section>
  )
}
