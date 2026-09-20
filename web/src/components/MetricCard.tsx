interface Props {
  label: string
  value: string | number
  unit?: string
  tone?: 'neutral' | 'fixed' | 'dqn'
}

export function MetricCard({ label, value, unit, tone = 'neutral' }: Props) {
  return (
    <div className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {unit ? <small>{unit}</small> : null}
    </div>
  )
}
