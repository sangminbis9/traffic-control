import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export interface ChartPoint {
  time: number
  fixedRemaining: number
  dqnRemaining: number
  fixedQueue: number
  dqnQueue: number
  fixedWait: number
  dqnWait: number
  fixedThroughput: number
  dqnThroughput: number
  fixedPhase: number
  dqnPhase: number
}

interface Props { data: ChartPoint[] }

function Chart({ data, title, fixedKey, dqnKey, unit }: { data: ChartPoint[]; title: string; fixedKey: keyof ChartPoint; dqnKey: keyof ChartPoint; unit: string }) {
  return (
    <div className="chart-panel">
      <header><strong>{title}</strong><span>{unit}</span></header>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={data} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
          <CartesianGrid stroke="#d9e2ea" vertical={false} />
          <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 11 }} />
          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
          <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '6px', color: '#0f172a' }} />
          <Legend />
          <Line dataKey={fixedKey} name="고정 주기" stroke="#d97706" dot={false} strokeWidth={2} isAnimationActive={false} />
          <Line dataKey={dqnKey} name="DQN" stroke="#059669" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export function LiveCharts({ data }: Props) {
  return (
    <div className="live-chart-grid">
      <Chart data={data} title="남은 차량" fixedKey="fixedRemaining" dqnKey="dqnRemaining" unit="대" />
      <Chart data={data} title="전체 대기 행렬" fixedKey="fixedQueue" dqnKey="dqnQueue" unit="대" />
      <Chart data={data} title="평균 대기시간" fixedKey="fixedWait" dqnKey="dqnWait" unit="초" />
      <Chart data={data} title="누적 통과량" fixedKey="fixedThroughput" dqnKey="dqnThroughput" unit="대" />
    </div>
  )
}
