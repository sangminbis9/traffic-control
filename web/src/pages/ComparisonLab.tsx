import { useEffect, useMemo, useRef, useState } from 'react'
import { Maximize2, Pause, Play, RotateCcw, Square } from 'lucide-react'
import { api, websocketUrl } from '../api'
import { IntersectionView } from '../components/IntersectionView'
import { LiveCharts, type ChartPoint } from '../components/LiveCharts'
import { scenarioLabel, statusLabel } from '../i18n'
import type { LanePlacement, ModelMetadata, PageKey, SimulationFrame } from '../types'

const approaches = ['N', 'S', 'E', 'W'] as const
const emptyLane = (): LanePlacement => ({ left: 1, straight: 1, lane3_total: 1, lane3_right: 0 })

interface Props {
  onNavigate: (page: PageKey) => void
  onFrame: (frame: SimulationFrame) => void
}

export function ComparisonLab({ onNavigate, onFrame }: Props) {
  const [testMode, setTestMode] = useState<'initial' | 'continuous'>('initial')
  const [runMode, setRunMode] = useState<'single' | 'batch'>('single')
  const [lanes, setLanes] = useState<Record<typeof approaches[number], LanePlacement>>(() => ({ N: emptyLane(), S: emptyLane(), E: emptyLane(), W: emptyLane() }))
  const [models, setModels] = useState<ModelMetadata[]>([])
  const [modelPath, setModelPath] = useState('')
  const [seed, setSeed] = useState(20_001)
  const [runs, setRuns] = useState(30)
  const [scenario, setScenario] = useState('uniform')
  const [rates, setRates] = useState({ n_rate: 0.15, s_rate: 0.15, e_rate: 0.15, w_rate: 0.15 })
  const [turnRatios, setTurnRatios] = useState({ left_ratio: 0.2, straight_ratio: 0.6, right_ratio: 0.2 })
  const [speed, setSpeed] = useState<'0.5' | '1' | '2' | '4' | 'max'>('max')
  const [experimentId, setExperimentId] = useState('')
  const [frame, setFrame] = useState<SimulationFrame | null>(null)
  const [status, setStatus] = useState('READY')
  const [error, setError] = useState('')
  const [chartData, setChartData] = useState<ChartPoint[]>([])
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  useEffect(() => {
    api<ModelMetadata[]>('/api/models').then((rows) => { setModels(rows); setModelPath(rows.find((row) => row.compatible)?.path ?? '') }).catch((reason: Error) => setError(reason.message))
    return () => socketRef.current?.close()
  }, [])
  const totalVehicles = useMemo(() => approaches.reduce((sum, approach) => sum + lanes[approach].left + lanes[approach].straight + lanes[approach].lane3_total, 0), [lanes])
  const resultFixed = (result?.fixed ?? {}) as Record<string, number>
  const resultDqn = (result?.dqn ?? {}) as Record<string, number>
  const summaryMetrics = [
    ['clearance_time', '전체 통과 시간', '초'], ['avg_waiting_time', '평균 대기시간', '초'],
    ['max_waiting_time', '최대 대기시간', '초'], ['avg_queue', '평균 대기 행렬', ''],
    ['max_queue', '최대 대기 행렬', ''], ['throughput', '통과량', ''],
    ['phase_changes', '신호 변경 횟수', ''], ['queue_auc', '대기 행렬 누적값', '대·초'],
  ] as const
  const updateLane = (approach: typeof approaches[number], field: keyof LanePlacement, value: number) => {
    setLanes((current) => {
      const next = { ...current[approach], [field]: Math.max(0, Math.min(5, value)) }
      if (field === 'lane3_total' && next.lane3_right > next.lane3_total) next.lane3_right = next.lane3_total
      return { ...current, [approach]: next }
    })
  }
  const start = async () => {
    setError(''); setResult(null); setChartData([])
    try {
      const customRates = scenario === 'custom' ? rates : {}
      const response = await api<{ id: string; status: string }>('/api/comparisons', { method: 'POST', body: JSON.stringify({ test_mode: testMode, run_mode: runMode, initial: lanes, continuous: { scenario: scenario === 'custom' ? 'uniform' : scenario, ...customRates, ...turnRatios }, model_path: modelPath, seed, runs: runMode === 'batch' ? runs : 1, duration: testMode === 'initial' ? 300 : 180, render_interval: 1, speed }) })
      setExperimentId(response.id); setStatus(response.status)
      const socket = new WebSocket(websocketUrl(`/ws/comparisons/${response.id}`))
      socketRef.current = socket
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === 'simulation_frame') {
          const next = message as SimulationFrame
          setFrame(next); onFrame(next); setStatus(next.status)
          setChartData((current) => [...current.slice(-599), { time: next.simulation_time, fixedRemaining: next.fixed.metrics.vehicles_remaining, dqnRemaining: next.dqn.metrics.vehicles_remaining, fixedQueue: next.fixed.metrics.current_queue, dqnQueue: next.dqn.metrics.current_queue, fixedWait: next.fixed.metrics.average_waiting, dqnWait: next.dqn.metrics.average_waiting, fixedThroughput: next.fixed.metrics.throughput, dqnThroughput: next.dqn.metrics.throughput, fixedPhase: next.fixed.phase, dqnPhase: next.dqn.phase }])
        } else if (message.type === 'comparison_complete') { setStatus(message.status); setResult(message.result) }
        else if (message.type === 'comparison_error') { setStatus('FAILED'); setError(message.error) }
        else if (message.type === 'batch_progress') { setStatus(`BATCH ${message.completed_runs}/${message.total_runs}`) }
      }
    } catch (reason) { setError((reason as Error).message) }
  }
  const control = async (action: 'pause' | 'resume' | 'stop') => {
    if (!experimentId) return
    try { const response = await api<{ status: string }>(`/api/comparisons/${experimentId}/${action}`, { method: 'POST' }); setStatus(response.status) } catch (reason) { setError((reason as Error).message) }
  }
  const changeSpeed = async (value: typeof speed) => {
    setSpeed(value)
    if (!experimentId || !['RUNNING', 'PAUSED'].includes(status)) return
    try { await api(`/api/comparisons/${experimentId}/speed`, { method: 'POST', body: JSON.stringify({ speed: value }) }) } catch (reason) { setError((reason as Error).message) }
  }
  return (
    <div className="page comparison-page">
      <div className="page-heading"><div><h1>제어기 비교실</h1><p>동일한 교통량·신호 시간·시드를 적용한 동기화 시뮬레이션</p></div><span className="status-line"><i className={status === 'RUNNING' ? 'pulse' : ''} />{statusLabel(status)}</span></div>
      {error ? <div className="error-banner">{error}</div> : null}
      <div className="comparison-layout">
        <aside className="panel comparison-controls">
          <div className="segmented"><button className={testMode === 'initial' ? 'active' : ''} onClick={() => setTestMode('initial')}>초기 차량 통과</button><button className={testMode === 'continuous' ? 'active' : ''} onClick={() => setTestMode('continuous')}>연속 교통량</button></div>
          <div className="segmented"><button className={runMode === 'single' ? 'active' : ''} onClick={() => setRunMode('single')}>단일 실행</button><button className={runMode === 'batch' ? 'active' : ''} onClick={() => setRunMode('batch')}>반복 실행</button></div>
          {testMode === 'initial' ? <div className="lane-editor"><header><strong>초기 차량</strong><span>총 {totalVehicles}대</span></header>{approaches.map((approach) => <fieldset key={approach}><legend>{approach === 'N' ? '북쪽' : approach === 'S' ? '남쪽' : approach === 'E' ? '동쪽' : '서쪽'}</legend><label>좌회전<input type="number" min="0" max="5" value={lanes[approach].left} onChange={(event) => updateLane(approach, 'left', Number(event.target.value))} /></label><label>직진<input type="number" min="0" max="5" value={lanes[approach].straight} onChange={(event) => updateLane(approach, 'straight', Number(event.target.value))} /></label><label>3차로 전체<input type="number" min="0" max="5" value={lanes[approach].lane3_total} onChange={(event) => updateLane(approach, 'lane3_total', Number(event.target.value))} /></label><label>우회전 차량<input type="number" min="0" max={lanes[approach].lane3_total} value={lanes[approach].lane3_right} onChange={(event) => updateLane(approach, 'lane3_right', Number(event.target.value))} /></label></fieldset>)}</div> : <div className="continuous-editor"><label>교통량 프리셋<select value={scenario} onChange={(event) => setScenario(event.target.value)}>{['uniform', 'north_south_congested', 'east_west_congested', 'left_turn_congested', 'heavy', 'low', 'random', 'custom'].map((value) => <option key={value} value={value}>{scenarioLabel(value)}</option>)}</select></label>{scenario === 'custom' ? <div className="rate-grid">{(['n_rate', 's_rate', 'e_rate', 'w_rate'] as const).map((key) => <label key={key}>{key[0].toUpperCase()} 발생률<input type="number" min="0" max="1" step="0.01" value={rates[key]} onChange={(event) => setRates((current) => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}</div> : null}<div className="rate-grid ratios">{(['left_ratio', 'straight_ratio', 'right_ratio'] as const).map((key) => <label key={key}>{key === 'left_ratio' ? '좌회전' : key === 'straight_ratio' ? '직진' : '우회전'}<input type="number" min="0" max="1" step="0.05" value={turnRatios[key]} onChange={(event) => setTurnRatios((current) => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}</div><p>회전 비율의 합은 1.00이어야 합니다. 직접 설정 발생률은 각 방향의 초당 차량 생성 확률입니다.</p></div>}
          <label>모델<select value={modelPath} onChange={(event) => setModelPath(event.target.value)}>{models.map((model) => <option key={model.path} value={model.path} disabled={!model.compatible}>{model.name}{model.compatible ? '' : ' (호환 불가)'}</option>)}</select></label>
          <div className="inline-fields"><label>무작위 시드<input type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>{runMode === 'batch' ? <label>반복 횟수<input type="number" min="1" max="100" value={runs} onChange={(event) => setRuns(Number(event.target.value))} /></label> : null}</div>
          <button className="primary run-button" onClick={start} disabled={!modelPath}><Play size={16} />비교 실행</button>
          <div className="control-buttons compact"><button onClick={() => control('pause')}><Pause size={15} />일시정지</button><button onClick={() => control('resume')}><RotateCcw size={15} />재개</button><button onClick={() => control('stop')}><Square size={15} />중지</button></div>
          <div className="speed-control"><span>시뮬레이션 속도</span>{(['0.5', '1', '2', '4', 'max'] as const).map((value) => <button key={value} className={speed === value ? 'active' : ''} onClick={() => void changeSpeed(value)}>{value === 'max' ? '최대' : `${value}x`}</button>)}</div>
          <button className="presentation-button" onClick={() => onNavigate('presentation')}><Maximize2 size={16} />발표 모드</button>
        </aside>
        <div className="comparison-main">
          <div className="simulation-clock">시뮬레이션 시간 <strong>{(frame?.simulation_time ?? 0).toFixed(1)}초</strong></div>
          <div className="intersection-grid"><IntersectionView title="고정 주기" tone="fixed" side={frame?.fixed} bounds={frame?.network_bounds} /><IntersectionView title="DQN" tone="dqn" side={frame?.dqn} bounds={frame?.network_bounds} /></div>
          {frame ? <section className="panel delta-strip"><strong>DQN vs 고정 주기</strong>{['average_waiting', 'current_queue', 'throughput'].map((metric) => { const value = frame.delta[metric]; const label = metric === 'average_waiting' ? '평균 대기' : metric === 'current_queue' ? '대기 행렬' : '통과량'; return <div key={metric}><span>{label}</span><b className={value !== null && ((metric === 'throughput' && value > 0) || (metric !== 'throughput' && value < 0)) ? 'positive' : ''}>{value === null ? '—' : `${value > 0 ? '+' : ''}${value.toFixed(1)}%`}</b></div> })}</section> : null}
          <LiveCharts data={chartData} />
          <section className="panel phase-timeline"><header><h2>신호 단계 타임라인</h2><span>0 남북 직진 · 1 남북 좌회전 · 2 동서 직진 · 3 동서 좌회전</span></header>{(['fixed', 'dqn'] as const).map((key) => <div key={key}><strong>{key === 'fixed' ? '고정 주기' : 'DQN'}</strong><div className="phase-bars">{chartData.slice(-80).map((point, index) => <i key={index} className={`phase-${key === 'fixed' ? point.fixedPhase : point.dqnPhase}`} />)}</div></div>)}</section>
          {result ? <section className="panel result-summary"><header><h2>실험 완료</h2><span>측정값을 그대로 비교합니다</span></header>{Number.isFinite(resultFixed.clearance_time) && Number.isFinite(resultDqn.clearance_time) ? <p className="objective-summary">{resultDqn.clearance_time < resultFixed.clearance_time ? `DQN이 모든 차량을 ${(resultFixed.clearance_time - resultDqn.clearance_time).toFixed(1)}초 먼저 통과시켰습니다.` : resultDqn.clearance_time > resultFixed.clearance_time ? `고정 주기 제어가 모든 차량을 ${(resultDqn.clearance_time - resultFixed.clearance_time).toFixed(1)}초 먼저 통과시켰습니다.` : '두 제어기가 같은 시간에 모든 차량을 통과시켰습니다.'}</p> : null}<div className="summary-table-wrap"><table><thead><tr><th>지표</th><th>고정 주기</th><th>DQN</th><th>DQN − 고정 주기</th></tr></thead><tbody>{summaryMetrics.map(([key, label, unit]) => { const fixed = resultFixed[key]; const dqn = resultDqn[key]; if (!Number.isFinite(fixed) || !Number.isFinite(dqn)) return null; const difference = dqn - fixed; return <tr key={key}><th>{label}</th><td>{fixed.toFixed(2)} {unit}</td><td>{dqn.toFixed(2)} {unit}</td><td className={difference < 0 && key !== 'throughput' ? 'positive' : difference > 0 && key === 'throughput' ? 'positive' : ''}>{difference > 0 ? '+' : ''}{difference.toFixed(2)} {unit}</td></tr> })}</tbody></table></div><details><summary>재현성 정보와 원본 결과</summary><pre>{JSON.stringify(result, null, 2)}</pre></details></section> : null}
        </div>
      </div>
    </div>
  )
}
