import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, RotateCcw, Square } from 'lucide-react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, websocketUrl } from '../api'
import { IntersectionView } from '../components/IntersectionView'
import { FixedBaselinePanel } from '../components/FixedBaselinePanel'
import { MetricCard } from '../components/MetricCard'
import { scenarioLabel, statusLabel } from '../i18n'
import type { PageKey, TrainingFrame, TrainingProgress } from '../types'

const SESSION_STORAGE_KEY = 'traffic-control.active-training-session'
const activeStatuses = new Set(['PENDING', 'TRAINING', 'RESUMING', 'PAUSED'])
const runningStatuses = new Set(['PENDING', 'TRAINING', 'RESUMING'])
const queueLabels: Record<string, string> = {
  N_left: '북 좌회전', N_straight: '북 직진', S_left: '남 좌회전', S_straight: '남 직진',
  E_left: '동 좌회전', E_straight: '동 직진', W_left: '서 좌회전', W_straight: '서 직진',
}

interface TrainingConfig {
  mode?: 'fixed_steps' | 'auto_convergence'
  total_steps?: number
  maximum_steps?: number
  minimum_steps?: number
  validation_interval?: number
  validation_episodes?: number
  no_improvement_patience?: number
  minimum_improvement?: number
  scenario?: string
}

interface SessionResponse {
  id: string
  status: string
  detail?: Partial<TrainingProgress>
  config?: TrainingConfig
  history?: TrainingProgress[]
  frame?: TrainingFrame | null
}

function progressFromSession(response: SessionResponse): TrainingProgress {
  const maximum = response.config?.mode === 'auto_convergence'
    ? response.config.maximum_steps
    : response.config?.total_steps
  return {
    status: response.detail?.status ?? response.status,
    timesteps: response.detail?.timesteps ?? 0,
    total_timesteps: response.detail?.total_timesteps ?? maximum ?? 0,
    episodes: response.detail?.episodes ?? 0,
    epsilon: response.detail?.epsilon ?? 1,
    replay_buffer_size: response.detail?.replay_buffer_size ?? 0,
    rolling_episode_reward: response.detail?.rolling_episode_reward ?? 0,
    elapsed_seconds: response.detail?.elapsed_seconds ?? 0,
    ...response.detail,
  }
}

export function TrainingLab({ onNavigate }: { onNavigate: (page: PageKey) => void }) {
  const [mode, setMode] = useState<'fixed_steps' | 'auto_convergence'>('fixed_steps')
  const [steps, setSteps] = useState(20_000)
  const [minimumSteps, setMinimumSteps] = useState(10_000)
  const [validationInterval, setValidationInterval] = useState(5_000)
  const [validationEpisodes, setValidationEpisodes] = useState(3)
  const [patience, setPatience] = useState(8)
  const [minimumImprovement, setMinimumImprovement] = useState(0.05)
  const [scenario, setScenario] = useState('random')
  const [sessionId, setSessionId] = useState('')
  const [progress, setProgress] = useState<TrainingProgress | null>(null)
  const [history, setHistory] = useState<Array<TrainingProgress & { index: number }>>([])
  const [frame, setFrame] = useState<TrainingFrame | null>(null)
  const [error, setError] = useState('')
  const [restoring, setRestoring] = useState(true)
  const [connectionMode, setConnectionMode] = useState<'websocket' | 'polling' | 'offline'>('offline')
  const socketRef = useRef<WebSocket | null>(null)
  const socketSessionRef = useRef('')
  const socketGenerationRef = useRef(0)
  const lastSocketMessageAtRef = useRef(0)

  const hydrate = useCallback((response: SessionResponse) => {
    setSessionId(response.id)
    setProgress(progressFromSession(response))
    setFrame(response.frame ?? null)
    setHistory((response.history ?? []).map((item, index) => ({ ...item, index })))
    window.localStorage.setItem(SESSION_STORAGE_KEY, response.id)
    const config = response.config
    if (!config) return
    if (config.mode) setMode(config.mode)
    setSteps(config.mode === 'auto_convergence' ? config.maximum_steps ?? 20_000 : config.total_steps ?? 20_000)
    if (config.minimum_steps !== undefined) setMinimumSteps(config.minimum_steps)
    if (config.validation_interval !== undefined) setValidationInterval(config.validation_interval)
    if (config.validation_episodes !== undefined) setValidationEpisodes(config.validation_episodes)
    if (config.no_improvement_patience !== undefined) setPatience(config.no_improvement_patience)
    if (config.minimum_improvement !== undefined) setMinimumImprovement(config.minimum_improvement)
    if (config.scenario) setScenario(config.scenario)
  }, [])

  const connect = useCallback((id: string, force = false) => {
    const current = socketRef.current
    if (
      !force
      && socketSessionRef.current === id
      && current
      && (current.readyState === WebSocket.OPEN || current.readyState === WebSocket.CONNECTING)
    ) return

    const generation = socketGenerationRef.current + 1
    socketGenerationRef.current = generation
    current?.close()
    const socket = new WebSocket(websocketUrl(`/ws/training/${id}`))
    socketRef.current = socket
    socketSessionRef.current = id
    socket.onopen = () => {
      if (socketGenerationRef.current === generation) {
        lastSocketMessageAtRef.current = Date.now()
        setConnectionMode('websocket')
      }
    }
    socket.onmessage = (event) => {
      if (socketGenerationRef.current !== generation) return
      lastSocketMessageAtRef.current = Date.now()
      const message = JSON.parse(event.data) as TrainingFrame | TrainingProgress | SessionResponse | { type: 'heartbeat' }
      if ('detail' in message) {
        hydrate(message)
        return
      }
      if ('type' in message && message.type === 'training_frame') {
        setFrame(message as TrainingFrame)
        return
      }
      if (typeof (message as TrainingProgress).timesteps !== 'number') return
      const next = message as TrainingProgress
      setProgress(next)
      setHistory((current) => [...current.slice(-299), { ...next, index: current.length }])
    }
    socket.onerror = () => {
      if (socketGenerationRef.current === generation) setConnectionMode('polling')
    }
    socket.onclose = () => {
      if (socketGenerationRef.current !== generation) return
      socketRef.current = null
      socketSessionRef.current = ''
      lastSocketMessageAtRef.current = 0
      setConnectionMode('polling')
    }
  }, [hydrate])

  useEffect(() => {
    let cancelled = false
    const restore = async () => {
      try {
        const stored = window.localStorage.getItem(SESSION_STORAGE_KEY)
        let response: SessionResponse | undefined
        if (stored) {
          try { response = await api<SessionResponse>(`/api/training/${stored}`) } catch { window.localStorage.removeItem(SESSION_STORAGE_KEY) }
        }
        if (!response) {
          const sessions = await api<SessionResponse[]>('/api/training')
          response = sessions.find((item) => activeStatuses.has(item.status))
        }
        if (!cancelled && response) {
          hydrate(response)
          if (activeStatuses.has(response.status)) connect(response.id)
        }
      } catch (reason) {
        if (!cancelled) setError((reason as Error).message)
      } finally {
        if (!cancelled) setRestoring(false)
      }
    }
    void restore()
    return () => {
      cancelled = true
      socketGenerationRef.current += 1
      socketRef.current?.close()
      socketRef.current = null
      socketSessionRef.current = ''
    }
  }, [connect, hydrate])

  useEffect(() => {
    if (!sessionId || !activeStatuses.has(progress?.status ?? '')) return
    let cancelled = false
    let timer: number | undefined

    const synchronize = async () => {
      try {
        const response = await api<SessionResponse>(`/api/training/${sessionId}`)
        if (cancelled) return
        hydrate(response)
        setError('')
        const socketHealthy = socketRef.current?.readyState === WebSocket.OPEN
          && Date.now() - lastSocketMessageAtRef.current < 7_000
        if (activeStatuses.has(response.status)) connect(response.id, !socketHealthy)
        if (!socketHealthy) {
          setConnectionMode('polling')
        }
      } catch (reason) {
        if (!cancelled) {
          setConnectionMode('offline')
          setError((reason as Error).message)
        }
      } finally {
        if (!cancelled) timer = window.setTimeout(synchronize, 2_000)
      }
    }

    timer = window.setTimeout(synchronize, 500)
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [connect, hydrate, progress?.status, sessionId])

  const start = async () => {
    setError('')
    setHistory([])
    setFrame(null)
    try {
      const response = await api<SessionResponse>('/api/training', {
        method: 'POST',
        body: JSON.stringify({
          mode, total_steps: steps, maximum_steps: steps, minimum_steps: minimumSteps,
          validation_interval: validationInterval, validation_episodes: validationEpisodes,
          no_improvement_patience: patience, minimum_improvement: minimumImprovement,
          scenario, episode_seconds: 120,
        }),
      })
      hydrate(response)
      connect(response.id)
    } catch (reason) { setError((reason as Error).message) }
  }

  const control = async (action: 'pause' | 'resume' | 'stop') => {
    if (!sessionId) return
    setError('')
    try {
      const response = await api<SessionResponse>(`/api/training/${sessionId}/${action}`, { method: 'POST' })
      setProgress(progressFromSession(response))
      if (action === 'resume') connect(sessionId)
    } catch (reason) { setError((reason as Error).message) }
  }

  const percent = useMemo(
    () => progress ? Math.min(100, progress.timesteps / Math.max(progress.total_timesteps, 1) * 100) : 0,
    [progress],
  )
  const hasActiveSession = activeStatuses.has(progress?.status ?? '')
  const isRunning = runningStatuses.has(progress?.status ?? '')
  const queueEntries = Object.entries(frame?.side.metrics.approach_queues ?? {})

  return (
    <div className="page training-page">
      <div className="page-heading"><div><h1>강화학습 학습실</h1><p>고정 검증 시드와 안전한 체크포인트를 사용하는 DQN 학습</p></div><span className="status-line"><i className={progress?.status === 'TRAINING' ? 'pulse' : ''} />{restoring ? '세션 확인 중' : statusLabel(progress?.status)}</span></div>
      {error ? <div className="error-banner">{error}</div> : null}
      <div className="lab-layout">
        <aside className="panel control-panel">
          <h2>학습 설정</h2>
          <label>학습 모드<select value={mode} onChange={(event) => setMode(event.target.value as typeof mode)}><option value="fixed_steps">고정 스텝</option><option value="auto_convergence">자동 수렴</option></select></label>
          <label>{mode === 'fixed_steps' ? '전체 스텝' : '최대 스텝'}<input type="number" min="100" value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>
          {mode === 'auto_convergence' ? <><label>최소 스텝<input type="number" min="0" value={minimumSteps} onChange={(event) => setMinimumSteps(Number(event.target.value))} /></label><label>검증 간격<input type="number" min="1" value={validationInterval} onChange={(event) => setValidationInterval(Number(event.target.value))} /></label><label>검증 에피소드 수<input type="number" min="1" max="100" value={validationEpisodes} onChange={(event) => setValidationEpisodes(Number(event.target.value))} /></label><label>개선 없음 허용 횟수<input type="number" min="1" value={patience} onChange={(event) => setPatience(Number(event.target.value))} /></label><label>최소 개선값<input type="number" min="0" step="0.01" value={minimumImprovement} onChange={(event) => setMinimumImprovement(Number(event.target.value))} /></label></> : null}
          <label>교통 시나리오<select value={scenario} onChange={(event) => setScenario(event.target.value)}><option value="random">무작위 교통량</option><option value="uniform">균등 교통량</option><option value="north_south_congested">남북 방향 혼잡</option><option value="east_west_congested">동서 방향 혼잡</option><option value="left_turn_congested">좌회전 혼잡</option><option value="heavy">전체 혼잡</option><option value="low">낮은 교통량</option></select></label>
          <div className="control-buttons"><button className="primary" onClick={start} disabled={hasActiveSession}><Play size={16} />학습 시작</button><button onClick={() => void control('pause')} disabled={!isRunning}><Pause size={16} />일시정지</button><button onClick={() => void control('resume')} disabled={progress?.status !== 'PAUSED'}><RotateCcw size={16} />재개</button><button onClick={() => void control('stop')} disabled={!isRunning}><Square size={16} />중지</button></div>
          <div className="validation-rule"><strong>무엇을 보면 되나요?</strong><p>보상보다 고정 주기 대비 평균 대기·행렬·최대 대기·통과량을 보세요. 오른쪽 판정표가 네 조건을 모두 통과하면 1차 목표 달성입니다.</p></div>
        </aside>
        <div className="lab-main">
          <section className="panel training-status">
            <header><h2>학습 진행 상황</h2><span>{progress?.timesteps.toLocaleString() ?? 0} / {progress?.total_timesteps.toLocaleString() ?? steps.toLocaleString()}</span></header>
            <div className="progress-track"><span style={{ width: `${percent}%` }} /></div>
            <div className="metric-row">
              <MetricCard label="에피소드" value={progress?.episodes ?? 0} />
              <MetricCard label="탐험률 ε" value={(progress?.epsilon ?? 1).toFixed(3)} />
              <MetricCard label="경험 재현 버퍼" value={progress?.replay_buffer_size.toLocaleString() ?? 0} />
              <MetricCard label="최근 평균 보상" value={(progress?.rolling_episode_reward ?? 0).toFixed(2)} />
            </div>
          </section>

          <FixedBaselinePanel validation={progress?.validation ?? progress?.best_validation} validationEpisodes={validationEpisodes} />

          <section className="panel training-live">
            <header><h2>실시간 학습 시뮬레이션</h2><span>{frame ? `${frame.timestep.toLocaleString()} 스텝 · ${frame.simulation_time.toFixed(0)}초` : '학습 시작 대기'}</span></header>
            <div className="training-live-grid">
              <IntersectionView title="학습 중인 DQN" tone="dqn" side={frame?.side} bounds={frame?.network_bounds} />
              <aside className="training-frame-detail">
                <div className="training-frame-meta"><span>현재 에피소드<strong>{frame?.episode ?? 0}</strong></span><span>시나리오<strong>{scenarioLabel(frame?.scenario ?? scenario)}</strong></span><span>현재 스텝<strong>{frame?.timestep.toLocaleString() ?? 0}</strong></span><span>시뮬레이션 시간<strong>{frame?.simulation_time.toFixed(1) ?? '0.0'}초</strong></span></div>
                <div className="queue-title"><strong>방향별 대기 행렬</strong><span>정지 차량 수</span></div>
                <div className="queue-mini-grid">{queueEntries.length ? queueEntries.map(([key, value]) => <div key={key}><span>{queueLabels[key] ?? key}</span><strong>{value.toFixed(0)}</strong></div>) : <p>학습이 시작되면 실제 SUMO 차량 상태가 표시됩니다.</p>}</div>
              </aside>
            </div>
          </section>

          <section className="panel chart-panel tall"><header><strong>보상 및 탐험률</strong><span className={`connection-mode ${connectionMode}`}>{connectionMode === 'websocket' ? '실시간 연결' : connectionMode === 'polling' ? '2초 자동 동기화' : '연결 확인 중'}</span></header><ResponsiveContainer width="100%" height={300}><LineChart data={history}><CartesianGrid stroke="#d9e2ea" vertical={false} /><XAxis dataKey="timesteps" stroke="#64748b" /><YAxis yAxisId="reward" stroke="#64748b" /><YAxis yAxisId="epsilon" orientation="right" domain={[0, 1]} stroke="#64748b" /><Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '6px', color: '#0f172a' }} /><Legend /><Line yAxisId="reward" dataKey="rolling_episode_reward" name="보상" stroke="#0284c7" dot={false} isAnimationActive={false} /><Line yAxisId="epsilon" dataKey="epsilon" name="탐험률" stroke="#059669" dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></section>
          <section className="panel chart-panel tall"><header><strong>검증 교통 지표</strong><span>고정 시드 평가 에피소드</span></header><ResponsiveContainer width="100%" height={280}><LineChart data={history.filter((row) => row.validation)}><CartesianGrid stroke="#d9e2ea" vertical={false} /><XAxis dataKey="timesteps" stroke="#64748b" /><YAxis stroke="#64748b" /><Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '6px', color: '#0f172a' }} /><Legend /><Line dataKey="validation.avg_waiting_time" name="평균 대기" stroke="#0284c7" dot={false} /><Line dataKey="validation.max_waiting_time" name="최대 대기" stroke="#dc2626" dot={false} /><Line dataKey="validation.avg_queue" name="평균 대기 행렬" stroke="#d97706" dot={false} /><Line dataKey="validation.throughput" name="통과량" stroke="#059669" dot={false} /></LineChart></ResponsiveContainer></section>
          <section className="panel best-model"><header><h2>최고 검증 결과</h2></header>{progress?.best_validation ? <div className="metric-row"><MetricCard label="스텝" value={progress.best_validation.timestep?.toLocaleString() ?? '—'} /><MetricCard label="평균 대기" value={(progress.best_validation.avg_waiting_time ?? 0).toFixed(2)} unit="초" /><MetricCard label="최대 대기" value={(progress.best_validation.max_waiting_time ?? 0).toFixed(2)} unit="초" /><MetricCard label="통과량" value={(progress.best_validation.throughput ?? 0).toFixed(1)} /></div> : <div className="empty-state">{progress?.validation ? '안전 기준을 통과한 최적 모델이 아직 없습니다.' : '설정한 간격에 도달하면 검증을 시작합니다.'}</div>}</section>
          <section className="panel checkpoint-history"><header><h2>체크포인트 기록</h2><button className="text-button" onClick={() => onNavigate('comparison')}>비교실에서 확인</button></header>{progress?.checkpoints?.length ? <div className="checkpoint-list">{progress.checkpoints.slice(-8).map((checkpoint) => <div key={`${checkpoint.name}-${checkpoint.timestep}`}><strong>{checkpoint.name === 'best' ? '최적 모델' : checkpoint.name === 'final' ? '최종 모델' : checkpoint.name}</strong><span>{checkpoint.timestep.toLocaleString()} 스텝</span><code>{checkpoint.model_path}</code></div>)}</div> : <div className="empty-state">주기별·최적·일시정지·최종 체크포인트가 여기에 표시됩니다.</div>}</section>
        </div>
      </div>
    </div>
  )
}
