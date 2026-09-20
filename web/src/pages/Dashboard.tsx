import { useEffect, useState } from 'react'
import { ArrowRight, FilePlus2, FlaskConical, GitCompareArrows } from 'lucide-react'
import { api } from '../api'
import { MetricCard } from '../components/MetricCard'
import { experimentKindLabel, scenarioLabel, statusLabel } from '../i18n'
import type { ExperimentRecord, ModelMetadata, PageKey, TrainingRecord } from '../types'

export function Dashboard({ onNavigate }: { onNavigate: (page: PageKey) => void }) {
  const [models, setModels] = useState<ModelMetadata[]>([])
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([])
  const [training, setTraining] = useState<TrainingRecord[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    Promise.all([api<ModelMetadata[]>('/api/models'), api<ExperimentRecord[]>('/api/experiments'), api<TrainingRecord[]>('/api/training')])
      .then(([modelRows, experimentRows, trainingRows]) => { setModels(modelRows); setExperiments(experimentRows); setTraining(trainingRows) })
      .catch((reason: Error) => setError(reason.message))
  }, [])
  const active = models.find((model) => model.compatible)
  const latest = experiments[0]
  const latestTraining = training[0]
  return (
    <div className="page dashboard-page">
      <div className="page-heading"><div><h1>대시보드</h1><p>교통 신호 제어 실험과 시스템 상태</p></div><span className="status-line"><i />연구 시스템 온라인</span></div>
      {error ? <div className="error-banner">API 연결 오류: {error}</div> : null}
      <div className="dashboard-top">
        <section className="panel active-model">
          <header><h2>활성 모델</h2><span className={active ? 'ready' : 'muted'}>{active ? '호환 가능' : '사용 불가'}</span></header>
          <div className="model-name"><FlaskConical /><div><strong>{active?.name ?? '모델 없음'}</strong><span>심층 Q 신경망(DQN)</span></div></div>
          <dl><div><dt>파일</dt><dd>{active?.relative_path ?? '—'}</dd></div><div><dt>학습 스텝</dt><dd>{active?.timesteps.toLocaleString() ?? '—'}</dd></div><div><dt>생성일</dt><dd>{active ? new Date(active.created_at).toLocaleDateString('ko-KR') : '—'}</dd></div><div><dt>시나리오</dt><dd>{scenarioLabel(active?.training_scenario)}</dd></div><div><dt>검증 점수</dt><dd>{active?.last_validation_score?.toFixed(3) ?? '—'}</dd></div><div><dt>SHA-256</dt><dd>{active?.sha256.slice(0, 12) ?? '—'}</dd></div></dl>
        </section>
        <section className="panel recent-training">
          <header><h2>실험 준비 상태</h2><button className="text-button" onClick={() => onNavigate('training')}>학습실 열기 <ArrowRight size={15} /></button></header>
          <div className="readiness-track"><span style={{ width: active ? '100%' : '15%' }} /></div>
          <div className="readiness-copy"><strong>{latestTraining ? `${statusLabel(latestTraining.status)} · ${(latestTraining.detail?.timesteps ?? 0).toLocaleString()} 스텝` : active ? '12개 관측값 호환 확인' : '호환 모델 대기 중'}</strong><p>{latestTraining?.detail?.current_checkpoint ? `체크포인트: ${latestTraining.detail.current_checkpoint}` : '학습실에서 학습, 경험 재현 버퍼 체크포인트, 고정 시드 검증과 최적 모델 선정을 진행할 수 있습니다.'}</p></div>
        </section>
        <section className="panel latest-comparison">
          <header><h2>최근 비교 실험</h2><button className="text-button" onClick={() => onNavigate('comparison')}>비교실 열기 <ArrowRight size={15} /></button></header>
          {latest ? <div className="latest-record"><strong>{experimentKindLabel(latest.kind)}</strong><span>{statusLabel(latest.status)}</span><small>{new Date(latest.created_at).toLocaleString('ko-KR')}</small></div> : <div className="empty-state">아직 실행된 비교 실험이 없습니다.</div>}
        </section>
      </div>
      <section className="panel system-overview">
        <header><h2>연구 파이프라인</h2><span>실제 SUMO 데이터만 사용</span></header>
        <div className="pipeline">
          {['교통 시나리오', '동일 조건 SUMO', '고정 주기 / DQN', '성능 지표', '발표 자료'].map((label, index) => <div key={label}><span>{index + 1}</span><strong>{label}</strong></div>)}
        </div>
        <div className="metric-row">
          <MetricCard label="등록 모델" value={models.length} />
          <MetricCard label="완료 실험" value={experiments.filter((row) => row.status === 'COMPLETED').length} />
          <MetricCard label="관측값" value={12} />
          <MetricCard label="제어 행동" value={4} />
        </div>
      </section>
      <section className="panel quick-actions">
        <header><h2>빠른 실행</h2></header>
        <div className="action-grid">
          <button onClick={() => onNavigate('training')}><FlaskConical /><span><strong>새 학습</strong><small>고정 스텝 또는 자동 수렴</small></span><ArrowRight /></button>
          <button onClick={() => onNavigate('comparison')}><FilePlus2 /><span><strong>테스트 조건 만들기</strong><small>12개 차로 또는 연속 교통량</small></span><ArrowRight /></button>
          <button onClick={() => onNavigate('comparison')}><GitCompareArrows /><span><strong>비교 실행</strong><small>동일 교통량과 동기화된 시간</small></span><ArrowRight /></button>
          <button onClick={() => onNavigate('reports')}><FilePlus2 /><span><strong>보고서 열기</strong><small>실험 기록과 발표 자료</small></span><ArrowRight /></button>
        </div>
      </section>
    </div>
  )
}
