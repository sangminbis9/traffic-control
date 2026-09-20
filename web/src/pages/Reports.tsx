import { useEffect, useState } from 'react'
import { Download, FileArchive, RefreshCw } from 'lucide-react'
import { api } from '../api'
import { experimentKindLabel, statusLabel } from '../i18n'
import type { ExperimentRecord } from '../types'

export function Reports() {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const load = () => api<ExperimentRecord[]>('/api/experiments').then(setExperiments).catch((reason: Error) => setError(reason.message))
  useEffect(() => {
    void load()
  }, [])
  const generate = async (id: string) => {
    setBusy(id)
    try {
      const report = await api<{ id: string }>(`/api/reports/${id}`, { method: 'POST' })
      window.location.href = `/api/reports/${report.id}/download`
    } catch (reason) { setError((reason as Error).message) } finally { setBusy('') }
  }
  return <div className="page reports-page"><div className="page-heading"><div><h1>보고서</h1><p>실험 기록과 발표용 결과 파일</p></div><button className="secondary" onClick={load}><RefreshCw size={15} />새로고침</button></div>{error ? <div className="error-banner">{error}</div> : null}<section className="panel report-list"><header><h2>실험 기록</h2><span>{experiments.length}건</span></header>{experiments.length ? experiments.map((experiment) => <article key={experiment.id}><FileArchive /><div><strong>{experimentKindLabel(experiment.kind)}</strong><span>{new Date(experiment.created_at).toLocaleString('ko-KR')} · {statusLabel(experiment.status)}</span><small>{experiment.id}</small></div><button onClick={() => generate(experiment.id)} disabled={busy === experiment.id || experiment.status !== 'COMPLETED'}><Download size={15} />{busy === experiment.id ? '생성 중…' : '발표 자료 받기'}</button></article>) : <div className="empty-state">완료된 비교 실험의 설정과 결과 파일이 여기에 표시됩니다.</div>}</section></div>
}
