import { CheckCircle2, CircleDashed, TriangleAlert } from 'lucide-react'
import type { TrainingValidation } from '../types'

interface Props {
  validation?: TrainingValidation | null
  validationEpisodes: number
}

function deltaLabel(value: number, lowerIsBetter = true) {
  const favorable = lowerIsBetter ? value >= 0 : value >= 0
  const direction = lowerIsBetter
    ? (value >= 0 ? '감소' : '증가')
    : (value >= 0 ? '증가' : '감소')
  return { text: `${Math.abs(value).toFixed(1)}% ${direction}`, favorable }
}

export function FixedBaselinePanel({ validation, validationEpisodes }: Props) {
  const hasFixedBaseline = validation
    && Number.isFinite(validation.fixed_avg_waiting_time)
    && Number.isFinite(validation.fixed_max_waiting_time)
    && Number.isFinite(validation.fixed_avg_queue)
    && Number.isFinite(validation.fixed_throughput)

  if (!validation || !hasFixedBaseline) {
    return (
      <section className="panel baseline-panel baseline-empty">
        <header><div><h2>고정 주기 대비 학습 판정</h2><p>같은 교통량과 같은 검증 시드로 공정하게 비교합니다.</p></div><span className="verdict pending"><CircleDashed size={15} />검증 대기</span></header>
        <div className="baseline-guide">
          <strong>모델 합격선</strong>
          <span>평균 대기시간 10% 이상 감소</span>
          <span>평균 대기행렬 5% 이상 감소</span>
          <span>최대 대기시간이 고정식 이하이면서 120초 이하</span>
          <span>통과량이 고정식의 95% 이상</span>
        </div>
        <p className="baseline-note">{validation ? '이 결과는 기준 비교 기능을 추가하기 전에 만든 기록입니다. 새 학습의 첫 검증부터 자동 판정됩니다. ' : ''}보상은 학습 추세, 탐험률 ε는 탐색 정도를 보여줄 뿐 성능 합격 기준은 아닙니다.</p>
      </section>
    )
  }

  const rows = [
    {
      label: '평균 대기시간', dqn: validation.avg_waiting_time, fixed: validation.fixed_avg_waiting_time,
      unit: '초', delta: deltaLabel(validation.waiting_improvement_pct), target: '10% 이상 감소', pass: validation.waiting_target_met,
    },
    {
      label: '평균 대기행렬', dqn: validation.avg_queue, fixed: validation.fixed_avg_queue,
      unit: '대', delta: deltaLabel(validation.queue_improvement_pct), target: '5% 이상 감소', pass: validation.queue_target_met,
    },
    {
      label: '최대 대기시간', dqn: validation.max_waiting_time, fixed: validation.fixed_max_waiting_time,
      unit: '초', delta: deltaLabel(validation.max_waiting_improvement_pct), target: '고정식 이하 · 120초 이하', pass: validation.max_waiting_target_met,
    },
    {
      label: '통과량', dqn: validation.throughput, fixed: validation.fixed_throughput,
      unit: '대', delta: deltaLabel(validation.throughput_change_pct, false), target: '고정식의 95% 이상', pass: validation.throughput_target_met,
    },
  ]
  const passed = rows.filter((row) => row.pass).length
  const improving = validation.waiting_improvement_pct > 0 && validation.throughput_target_met
  const verdict = validation.beats_fixed ? '목표 달성' : improving ? '개선 중' : '목표 미달'

  return (
    <section className="panel baseline-panel">
      <header>
        <div><h2>고정 주기 대비 학습 판정</h2><p>{validation.timestep.toLocaleString()} 스텝 · 동일 검증 시드 {validationEpisodes}개 평균</p></div>
        <span className={`verdict ${validation.beats_fixed ? 'pass' : improving ? 'progress' : 'fail'}`}>
          {validation.beats_fixed ? <CheckCircle2 size={15} /> : <TriangleAlert size={15} />}{verdict}
        </span>
      </header>
      <div className="baseline-score"><strong>{passed} / 4 조건 통과</strong><span>{validation.beats_fixed ? '현재 검증에서는 고정 신호보다 우수합니다.' : '네 조건을 모두 만족해야 우수 모델로 판정합니다.'}</span></div>
      <div className="baseline-table" role="table" aria-label="고정 주기 대비 DQN 성능">
        <div className="baseline-table-head" role="row"><span>핵심 지표</span><span>DQN</span><span>고정 주기</span><span>차이</span><span>합격 기준</span></div>
        {rows.map((row) => <div className="baseline-table-row" role="row" key={row.label}>
          <strong>{row.label}</strong>
          <span>{row.dqn.toFixed(2)} {row.unit}</span>
          <span>{row.fixed.toFixed(2)} {row.unit}</span>
          <span className={row.delta.favorable ? 'positive' : 'negative'}>{row.delta.text}</span>
          <span className={`target ${row.pass ? 'pass' : 'fail'}`}>{row.pass ? <CheckCircle2 size={14} /> : <TriangleAlert size={14} />}{row.target}</span>
        </div>)}
      </div>
      <div className="baseline-foot">
        <span>참고: 신호 변경 DQN {validation.phase_changes.toFixed(1)}회 · 고정 주기 {validation.fixed_phase_changes.toFixed(1)}회</span>
        <strong>최종 제출용 판정은 비교실에서 학습에 쓰지 않은 시드 30회로 평균과 표준편차를 확인하세요.</strong>
      </div>
    </section>
  )
}
