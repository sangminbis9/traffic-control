const statusLabels: Record<string, string> = {
  READY: '준비',
  PENDING: '대기 중',
  TRAINING: '학습 중',
  RESUMING: '재개 중',
  RUNNING: '실행 중',
  PAUSED: '일시정지',
  STOPPED: '중지됨',
  COMPLETED: '완료',
  FAILED: '실패',
  INTERRUPTED: '서버 재시작으로 중단',
  IDLE: '대기',
}

const scenarioLabels: Record<string, string> = {
  random: '무작위 교통량',
  uniform: '균등 교통량',
  north_south_congested: '남북 방향 혼잡',
  east_west_congested: '동서 방향 혼잡',
  left_turn_congested: '좌회전 혼잡',
  heavy: '전체 혼잡',
  low: '낮은 교통량',
  custom: '방향별 직접 설정',
}

const phaseLabels: Record<string, string> = {
  'NS Straight': '남북 직진',
  'NS Left': '남북 좌회전',
  'EW Straight': '동서 직진',
  'EW Left': '동서 좌회전',
}

const signalLabels: Record<string, string> = {
  GREEN: '녹색',
  YELLOW: '황색',
  ALL_RED: '전적색',
  IDLE: '대기',
}

export function statusLabel(value?: string | null): string {
  if (!value) return '준비'
  if (value.startsWith('BATCH ')) return value.replace('BATCH', '배치 실행')
  return statusLabels[value] ?? value
}

export function scenarioLabel(value?: string | null): string {
  return value ? scenarioLabels[value] ?? value : '정보 없음'
}

export function phaseLabel(value?: string | null): string {
  return value ? phaseLabels[value] ?? value : '시작 대기 중'
}

export function signalLabel(value?: string | null): string {
  return value ? signalLabels[value] ?? value : '대기'
}

export function experimentKindLabel(value: string): string {
  return value
    .replace('single', '단일')
    .replace('batch', '배치')
    .replace('initial', '초기 차량')
    .replace('continuous', '연속 교통')
    .replaceAll('_', ' · ')
}
