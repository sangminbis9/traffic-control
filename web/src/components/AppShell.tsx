import { BarChart3, FlaskConical, Gauge, MonitorPlay, ScrollText } from 'lucide-react'
import type { PageKey } from '../types'

const navigation: Array<{ key: PageKey; label: string; icon: typeof Gauge }> = [
  { key: 'dashboard', label: '대시보드', icon: Gauge },
  { key: 'training', label: '학습실', icon: FlaskConical },
  { key: 'comparison', label: '비교실', icon: BarChart3 },
  { key: 'reports', label: '보고서', icon: ScrollText },
  { key: 'presentation', label: '발표 모드', icon: MonitorPlay },
]

interface Props {
  page: PageKey
  onNavigate: (page: PageKey) => void
  children: React.ReactNode
}

export function AppShell({ page, onNavigate, children }: Props) {
  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="주 메뉴">
        <div className="brand">
          <span className="brand-signal" aria-hidden="true"><i /><i /><i /></span>
          <span><strong>AI 교통 신호 연구실</strong><small>강화학습 신호 제어 플랫폼</small></span>
        </div>
        <nav>
          {navigation.map((item) => {
            const Icon = item.icon
            return (
              <button key={item.key} className={page === item.key ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate(item.key)}>
                <Icon size={18} strokeWidth={1.8} />
                <span>{item.label}</span>
              </button>
            )
          })}
        </nav>
        <div className="sidebar-foot"><span className="online-dot" />SUMO · DQN 준비 완료</div>
      </aside>
      <main className="main-content">{children}</main>
    </div>
  )
}
