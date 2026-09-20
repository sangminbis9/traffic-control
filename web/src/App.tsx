import { useCallback, useEffect, useState } from 'react'
import { AppShell } from './components/AppShell'
import { ComparisonLab } from './pages/ComparisonLab'
import { Dashboard } from './pages/Dashboard'
import { PresentationMode } from './pages/PresentationMode'
import { Reports } from './pages/Reports'
import { TrainingLab } from './pages/TrainingLab'
import type { PageKey, SimulationFrame } from './types'

const validPages = new Set<PageKey>(['dashboard', 'training', 'comparison', 'reports', 'presentation'])

function pageFromHash(): PageKey {
  const value = window.location.hash.replace('#/', '') as PageKey
  return validPages.has(value) ? value : 'dashboard'
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash)
  const [frame, setFrame] = useState<SimulationFrame | null>(null)
  const [frameHistory, setFrameHistory] = useState<SimulationFrame[]>([])
  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const navigate = useCallback((next: PageKey) => {
    window.location.hash = `/${next}`
    setPage(next)
  }, [])
  const receiveFrame = useCallback((next: SimulationFrame) => {
    setFrame(next)
    setFrameHistory((current) => [...current.slice(-599), next])
  }, [])
  if (page === 'presentation') return <PresentationMode frame={frame} history={frameHistory} onNavigate={navigate} />
  return (
    <AppShell page={page} onNavigate={navigate}>
      {page === 'dashboard' ? <Dashboard onNavigate={navigate} /> : null}
      {page === 'training' ? <TrainingLab onNavigate={navigate} /> : null}
      {page === 'comparison' ? <ComparisonLab onNavigate={navigate} onFrame={receiveFrame} /> : null}
      {page === 'reports' ? <Reports /> : null}
    </AppShell>
  )
}
