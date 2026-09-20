import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { IntersectionView, projectVehicleToDiagram } from './components/IntersectionView'
import { FixedBaselinePanel } from './components/FixedBaselinePanel'

const compatibleModel = {
  name: 'dqn_intersection',
  path: 'C:/models/dqn_intersection.zip',
  relative_path: 'model/results/dqn_intersection.zip',
  sha256: '0123456789abcdef',
  timesteps: 1000,
  observation_shape: [12],
  action_count: 4,
  compatible: true,
  error: null,
}

describe('Traffic Control Lab shell', () => {
  beforeEach(() => {
    window.location.hash = '#/dashboard'
    window.localStorage.clear()
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      const body = url.includes('/api/models') ? [compatibleModel] : []
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } }))
    }))
  })

  afterEach(() => vi.unstubAllGlobals())

  it('loads real model metadata and navigates to comparison controls', async () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: '대시보드' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('dqn_intersection')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: '비교실' }))
    expect(screen.getByRole('heading', { name: '제어기 비교실' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '비교 실행' })).toBeEnabled())
    expect(screen.getByText('초기 차량')).toBeInTheDocument()
  })

  it('renders three inbound and three outbound lanes on every approach', () => {
    const { container } = render(<IntersectionView title="시험 교차로" tone="dqn" />)

    expect(container.querySelectorAll('.lane-line')).toHaveLength(16)
    expect(container.querySelectorAll('.lane-direction.inbound')).toHaveLength(12)
    expect(container.querySelectorAll('.lane-direction.outbound')).toHaveLength(12)
    expect(container.querySelectorAll('.lane-direction.inbound.turn')).toHaveLength(4)
    const legend = container.querySelector('.lane-legend')
    expect(legend).toHaveTextContent('진입 3차로')
    expect(legend).toHaveTextContent('진출 3차로')
    expect(legend).toHaveTextContent('최좌측 좌회전 전용')
  })

  it('projects SUMO vehicles onto the rendered lane centers', () => {
    const bounds: [[number, number], [number, number]] = [[0, 0], [1000, 1000]]
    const northLeft = projectVehicleToDiagram({ x: 492, y: 800, lane: 'N_in_0' }, bounds)
    const southLeft = projectVehicleToDiagram({ x: 508, y: 200, lane: 'S_in_0' }, bounds)
    const eastLeft = projectVehicleToDiagram({ x: 800, y: 508, lane: 'E_in_0' }, bounds)
    const westLeft = projectVehicleToDiagram({ x: 200, y: 492, lane: 'W_in_0' }, bounds)

    expect(northLeft.x).toBeCloseTo(47.333, 2)
    expect(southLeft.x).toBeCloseTo(52.667, 2)
    expect(eastLeft.y).toBeCloseTo(47.333, 2)
    expect(westLeft.y).toBeCloseTo(52.667, 2)
    expect(northLeft.y).toBeLessThan(34)
    expect(westLeft.x).toBeLessThan(34)
  })

  it('judges a DQN result against the fixed-time acceptance thresholds', () => {
    render(<FixedBaselinePanel validation={{
      timestep: 20_000,
      avg_waiting_time: 18, max_waiting_time: 70, avg_queue: 9, max_queue: 15,
      throughput: 98, phase_changes: 14, validation_reward: -10,
      fixed_avg_waiting_time: 20, fixed_max_waiting_time: 80, fixed_avg_queue: 10,
      fixed_max_queue: 18, fixed_throughput: 100, fixed_phase_changes: 12,
      waiting_improvement_pct: 10, max_waiting_improvement_pct: 12.5,
      queue_improvement_pct: 10, throughput_change_pct: -2, throughput_retention_pct: 98,
      waiting_target_met: true, max_waiting_target_met: true, queue_target_met: true,
      throughput_target_met: true, beats_fixed: true, score: 35.5, eligible: true,
    }} validationEpisodes={5} />)

    expect(screen.getByText('목표 달성')).toBeInTheDocument()
    expect(screen.getByText('4 / 4 조건 통과')).toBeInTheDocument()
    expect(screen.getAllByText('10.0% 감소')).toHaveLength(2)
    expect(screen.getByText('2.0% 감소')).toBeInTheDocument()
  })
})
