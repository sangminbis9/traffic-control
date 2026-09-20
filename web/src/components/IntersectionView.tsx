import type { SimulationSide, VehicleFrame } from '../types'
import { phaseLabel, signalLabel } from '../i18n'

interface Props {
  title: string
  tone: 'fixed' | 'dqn'
  side?: SimulationSide
  bounds?: [[number, number], [number, number]]
}

const defaultBounds: [[number, number], [number, number]] = [[-250, -250], [250, 250]]
const roadStart = 34
const roadEnd = 66
const laneWidth = (roadEnd - roadStart) / 6
const laneDividers = [1, 2, 4, 5].map((lane) => roadStart + lane * laneWidth)
const laneCenters = Array.from({ length: 6 }, (_, lane) => roadStart + (lane + 0.5) * laneWidth)
const junctionStartRatio = 0.4704
const junctionEndRatio = 0.5296

function warpAxis(value: number) {
  const clamped = Math.min(1, Math.max(0, value))
  if (clamped < junctionStartRatio) return clamped / junctionStartRatio * roadStart
  if (clamped > junctionEndRatio) {
    return roadEnd + (clamped - junctionEndRatio) / (1 - junctionEndRatio) * (100 - roadEnd)
  }
  return roadStart + (clamped - junctionStartRatio) / (junctionEndRatio - junctionStartRatio) * (roadEnd - roadStart)
}

export function projectVehicleToDiagram(
  vehicle: Pick<VehicleFrame, 'x' | 'y' | 'lane'>,
  bounds: [[number, number], [number, number]],
) {
  const [[minX, minY], [maxX, maxY]] = bounds
  const normalizedX = (vehicle.x - minX) / Math.max(maxX - minX, 1)
  const normalizedY = (vehicle.y - minY) / Math.max(maxY - minY, 1)
  let x = warpAxis(normalizedX)
  let y = 100 - warpAxis(normalizedY)
  const laneMatch = /^([NSEW])_(in|out)_(\d+)$/.exec(vehicle.lane)
  if (!laneMatch) return { x, y }

  const [, direction, flow, laneText] = laneMatch
  const laneIndex = Math.min(2, Number(laneText))
  const centerByRoad: Record<string, number[]> = {
    N_in: [2, 1, 0], N_out: [5, 4, 3],
    S_in: [3, 4, 5], S_out: [0, 1, 2],
    E_in: [2, 1, 0], E_out: [5, 4, 3],
    W_in: [3, 4, 5], W_out: [0, 1, 2],
  }
  const laneCenter = laneCenters[centerByRoad[`${direction}_${flow}`][laneIndex]]
  if (direction === 'N' || direction === 'S') x = laneCenter
  else y = laneCenter
  return { x, y }
}

interface LaneArrowProps {
  x: number
  y: number
  rotation: number
  turn?: boolean
  outbound?: boolean
}

function LaneArrow({ x, y, rotation, turn = false, outbound = false }: LaneArrowProps) {
  return (
    <g className={`lane-direction ${outbound ? 'outbound' : 'inbound'}${turn ? ' turn' : ''}`} transform={`translate(${x} ${y}) rotate(${rotation})`}>
      {turn ? (
        <path d="M 0 3 V 0.2 C 0 -2 -1.2 -3 -3.4 -3 M -1.7 -4.6 L -3.4 -3 L -1.7 -1.4" />
      ) : (
        <path d="M 0 3 V -3 M -1.8 -1.2 L 0 -3 L 1.8 -1.2" />
      )}
    </g>
  )
}

export function IntersectionView({ title, tone, side, bounds = defaultBounds }: Props) {
  return (
    <section className={`intersection-panel ${tone}`}>
      <header>
        <div><span className="controller-mark" /> <strong>{title}</strong></div>
        <span>{signalLabel(side?.signal_state)}</span>
      </header>
      <div className="intersection-canvas">
        <svg viewBox="0 0 100 100" role="img" aria-label={`${title} 교차로 시뮬레이션. 각 방향 진입 3차로와 진출 3차로이며 진입 최좌측 차로는 좌회전 전용입니다.`}>
          <rect className="map-ground" width="100" height="100" />
          <rect className="map-road" x={roadStart} width={roadEnd - roadStart} height="100" />
          <rect className="map-road" y={roadStart} width="100" height={roadEnd - roadStart} />
          {laneDividers.map((value) => (
            <g key={`vertical-divider-${value}`}>
              <line x1={value} y1="0" x2={value} y2={roadStart} className="lane-line" />
              <line x1={value} y1={roadEnd} x2={value} y2="100" className="lane-line" />
            </g>
          ))}
          {laneDividers.map((value) => (
            <g key={`horizontal-divider-${value}`}>
              <line x1="0" y1={value} x2={roadStart} y2={value} className="lane-line" />
              <line x1={roadEnd} y1={value} x2="100" y2={value} className="lane-line" />
            </g>
          ))}
          {[49.55, 50.45].map((value) => (
            <g key={`center-line-${value}`}>
              <line x1={value} y1="0" x2={value} y2={roadStart} className="road-center-line" />
              <line x1={value} y1={roadEnd} x2={value} y2="100" className="road-center-line" />
              <line x1="0" y1={value} x2={roadStart} y2={value} className="road-center-line" />
              <line x1={roadEnd} y1={value} x2="100" y2={value} className="road-center-line" />
            </g>
          ))}
          <rect className="map-center" x={roadStart} y={roadStart} width={roadEnd - roadStart} height={roadEnd - roadStart} />

          {/* North and south: inbound arrows include a dedicated driver-left turn lane. */}
          <LaneArrow x={laneCenters[2]} y={19} rotation={180} turn />
          <LaneArrow x={laneCenters[1]} y={19} rotation={180} />
          <LaneArrow x={laneCenters[0]} y={19} rotation={180} />
          <LaneArrow x={laneCenters[3]} y={19} rotation={0} outbound />
          <LaneArrow x={laneCenters[4]} y={19} rotation={0} outbound />
          <LaneArrow x={laneCenters[5]} y={19} rotation={0} outbound />
          <LaneArrow x={laneCenters[3]} y={81} rotation={0} turn />
          <LaneArrow x={laneCenters[4]} y={81} rotation={0} />
          <LaneArrow x={laneCenters[5]} y={81} rotation={0} />
          <LaneArrow x={laneCenters[0]} y={81} rotation={180} outbound />
          <LaneArrow x={laneCenters[1]} y={81} rotation={180} outbound />
          <LaneArrow x={laneCenters[2]} y={81} rotation={180} outbound />

          {/* West and east follow the same right-hand traffic layout. */}
          <LaneArrow x={19} y={laneCenters[3]} rotation={90} turn />
          <LaneArrow x={19} y={laneCenters[4]} rotation={90} />
          <LaneArrow x={19} y={laneCenters[5]} rotation={90} />
          <LaneArrow x={19} y={laneCenters[0]} rotation={-90} outbound />
          <LaneArrow x={19} y={laneCenters[1]} rotation={-90} outbound />
          <LaneArrow x={19} y={laneCenters[2]} rotation={-90} outbound />
          <LaneArrow x={81} y={laneCenters[2]} rotation={-90} turn />
          <LaneArrow x={81} y={laneCenters[1]} rotation={-90} />
          <LaneArrow x={81} y={laneCenters[0]} rotation={-90} />
          <LaneArrow x={81} y={laneCenters[3]} rotation={90} outbound />
          <LaneArrow x={81} y={laneCenters[4]} rotation={90} outbound />
          <LaneArrow x={81} y={laneCenters[5]} rotation={90} outbound />
          {side?.vehicles.map((vehicle) => {
            const point = projectVehicleToDiagram(vehicle, bounds)
            return (
              <rect
                key={vehicle.id}
                data-lane={vehicle.lane}
                x={point.x - 1.25}
                y={point.y - 0.7}
                width="2.5"
                height="1.4"
                rx="0.35"
                className={vehicle.waiting_time > 0 ? 'vehicle waiting' : 'vehicle'}
                transform={`rotate(${vehicle.angle - 90} ${point.x} ${point.y})`}
              />
            )
          })}
        </svg>
        <div className="lane-legend" aria-hidden="true"><span><i className="inbound" />진입 3차로</span><span><i className="outbound" />진출 3차로</span><span><i className="left" />최좌측 좌회전 전용</span></div>
        <div className="phase-overlay">
          <small>현재 신호 단계</small>
          <strong>{phaseLabel(side?.phase_name)}</strong>
          <span>{(side?.phase_elapsed ?? 0).toFixed(1)}초</span>
        </div>
      </div>
      <div className="intersection-metrics">
        <div><span>남은 차량</span><strong>{side?.metrics.vehicles_remaining ?? 0}</strong></div>
        <div><span>대기 행렬</span><strong>{side?.metrics.current_queue.toFixed(0) ?? 0}</strong></div>
        <div><span>평균 대기</span><strong>{side?.metrics.average_waiting.toFixed(1) ?? '0.0'}초</strong></div>
        <div><span>통과 차량</span><strong>{side?.metrics.throughput ?? 0}</strong></div>
      </div>
    </section>
  )
}
