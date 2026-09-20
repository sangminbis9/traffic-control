export type PageKey = 'dashboard' | 'training' | 'comparison' | 'reports' | 'presentation'

export interface ModelMetadata {
  name: string
  path: string
  relative_path: string
  sha256: string
  size_bytes: number
  created_at: string
  timesteps: number
  observation_shape: number[] | null
  action_count: number | null
  compatible: boolean
  error: string | null
  training_scenario?: string | null
  last_validation_score?: number | null
}

export interface VehicleFrame {
  id: string
  x: number
  y: number
  angle: number
  speed: number
  waiting_time: number
  lane: string
  route: string[]
}

export interface LiveMetrics {
  vehicles_remaining: number
  current_queue: number
  average_waiting: number
  maximum_waiting: number
  throughput: number
  phase_changes: number
  clearance_percent: number
  approach_queues: Record<string, number>
}

export interface SimulationSide {
  phase: number
  phase_name: string
  phase_elapsed: number
  signal_state: string
  vehicles: VehicleFrame[]
  metrics: LiveMetrics
}

export interface SimulationFrame {
  type: 'simulation_frame'
  experiment_id: string
  status: string
  simulation_time: number
  network_bounds: [[number, number], [number, number]]
  fixed: SimulationSide
  dqn: SimulationSide
  delta: Record<string, number | null>
}

export interface TrainingProgress {
  type?: string
  status: string
  timesteps: number
  total_timesteps: number
  episodes: number
  epsilon: number
  replay_buffer_size: number
  rolling_episode_reward: number
  elapsed_seconds: number
  best_validation?: TrainingValidation | null
  validation?: TrainingValidation
  fixed_baseline?: ValidationMetrics | null
  stop_reason?: string | null
  current_checkpoint?: string | null
  checkpoints?: Array<{ timestep: number; name: string; model_path: string; replay_buffer_path: string }>
}

export interface ValidationMetrics {
  avg_waiting_time: number
  max_waiting_time: number
  avg_queue: number
  max_queue: number
  throughput: number
  phase_changes: number
  episode_reward?: number
}

export interface TrainingValidation extends ValidationMetrics {
  timestep: number
  validation_reward: number
  fixed_avg_waiting_time: number
  fixed_max_waiting_time: number
  fixed_avg_queue: number
  fixed_max_queue: number
  fixed_throughput: number
  fixed_phase_changes: number
  waiting_improvement_pct: number
  max_waiting_improvement_pct: number
  queue_improvement_pct: number
  throughput_change_pct: number
  throughput_retention_pct: number
  waiting_target_met: boolean
  max_waiting_target_met: boolean
  queue_target_met: boolean
  throughput_target_met: boolean
  beats_fixed: boolean
  score: number
  eligible: boolean
}

export interface TrainingFrame {
  type: 'training_frame'
  session_id: string
  status: string
  timestep: number
  episode: number
  scenario: string
  simulation_time: number
  network_bounds: [[number, number], [number, number]]
  side: SimulationSide
}

export interface TrainingRecord {
  id: string
  status: string
  created_at: string
  updated_at: string
  config: Record<string, unknown>
  detail: TrainingProgress
  artifact_dir: string
}

export interface ExperimentRecord {
  id: string
  kind: string
  status: string
  created_at: string
  config: Record<string, unknown>
  result: Record<string, unknown>
  artifact_dir: string
}

export interface LanePlacement {
  left: number
  straight: number
  lane3_total: number
  lane3_right: number
}
