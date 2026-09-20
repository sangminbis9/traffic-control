const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch (cause) {
    throw new Error('서버에 연결할 수 없습니다. start_traffic_control.bat을 다시 실행한 뒤 페이지를 새로고침하세요.', { cause })
  }
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed: ${response.status}`)
  }
  return response.json() as Promise<T>
}

export function websocketUrl(path: string): string {
  const configured = import.meta.env.VITE_WS_BASE
  if (configured) return `${configured}${path}`
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}
