import { afterEach, expect, it, vi } from 'vitest'
import { api } from './api'

afterEach(() => vi.unstubAllGlobals())

it('shows a useful launcher instruction when the server is unreachable', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

  await expect(api('/api/training')).rejects.toThrow(
    '서버에 연결할 수 없습니다. start_traffic_control.bat을 다시 실행한 뒤 페이지를 새로고침하세요.',
  )
})
