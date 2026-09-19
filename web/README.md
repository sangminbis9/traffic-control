# Web

반응형 교통 신호 제어 대시보드를 위한 독립 영역입니다. 아직 프론트엔드 프레임워크가 결정되지 않아 실행 코드는 생성하지 않았습니다.

예정 역할:

- 현재 교통량, queue, waiting time, signal phase 시각화
- Fixed-Time과 DQN 평가 결과 비교
- 학습·평가 실행 요청과 진행 상태 표시
- 데스크톱·태블릿·모바일 반응형 UI

권장 연동 경계:

```text
web UI
  -> HTTP/WebSocket API (추후 추가)
  -> model의 TrafficStateProvider / controller / evaluation 결과
```

웹 프레임워크를 선택할 때 이 폴더 안에 앱을 초기화하면 됩니다. 예를 들어 Next.js를 선택하면 `web/package.json`, `web/src/`, `web/public/`이 이 위치에 생성됩니다.
