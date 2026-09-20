# Web dashboard

React, TypeScript, Vite, Recharts, SVG로 구현한 반응형 연구·시연 대시보드입니다. 데스크톱은 좌측 navigation과 16:9 비교 화면, tablet은 축소 sidebar, mobile은 하단 navigation과 단일 열 layout을 사용합니다.

## Pages

- 대시보드: 모델 호환성·SHA·학습 정보, 최근 학습/비교, 빠른 실행
- 학습실: 고정 스텝/자동 수렴 설정, 시작·일시정지·재개·중지, 실시간 SUMO 교차로, 검증 및 체크포인트 기록
  - 동일한 검증 시드로 Fixed-Time 기준을 한 번 계산한 뒤 DQN과 직접 비교
  - 1차 우수 판정: 평균 대기시간 10% 이상 감소, 평균 대기행렬 5% 이상 감소,
    최대 대기시간이 Fixed-Time 이하이면서 120초 이하, 통과량 95% 이상 유지
  - 최종 보고서에는 비교실에서 학습에 사용하지 않은 시드 30회 이상의 평균과 표준편차 사용 권장
- 비교실: 12개 차로 초기 배치 또는 연속 교통량, 단일/반복 실행, 모델·시드·속도 선택, 실제 SUMO 양측 교차로와 지표·차트·신호 단계 타임라인
- 보고서: 실험 기록과 발표 자료 ZIP 생성
- 발표 모드: 설정 화면을 숨긴 전체 화면 비교, 재생 제어, 핵심 지표, 남은 차량 차트

## Development

API를 먼저 `python -m api.app.main`으로 실행한 뒤:

```powershell
cd web
npm install
npm run dev
```

Vite 개발 서버의 `/api`와 `/ws` proxy는 기본적으로 `127.0.0.1:8000`을 사용합니다. 별도 배포에서는 `VITE_API_BASE`와 `VITE_WS_BASE`를 설정할 수 있습니다.

## Production build and tests

```powershell
npm test
npm run build
npm run preview
```

빌드 결과는 `web/dist/`에 생성됩니다. 현재 Recharts를 포함한 단일 JS chunk가 약 625 KB이므로 실제 인터넷 배포 전에는 page-level lazy loading을 추가할 수 있습니다.

## Data contract

`src/types.ts`는 backend payload와 대응합니다. `SimulationFrame`의 차량과 metric은 모두 SUMO/TraCI 원본이며 frontend가 교통 결과를 계산하거나 만들어내지 않습니다. SVG 좌표 변환은 frame의 `network_bounds`를 기준으로 Fixed/DQN에 동일하게 적용됩니다.

디자인 기준 이미지는 `web/design/dashboard-concept.png`, `web/design/comparison-concept.png`에 보관합니다.
