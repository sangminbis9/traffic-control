# 강화학습 교통 신호 제어 캡스톤

실행 프로젝트는 [traffic-rl](traffic-rl)에 있습니다.

학습 시드 3개 × 각 15만 스텝 DQN을 고정 신호와 별도 교통 시드 30개·6시나리오에서 비교했습니다. 세 모델 평균으로 대기시간 **31.1% 감소**, Queue **22.0% 감소**, 처리량 **6.7% 증가**했습니다. 개별 시나리오의 예외와 긴급 제동 경고 등 한계는 보고서에 기록했습니다.

- [최종 실험 결과·신뢰구간·전체 시도](traffic-rl/EXPERIMENTS.md)
- [설치·학습·평가·GUI 실행 설명](traffic-rl/README.md)
- [선정 모델](traffic-rl/results/models/dqn.zip) · [설정](traffic-rl/config.json)
- [전체 52개 checkpoint 비교](traffic-rl/results/experiment_leaderboard.csv)
- [초기 개발 검증 기록](traffic-rl/VALIDATION.md)
