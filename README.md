# Traffic Control

강화학습 기반 교통 신호 제어 시스템을 위한 모노레포입니다. 현재 완성된 SUMO·DQN 코드는 `model/`에 있으며, 향후 반응형 웹 애플리케이션은 `web/`에서 독립적으로 개발합니다.

## Repository structure

```text
traffic-control/
├── model/                 # SUMO, TraCI, Gymnasium, DQN 학습·평가
│   ├── controller/        # 신호 전환 및 Fixed-Time 제어
│   ├── env/               # Gymnasium 환경, state, reward
│   ├── traffic/           # 재현 가능한 교통 수요 생성
│   ├── sumo/              # 교차로 네트워크와 SUMO 설정
│   ├── tests/             # 독립 단위 테스트
│   ├── results/           # 학습 모델, CSV, 그래프
│   ├── train.py
│   ├── evaluate.py
│   └── test_sumo.py
├── web/                   # 향후 반응형 웹 UI/API 연동 영역
└── README.md
```

## Model quick start

모든 명령은 저장소 루트에서 실행합니다.

```powershell
python -m pip install -r model\requirements.txt
python -m model.sumo.build_network
python -m model.test_sumo --seconds 60
python -m model.train --timesteps 1000 --episode-seconds 120 --check-env
python -m model.evaluate --episodes 30 --seed-start 2001
```

상세한 상태·액션·보상 설계와 실행 방법은 [model/README.md](model/README.md)를 참고하세요. 웹 영역의 예정된 역할은 [web/README.md](web/README.md)에 정리되어 있습니다.

## Integration direction

웹은 학습 프로세스를 직접 import하기보다 추후 추가할 API 계층을 통해 모델과 연결합니다. API는 교통 상태, 현재 phase, controller 종류, 평가 metric을 JSON으로 제공하고, 웹은 실시간 대시보드와 학습 결과 비교를 담당하는 구조를 권장합니다.
