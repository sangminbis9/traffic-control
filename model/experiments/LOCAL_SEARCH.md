# 중간발표용 국소 탐색

전체 staged sensitivity search 대신 사용자가 지정한 여섯 후보만 비교합니다.
기존 모델/웹 기본 설정과 CLI는 수정하지 않습니다.

```powershell
python -m model.experiments.local_reward_search --workers 5 --output model/results/reward_sensitivity_local
```

후보: (0.25,0.25,0.40), (0.30,0.30,0.50), (0.35,0.35,0.60),
(0.40,0.40,0.70), (0.30,0.30,0.60), (0.35,0.35,0.50).

두 번째 후보의 실제 50,000-step 결과를 재사용합니다. 나머지 다섯 후보는
각각 새 모델과 빈 replay buffer로 독립 학습합니다. Queue 계수 1.0,
학습 seed 22, random 교통, 회차 300초, 수요 240초를 유지합니다.
기존 결과와 동일하게 150,000-step 탐색률 일정으로 학습하되 정확히
50,000 step에서 종료합니다. 평가 seed 2001~2030은 deterministic=True로
모델마다 30회 평가합니다. 신규 모델 간 경험이나 모델을 공유하지 않습니다.

기존 결과의 모델 해시, 실제 step/탐색 일정, DQN 설정, Reward 정규화,
신호·교통 설정, Python/SUMO/Git 버전, 네트워크와 캐시 코드 해시를 확인합니다.
기존 결과와 신규 결과의 학습·평가 수요 seed/파일 SHA256이 다르면
전체 비교를 성공 처리하지 않습니다.

사전 선정 규칙: 충돌 또는 텔레포트가 있거나 기존 기준 처리량의 95%보다
작으면 제외합니다. 남은 후보에서 평균 대기 → 최대 대기 → 평균 대기열 →
전환 횟수를 순서대로 최소화합니다. 정확한 동률에 한해 처리량과 후보 ID로
순서를 결정합니다. episode_reward는 후보 선정에 사용하지 않습니다.
평균과 표본 표준편차는 교통 seed 변동만 나타냅니다. bootstrap, 유의성 검정,
다중 학습 seed 검증은 수행하지 않습니다.

생성 파일: search_config.json, comparison.csv/json/md,
changes_vs_baseline.csv, recommendation.json, report-ko.md, status.json.
각 experiment 폴더에 final.zip, 신규 후보의 final.replay.pkl,
evaluation_metrics.csv, training_progress.csv, 실험 설정과 수요 해시가 저장됩니다.
기존 후보는 원래 결과만 재사용하므로 새로운 replay buffer가 생성되지 않습니다.
실패한 실행은 progress.json과 failures.json에 기록하며 임의로 제외하지 않습니다.

OneDrive 등 동기화 폴더에서 파일 잠금이 발생할 수 있으므로 `--output`을
동기화되지 않는 로컬 폴더로 지정할 수 있습니다. 작업 완료 후 결과를
프로젝트 폴더에 복사해도 모델 SHA256과 재현 정보는 보존됩니다.

결과 보고서만 다시 생성하려면:

```powershell
python -m model.experiments.local_reward_search --output model/results/reward_sensitivity_local --summarize-only
```
