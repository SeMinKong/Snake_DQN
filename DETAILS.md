# Snake AI: DQN 아키텍처 및 설정 상세

## 1. 하이퍼파라미터 (config.py)
**AgentConfig**
- `max_memory = 100,000`: 경험 재생(Experience Replay) 큐의 최대 크기.
- `batch_size = 256`: 매 훈련 스텝마다 메모리에서 샘플링하는 미니배치 크기.
- `lr = 0.0001`: 신경망 가중치 학습률.
- `epsilon_start = 1.0` / `epsilon_min = 0.05` / `epsilon_decay = 0.999`: 에피소드가 끝날 때마다 탐색 확률(Epsilon)을 점진적으로 줄입니다.
- `gamma = 0.99`: 할인 인수(Discount Factor). 미래 보상의 중요도를 설정합니다.
- `sync_target_frames = 1000`: Target Network의 가중치를 메인 Network와 동기화하는 주기.

## 2. 상태 표현(State Representation) 차이
**CNN (SnakeDQN)**
- Pygame의 210x210 RGB 화면을 84x84 그레이스케일로 변환 후 0~1로 정규화하여 `(1, 1, 84, 84)` 형태의 텐서로 입력합니다.

**선형 네트워크 (SnakeDQN_linear)**
- 11차원 바이너리 벡터를 사용합니다.
- `0~2`: 전방/좌측/우측 충돌 여부 (0/1)
- `3~6`: 현재 이동 방향 (상/하/좌/우 One-hot 인코딩)
- `7~10`: 뱀 머리 기준 먹이의 상대적 방향 (상/하/좌/우)

## 3. 보상 함수 (Reward Function)
- **CNN 기반 보상**: 먹이 획득(+15.0 + 뱀 길이x0.2), 충돌(-20.0), 먹이로 접근(+1.0), 멀어짐(-1.5), 시간 지연 페널티(-0.05/스텝)
- **Linear 모델 보상**: 먹이 획득(+15), 자가 충돌 및 벽 충돌(-15), 먹이로 접근(+1), 생존 보너스(+0.01)
