# Snake AI:DQN

**[English Version](./README.en.md)**

**Deep Q-Network, DQN**를 활용하여 Snake 게임을 학습시키는 강화학습 프로젝트입니다. 본 프로젝트는 픽셀 기반의 시각 정보를 학습하는 CNN 모델과 수작업으로 추출한 특징 벡터를 학습하는 선형 모델의 두 가지 구현을 통해 강화학습의 다양한 접근 방식을 탐구합니다.

**Demo** 
Episode 0:
<video src="https://github.com/user-attachments/assets/43e9ba4d-298b-4cd3-869f-234fd6c26b4d" width="600" controls></video>


Episode 350~:
<video src="https://github.com/user-attachments/assets/d66776c8-d032-4678-85fa-5898da5a6bcf" width="600" controls></video>



## 주요 특징

- **두 가지 아키텍처 지원**:
  - **CNN 기반 (SnakeDQN)**: 4프레임 스택 그레이스케일 입력(4×84×84)으로 이동 방향을 시간적으로 인식하며 학습합니다.
  - **선형 기반 (SnakeDQN_linear)**: 충돌, 방향, 먹이 위치 등 11차원 특징 벡터를 사용하여 약 10배 빠른 수렴 속도를 보여줍니다.
- **Double DQN**: 행동 선택(온라인 네트워크)과 가치 평가(타겟 네트워크)를 분리하여 Q-값 과대평가 문제를 해결했습니다.
- **안정적인 학습**: Huber Loss(SmoothL1), 그래디언트 클리핑, 지수 감쇠 Epsilon으로 학습 안정성을 확보했습니다.
- **균형 잡힌 보상 설계**: 사망/먹이/이동 보상을 재조정하여 조기 사망 유인 등 왜곡된 인센티브를 제거했습니다.
- **Experience Replay & Target Network**: 데이터 상관관계 차단 및 안정적인 Q-값 수렴을 위한 기법을 모두 구현했습니다.

## 기술 스택

- **딥러닝**: PyTorch 2.0+
- **게임 엔진**: Pygame
- **컴퓨터 비전**: OpenCV, NumPy
- **언어**: Python 3.8+
- **하드웨어**: GPU 가속(CUDA) 지원

## 프로젝트 구조

```text
├── SnakeDQN/           # CNN 기반 모델 (시각 정보 학습)
│   ├── config.py       # 하이퍼파라미터 중앙 설정
│   ├── dqn/            # 에이전트 및 CNN 모델 로직
│   └── game/           # 픽셀 기반 게임 환경
├── SnakeDQN_linear/    # 선형 네트워크 모델 (특징 벡터 학습)
│   ├── dqn/            # 에이전트 및 Dense 모델 로직
│   └── game/           # 벡터 기반 게임 환경
└── GpuCheck.py         # CUDA 가용성 확인 유틸리티
```

## 핵심 기술 구현 내용

### 1. Double DQN
기존 Vanilla DQN은 타겟 네트워크가 행동 선택과 가치 평가를 모두 담당해 Q-값이 체계적으로 과대 평가되는 문제가 있었습니다. **Double DQN**은 온라인 네트워크로 최적 행동을 선택하고, 타겟 네트워크로 그 가치를 평가하도록 분리하여 이 문제를 해결합니다. 두 버전 모두에 적용되었습니다.

### 2. 프레임 스택 (CNN 버전)
단일 프레임만으로는 뱀의 이동 방향을 알 수 없습니다. **4프레임 스택**을 통해 CNN이 시간적 맥락을 인식하고 방향을 추론할 수 있도록 개선했습니다. 상태 표현이 `(1, 84, 84)`에서 `(4, 84, 84)`로 변경되었습니다.

### 3. 학습 안정성 개선
- **Huber Loss (SmoothL1)**: 사망 페널티(-10) 같은 이상치에 MSE보다 강건합니다.
- **그래디언트 클리핑**: 역전파 시 폭발적 그래디언트를 방지합니다.
- **지수 감쇠 Epsilon**: 선형 버전의 `80 - n_games` 방식(80 에피소드 후 음수 발생) 문제를 `max(1, 80 × 0.995^n)` 지수 감쇠로 수정했습니다.

### 4. 보상 재설계
| 이벤트 | 이전 | 이후 | 이유 |
|--------|------|------|------|
| 사망 | -20 | -10 | 먹이 보상 대비 비율 재조정 |
| 먹이 획득 | +15 | +10 | 사망 패널티와 균형 맞춤 |
| 먹이 방향 접근 | +1 | +0.1 | 과도한 밀집 보상 완화 |
| 먹이 방향 후퇴 | 0 | -0.1 | 소극적 탐색 방지 |

## 빠른 시작

### 설치 방법
```bash
git clone <repository-url>
cd Snake_DQN
pip install -r requirements.txt
# GPU 가속 버전 (권장)
pip install torch --index-url https://download.pytorch.org/whl/cu128 (버전 확인)
```

### 모델 학습
```bash
# 빠른 수렴이 특징인 선형 모델 학습 시
cd SnakeDQN_linear
python main.py

# 시각 기반의 CNN 모델 학습 시
cd ../SnakeDQN
python main.py
```

> **주의 (CNN 모델)**: 4프레임 스택 도입으로 입력 채널이 1→4로 변경되었습니다. 기존 저장된 모델(`models/model_cnn_best.pth`)은 호환되지 않으므로 처음부터 재학습이 필요합니다.

## 예상 성능
- **선형 모델**: 약 300 에피소드 내 30~50점 도달.
- **CNN 모델**: 약 1000 에피소드 내 25~35점 도달 (4프레임 스택으로 장기 학습 시 성능 향상 기대).

> 상세한 DQN 하이퍼파라미터 튜닝 가이드, 모델별 보상 함수 및 상태 표현법은 [상세 매뉴얼(DETAILS.md)](./DETAILS.md)에서 확인하실 수 있습니다.

---
PyTorch & Pygame으로 구축한 프로젝트입니다.
