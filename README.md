# Snake AI:DQN

**[English Version](./README.en.md)**

**(Deep Q-Network, DQN)**를 활용하여 Snake 게임을 학습시키는 강화학습 프로젝트입니다. 본 프로젝트는 픽셀 기반의 시각 정보를 학습하는 CNN 모델과 수작업으로 추출한 특징 벡터를 학습하는 선형 모델의 두 가지 구현을 통해 강화학습의 다양한 접근 방식을 탐구합니다.

## 주요 특징

- **두 가지 아키텍처 지원**:
  - **CNN 기반 (SnakeDQN)**: 84x84 그레이스케일 프레임을 직접 입력받아 사람의 시각과 유사한 방식으로 학습을 진행합니다.
  - **선형 기반 (SnakeDQN_linear)**: 충돌, 방향, 먹이 위치 등 11차원 특징 벡터를 사용하여 약 10배 빠른 수렴 속도를 보여줍니다.
- **고급 강화학습 기법**: 데이터 간의 상관관계(Correlation)를 끊기 위한 **Experience Replay**와 안정적인 학습을 위한 **Target Network**를 구현했습니다.
- **Epsilon-Greedy 전략**: 초기 1.0(무작위 탐색)에서 0.05(최적 행동)까지 점진적으로 감소하며 탐험과 활용의 균형을 맞춥니다.
- **커스텀 보상 설계**: BFS 기반의 경로 탐색 보상 등 복잡한 상황을 해결하기 위한 정교한 보상 체계를 적용했습니다.

## 기술 스택

- **딥러닝**: PyTorch 2.0+
- **게임 엔진**: Pygame
- **컴퓨터 비전**: OpenCV, NumPy
- **언어**: Python 3.8+
- **하드웨어**: GPU 가속(CUDA) 지원

## 프로젝트 구조

```text
├── SnakeDQN/           # CNN 기반 모델 (시각 정보 학습)
│   ├── dqn/            # 에이전트 및 CNN 모델 로직
│   └── game/           # 픽셀 기반 게임 환경
├── SnakeDQN_linear/    # 선형 네트워크 모델 (특징 벡터 학습)
│   ├── dqn/            # 에이전트 및 Dense 모델 로직
│   └── game/           # 벡터 기반 게임 환경
└── GpuCheck.py         # CUDA 가용성 확인 유틸리티
```

## 핵심 기술 구현 내용

### 1. CNN vs 선형 모델의 성능 분석
두 모델을 모두 구현하여 성능 차이를 분석했습니다. **선형 모델**은 효율적인 특징 추출로 인해 약 300 에피소드 이내에 고득점에 도달하지만, **CNN 모델**은 픽셀로부터 직접 공간 패턴을 학습함으로써 더 높은 일반화 가능성을 보여주었습니다.

### 2. 강화학습의 안정성 확보
DQN 학습 시 발생하는 불안정성 문제를 해결하기 위해 **Target Network**를 도입하고, 1,000 스텝마다 동기화하도록 설계했습니다. 이를 통해 Q-값이 요동치는 현상을 방지하고 안정적인 학습 곡선을 도출했습니다.

## 빠른 시작

### 설치 방법
```bash
git clone <repository-url>
cd Snake_DQN
pip install -r requirements.txt
# GPU 가속 버전 (권장)
pip install torch --index-url https://download.pytorch.org/whl/cu121
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

## 예상 성능
- **선형 모델**: 약 300 에피소드 내 30~50점 도달.
- **CNN 모델**: 약 1000 에피소드 내 25~35점 도달.

> 상세한 DQN 하이퍼파라미터 튜닝 가이드, 모델별 보상 함수 및 상태 표현법은 [상세 매뉴얼(DETAILS.md)](./DETAILS.md)에서 확인하실 수 있습니다.

---
PyTorch & Pygame으로 구축한 프로젝트입니다.
