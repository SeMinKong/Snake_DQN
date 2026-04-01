# Snake DQN — Deep Q-Network for Snake Game

**Table of Contents**
- [한국어 문서 (Korean Documentation)](#한국어-문서)
- [English Documentation](#english-documentation)

---

# 한국어 문서

## 프로젝트 개요

**Snake_DQN**은 심층 Q-네트워크(Deep Q-Network, DQN)를 사용하여 Snake 게임을 학습하는 심화 강화학습 프로젝트입니다. 본 프로젝트는 두 가지 구현을 제공합니다:

- **SnakeDQN (CNN 기반)**: 픽셀 기반 관찰을 CNN으로 처리하는 엔드-투-엔드 학습 방식
- **SnakeDQN_linear (선형 네트워크 기반)**: 손으로 만든 특성(벡터)을 선형 네트워크로 처리하는 방식

이 프로젝트는 다음을 학습하는 데 최적화되어 있습니다:
- Deep Q-Learning 알고리즘의 실제 구현
- Experience Replay와 Target Network의 중요성
- Epsilon-Greedy 탐색 전략
- 신경망 아키텍처 선택이 학습 성능에 미치는 영향

## 기술 스택

| 카테고리 | 기술 |
|--------|------|
| **프레임워크** | PyTorch 2.0+ |
| **게임 엔진** | Pygame |
| **이미지 처리** | OpenCV (cv2), NumPy |
| **Python 버전** | 3.8+ |
| **학습 지원** | CUDA (선택사항, GPU 가속) |

## 주요 기능

### SnakeDQN (CNN 기반)
- **픽셀 기반 관찰**: 84×84 그레이스케일 프레임을 입력으로 사용
- **3층 CNN**: 시각적 특성 추출을 위한 합성곱 신경망
- **Experience Replay**: 경험 메모리(100,000개 전환)에서 무작위 샘플링
- **Target Network**: 안정적인 학습을 위한 고정된 목표 신경망 (1,000 스텝마다 동기화)
- **Epsilon-Greedy 탐색**: 1.0에서 시작해 0.05로 감소 (에피소드당 0.999배)

### SnakeDQN_linear (선형 기반)
- **손으로 만든 특성**: 11차원 벡터 상태 표현
  - 3개: 직진/좌측/우측 충돌 감지
  - 4개: 현재 이동 방향
  - 4개: 먹이 방향 상대정보
- **선형 신경망**: 입력층(11) → 숨은층(256) → 출력층(3)
- **BFS 기반 보상**: 접근 가능한 영역과 경로 거리를 고려한 보상
- **빠른 학습**: CNN 기반보다 훨씬 빠른 수렴

## 사용자 가이드

### 설치

#### 필수 조건
- Python 3.8 이상
- pip (Python 패키지 관리자)
- CUDA 12.1 (선택사항, GPU 가속용)

#### 1단계: 저장소 복제

```bash
git clone <repository_url>
cd Snake_DQN
```

#### 2단계: 가상 환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

#### 3단계: 의존성 설치

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### GPU 확인

```bash
python GpuCheck.py
```

출력: `True` (CUDA 가능) 또는 `False` (CPU 모드)

### 훈련 시작

#### CNN 모델

```bash
cd SnakeDQN
python main.py
```

#### 선형 모델

```bash
cd SnakeDQN_linear
python main.py
```

### 훈련 지표

| 지표 | 의미 | 목표 |
|-----|------|------|
| **Score** | 현재 게임에서 획득한 먹이 | 증가하는 추세 |
| **Record** | 지금까지의 최고 점수 | 꾸준히 상향 |
| **Epsilon** | 탐색 확률 (1.0=무작위, 0.05=욕심쟁이) | 감소 추세 |

### 예상 성능

- **CNN 모델**: 500-1000 에피소드 후 25-35점
- **선형 모델**: 100-300 에피소드 후 25-40점

---

## 개발자 가이드

### 필수 구성요소

#### CUDA & PyTorch

```bash
# GPU 지원
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU만
pip install torch
```

### 프로젝트 구조

```
Snake_DQN/
├── README.md
├── GpuCheck.py
├── requirements.txt
├── SnakeDQN/
│   ├── main.py           # 훈련 진입점
│   ├── config.py         # 하이퍼파라미터
│   ├── dqn/
│   │   ├── agent.py      # DQN 에이전트
│   │   └── model.py      # CNN 신경망
│   └── game/
│       └── snake_game.py # 게임 환경
└── SnakeDQN_linear/
    ├── main.py
    ├── dqn/
    │   ├── agent.py
    │   └── model.py
    └── game/
        └── snake_game.py
```

### config.py 파라미터

#### AgentConfig

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| **max_memory** | 100,000 | 50k-500k | 경험 재생 메모리 크기 |
| **batch_size** | 256 | 16-512 | 미니배치 크기 |
| **lr** | 0.0001 | 0.00001-0.001 | 학습률 |
| **epsilon_start** | 1.0 | 고정 | 초기 탐색률 |
| **epsilon_min** | 0.05 | 0.01-0.1 | 최소 탐색률 |
| **epsilon_decay** | 0.999 | 0.9995-0.9999 | 에피소드당 감소율 |
| **gamma** | 0.99 | 0.9-0.99 | 할인 인수 |
| **sync_target_frames** | 1000 | 500-2000 | Target Network 동기화 간격 |

#### GameConfig

| 파라미터 | 기본값 | 범위 |
|---------|-------|------|
| **grid_w / grid_h** | 10 / 10 | 8-15 |
| **speed** | 100 | 50-200 |
| **frame_h / frame_w** | 84 / 84 | - |

#### ModelConfig

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| **output_size** | 3 | 액션 수 (고정) |
| **input_channels** | 1 | 입력 채널 (고정) |
| **save_filename** | model_cnn_best.pth | 모델 저장 이름 |
| **model_dir** | ./models | 저장 디렉토리 |

### DQN 알고리즘 상세

#### Experience Replay

```python
def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

def train_replay(self):
    if len(self.memory) < self.cfg.batch_size:
        return
    mini_sample = random.sample(self.memory, self.cfg.batch_size)
    # 미니배치로 훈련
```

**효과**: 100,000개 메모리에서 256개씩 샘플링 → 안정적 학습

#### Target Network

```python
# 1,000 훈련 스텝마다 동기화
if self.learn_step_counter % self.cfg.sync_target_frames == 0:
    self.target_model.load_state_dict(self.model.state_dict())
```

**효과**: 안정적인 Q-값 목표 → 빠른 수렴

#### Epsilon-Greedy

```python
if random.random() < self.epsilon:
    move = random.randint(0, 2)  # 탐색
else:
    prediction = self.model(state_t)
    move = torch.argmax(prediction).item()  # 활용

# 매 에피소드마다 감소
self.epsilon = max(self.cfg.epsilon_min, self.epsilon * 0.999)
```

**타임라인**:
- Game 1: 100% 탐색
- Game 100: ~90.5% 탐색
- Game 500: ~60.6% 탐색
- Game 1000: ~36.8% 탐색

### CNN vs 선형 모델

#### CNN 모델

**아키텍처**: Conv(1→32, 8×8) → Conv(32→64, 4×4) → Conv(64→64, 3×3) → FC(3136→512) → FC(512→3)

**장점**:
- 엔드-투-엔드 학습
- 복잡한 패턴 학습
- 일반화 가능

**단점**:
- 느린 훈련
- 더 많은 데이터 필요

**성능**: 25-35점 (500-1000 에피소드)

#### 선형 모델

**상태**: 11차원 벡터 (충돌 3 + 방향 4 + 먹이 위치 4)

**아키텍처**: Linear(11→256) → Linear(256→3)

**장점**:
- 10배 빠른 훈련
- 해석 가능
- CPU 친화적

**단점**:
- 특성 공학 필요
- 확장성 낮음

**성능**: 30-50점 (100-300 에피소드)

#### 선택 기준

| 상황 | 추천 | 이유 |
|------|------|------|
| 학습 | 선형 | 빠른 반복 |
| 성능 극대화 | CNN | 더 높은 점수 |
| GPU 없음 | 선형 | CPU 빠름 |
| 이해 | 선형 | 명확한 특성 |
| 일반화 | CNN | 다른 환경 적용 |

### 보상 함수

#### CNN 모델

```
먹이 획득:      +15.0 + 뱀길이×0.2
긴 뱀 보너스:   +5.0
먹이로 접근:    +1.0
멀어짐:         -1.5
시간 페널티:    -0.05/스텝
충돌:           -20.0
```

#### 선형 모델 (BFS 기반)

```
먹이 획득:      +15
스스로 갇힘:    -15
접근:           +1
생존 보너스:    +0.01
```

### 모델 교체 (DQNBase)

모든 모델은 `DQNBase`를 상속해야 합니다:

```python
class DQNBase(nn.Module):
    def forward(self, x):
        raise NotImplementedError
    
    def save(self, file_name=None, cfg=MODEL_CFG):
        pass

class MyDQN(DQNBase):
    def forward(self, x):
        # 구현
        return output

# Agent에 주입
agent = Agent(model=MyDQN(output_size=3, input_channels=1))
```

### 하이퍼파라미터 튜닝

**학습 부족**:
```python
lr = 0.001  # 증가
epsilon_decay = 0.995  # 감소
batch_size = 128  # 감소
```

**학습 불안정**:
```python
lr = 0.00005  # 감소
batch_size = 512  # 증가
sync_target_frames = 500  # 감소
```

**훈련 느림**:
```python
speed = 200  # 증가
grid_w, grid_h = 8, 8  # 감소
```

### 상태 표현

#### CNN (픽셀 기반)

```
Pygame 프레임 (210×210 RGB)
    ↓ [그레이스케일 + 리사이징]
84×84 그레이스케일 [0-255]
    ↓ [정규화]
(1, 1, 84, 84) float [0-1]
```

#### 선형 (특성 기반)

```
Index | 특성      | 값
------|----------|-----
0-2   | 충돌     | 0/1
3-6   | 방향     | 0/1 (원핫)
7-10  | 먹이위치 | 0/1
```

---

# English Documentation

## Project Overview

**Snake_DQN** is an advanced reinforcement learning project implementing Deep Q-Networks (DQN) to teach an agent the Snake game. Two implementations:

- **SnakeDQN (CNN-based)**: End-to-end learning from pixel observations
- **SnakeDQN_linear (Linear-based)**: Hand-crafted feature vectors with dense networks

Learn:
- Deep Q-Learning implementation
- Experience Replay and Target Networks
- Epsilon-Greedy exploration
- Architecture impact on performance

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Framework** | PyTorch 2.0+ |
| **Game Engine** | Pygame |
| **Image Processing** | OpenCV (cv2), NumPy |
| **Python Version** | 3.8+ |
| **Training Support** | CUDA (optional) |

## Key Features

### SnakeDQN (CNN-based)
- **Pixel-based**: 84×84 grayscale frames
- **3-layer CNN**: Feature extraction
- **100k Memory**: Experience replay buffer
- **Target Network**: Synced every 1,000 steps
- **Epsilon-Greedy**: 1.0→0.05 decay (×0.999/episode)

### SnakeDQN_linear (Linear-based)
- **11-dim Vector**: Hand-crafted state
  - 3 collision detection
  - 4 direction (one-hot)
  - 4 food location
- **Linear Network**: 11→256→3
- **BFS Rewards**: Accessibility-based
- **10× Faster**: Quick convergence

## User Guide

### Installation

#### Prerequisites
- Python 3.8+
- pip
- CUDA 12.1 (optional)

#### Step 1: Clone Repository

```bash
git clone <repository_url>
cd Snake_DQN
```

#### Step 2: Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Check GPU

```bash
python GpuCheck.py
```

Output: `True` (CUDA available) or `False` (CPU mode)

### Train Models

#### CNN Model

```bash
cd SnakeDQN
python main.py
```

#### Linear Model

```bash
cd SnakeDQN_linear
python main.py
```

**10× faster than CNN**

### Training Metrics

| Metric | Meaning | Goal |
|--------|---------|------|
| **Score** | Food eaten | Increasing |
| **Record** | Best score | Rising trend |
| **Epsilon** | Exploration | Decreasing |

### Expected Performance

- **CNN**: 25-35 points (500-1000 episodes)
- **Linear**: 30-50 points (100-300 episodes)

---

## Developer Guide

### Prerequisites

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Project Structure

```
Snake_DQN/
├── SnakeDQN/
│   ├── main.py           # Training entry point
│   ├── config.py         # Hyperparameters
│   ├── dqn/
│   │   ├── agent.py      # DQN agent
│   │   └── model.py      # CNN network
│   └── game/
│       └── snake_game.py # Game environment
└── SnakeDQN_linear/
    ├── main.py
    ├── dqn/
    │   ├── agent.py
    │   └── model.py
    └── game/
        └── snake_game.py
```

### config.py Parameters

#### AgentConfig

| Param | Default | Range | Meaning |
|-------|---------|-------|---------|
| max_memory | 100k | 50k-500k | Replay buffer |
| batch_size | 256 | 16-512 | Mini-batch |
| lr | 0.0001 | 0.00001-0.001 | Learning rate |
| epsilon_start | 1.0 | Fixed | Initial exploration |
| epsilon_min | 0.05 | 0.01-0.1 | Min exploration |
| epsilon_decay | 0.999 | 0.9995-0.9999 | Decay rate |
| gamma | 0.99 | 0.9-0.99 | Discount factor |
| sync_target_frames | 1000 | 500-2000 | Sync interval |

#### GameConfig

| Param | Default | Range |
|-------|---------|-------|
| grid_w / grid_h | 10 / 10 | 8-15 |
| speed | 100 | 50-200 |
| frame_h / frame_w | 84 / 84 | - |

### DQN Algorithm

#### Bellman Equation

```
Q(s, a) ≈ r + γ · max(Q(s', a'))
```

#### Experience Replay

```python
# Store transition
self.memory.append((state, action, reward, next_state, done))

# Sample mini-batch
mini_sample = random.sample(self.memory, batch_size)
```

**Benefits**: Remove correlation, stable learning

#### Target Network

```python
# Sync every N steps
if step % sync_interval == 0:
    target_model.load_state_dict(model.state_dict())
```

**Benefits**: Stable targets, faster convergence

#### Epsilon-Greedy

```
Game 1: 100% exploration
Game 100: ~90.5% exploration
Game 500: ~60.6% exploration
Game 1000: ~36.8% exploration
```

### CNN vs Linear

#### CNN Model

**Architecture**: Conv(1→32, 8×8) → Conv(32→64, 4×4) → Conv(64→64, 3×3) → FC(3136→512) → FC(512→3)

**Pros**: End-to-end, visual learning, generalizable
**Cons**: Slow, data-hungry, GPU needed
**Performance**: 25-35 points (500-1000 episodes)

#### Linear Model

**State**: 11-dim vector (collision 3 + direction 4 + food 4)
**Architecture**: Linear(11→256→3)

**Pros**: 10× faster, interpretable, CPU-friendly
**Cons**: Feature engineering, low extensibility
**Performance**: 30-50 points (100-300 episodes)

#### Choose Based On

| Goal | Use |
|------|-----|
| Learning | Linear |
| Max Performance | CNN |
| CPU only | Linear |
| Interpretability | Linear |
| Generalization | CNN |

### Reward Function

#### CNN Model

```
Eating food:    +15.0 + len(snake)×0.2
Long snake:     +5.0
Moving closer:  +1.0
Moving away:    -1.5
Time penalty:   -0.05/step
Collision:      -20.0
```

#### Linear Model (BFS-based)

```
Eating food:       +15
Self-trapped:      -15
Moving closer:     +1
Survival bonus:    +0.01
```

### Hyperparameter Tuning

**Insufficient Learning**:
- Increase lr (0.0001→0.001)
- Decrease epsilon_decay (0.999→0.995)
- Decrease batch_size (256→128)

**Unstable Learning**:
- Decrease lr (0.0001→0.00005)
- Increase batch_size (256→512)
- Increase sync frequency (1000→500)

**Slow Training**:
- Increase speed (100→200)
- Reduce grid (10→8)
- Reduce frame (84→64)

### State Representation

#### CNN (Pixel-based)

```
Pygame frame (210×210 RGB)
    ↓ [Grayscale + resize]
84×84 grayscale [0-255]
    ↓ [Normalize]
(1, 1, 84, 84) float [0-1]
```

#### Linear (Feature-based)

```
Index | Feature    | Value
------|-----------|-------
0-2   | Collisions | 0/1
3-6   | Direction  | 0/1 (one-hot)
7-10  | Food dir   | 0/1
```

---

## Conclusion

This project demonstrates Deep Q-Learning in practice. Experiment with both CNN and Linear models to understand architecture trade-offs and learning dynamics.

For questions or contributions, explore the codebase structure and inline code comments.
