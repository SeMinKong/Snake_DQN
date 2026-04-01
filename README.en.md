# Snake AI: Deep Q-Network (DQN)

**[한국어 버전](./README.md)**

A reinforcement learning project that teaches an agent to play the classic Snake game using **Deep Q-Networks (DQN)**. This project explores the trade-offs between end-to-end visual learning and hand-crafted feature engineering through two distinct implementations.

## Key Features

- **Dual Architectures**:
  - **CNN-based (SnakeDQN)**: Learns from a stack of 4 grayscale frames (4×84×84), giving the agent temporal context to infer movement direction.
  - **Linear-based (SnakeDQN_linear)**: Uses an 11-dimensional feature vector (collisions, direction, food location) for ~10x faster convergence.
- **Double DQN**: Decouples action selection (online network) from value evaluation (target network) to eliminate systematic Q-value overestimation. Applied to both variants.
- **Training Stability**: Huber Loss (SmoothL1), gradient clipping, and exponential epsilon decay work together to prevent unstable training.
- **Rebalanced Rewards**: Rescaled death/food/movement rewards to remove perverse incentives (e.g., cumulative retreat penalty exceeding the death penalty).
- **Experience Replay & Target Network**: Both techniques implemented across all variants for decorrelated sampling and stable Q-targets.

## Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **Game Engine**: Pygame
- **Computer Vision**: OpenCV, NumPy
- **Language**: Python 3.8+
- **Hardware**: CUDA support for GPU acceleration

## Project Structure

```text
├── SnakeDQN/           # CNN-based model (Visual learning)
│   ├── config.py       # Centralised hyperparameter config
│   ├── dqn/            # Agent & CNN model logic
│   └── game/           # Pixel-based game environment
├── SnakeDQN_linear/    # Linear-based model (Feature engineering)
│   ├── dqn/            # Agent & Dense model logic
│   └── game/           # Vector-based game environment
└── GpuCheck.py         # Utility to verify CUDA availability
```

## Technical Highlights

### 1. Double DQN
Vanilla DQN uses the same target network for both action selection and value evaluation, causing systematic overestimation of Q-values. **Double DQN** fixes this by selecting the best action with the online network and evaluating it with the target network. Applied to both the CNN and linear variants.

### 2. Frame Stacking (CNN variant)
A single frame gives the CNN no information about which direction the snake is moving. **4-frame stacking** provides temporal context so the network can infer velocity. The state shape changed from `(1, 84, 84)` to `(4, 84, 84)`.

### 3. Training Stability Improvements
- **Huber Loss (SmoothL1)**: More robust than MSE to large outlier rewards such as the death penalty.
- **Gradient Clipping**: Prevents exploding gradients during backpropagation.
- **Exponential Epsilon Decay**: The linear variant's broken `80 - n_games` schedule (goes negative after episode 80) was replaced with `max(1, 80 × 0.995^n)`.

### 4. Reward Rebalancing
| Event | Before | After | Reason |
|-------|--------|-------|--------|
| Death | -20 | -10 | Rebalanced relative to food reward |
| Eat food | +15 | +10 | Balanced against death penalty |
| Move toward food | +1 | +0.1 | Reduced dense reward magnitude |
| Move away from food | 0 | -0.1 | Penalise passive wandering |

## Quick Start

### Installation
```bash
git clone <repository-url>
cd Snake_DQN
pip install -r requirements.txt
# For GPU support (recommended)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Training
```bash
# To train the fast-converging Linear model
cd SnakeDQN_linear
python main.py

# To train the visual-based CNN model
cd ../SnakeDQN
python main.py
```

> **Note (CNN model)**: Frame stacking changed the input channels from 1 to 4. Any previously saved model (`models/model_cnn_best.pth`) is incompatible and must be retrained from scratch.

## Expected Performance
- **Linear**: 30-50 points within ~300 episodes.
- **CNN**: 25-35 points within ~1000 episodes (further gains expected with longer training due to 4-frame temporal context).

>  **Need more details?**
> For advanced hyperparameter tuning, reward shaping strategies, and state representations, please refer to the [Detailed Manual (DETAILS.en.md)](./DETAILS.en.md).

---
Built with PyTorch & Pygame.
