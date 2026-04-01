# Snake AI: Deep Q-Network (DQN) 🐍

**[한국어 버전](./README.md)**

A reinforcement learning project that teaches an agent to play the classic Snake game using **Deep Q-Networks (DQN)**. This project explores the trade-offs between end-to-end visual learning and hand-crafted feature engineering through two distinct implementations.

## 🚀 Key Features

- **Dual Architectures**:
  - **CNN-based (SnakeDQN)**: Learns directly from 84x84 grayscale pixel inputs, mimicking human-like visual perception.
  - **Linear-based (SnakeDQN_linear)**: Uses an 11-dimensional feature vector (collisions, direction, food location) for 10x faster convergence.
- **Advanced RL Techniques**: Implements **Experience Replay** to break data correlation and **Target Networks** for stable Q-value convergence.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation, starting from random moves and gradually transitioning to greedy actions.
- **Custom Reward Shaping**: Specialized reward functions including BFS-based pathfinding to guide the agent through complex maze-like scenarios.

## 🛠 Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **Game Engine**: Pygame
- **Computer Vision**: OpenCV, NumPy
- **Language**: Python 3.8+
- **Hardware**: CUDA support for GPU acceleration

## 🏗 Project Structure

```text
├── SnakeDQN/           # CNN-based model (Visual learning)
│   ├── dqn/            # Agent & CNN model logic
│   └── game/           # Pixel-based game environment
├── SnakeDQN_linear/    # Linear-based model (Feature engineering)
│   ├── dqn/            # Agent & Dense model logic
│   └── game/           # Vector-based game environment
└── GpuCheck.py         # Utility to verify CUDA availability
```

## 🧠 Technical Highlights

### 1. CNN vs. Linear Trade-offs
I implemented both models to analyze the performance gap. While the **Linear model** reaches high scores within 300 episodes due to efficient feature extraction, the **CNN model** demonstrates superior generalization capabilities by learning spatial patterns directly from raw pixels, albeit requiring more training time.

### 2. Stability in Reinforcement Learning
To solve the instability issues common in DQN, I integrated a **Target Network** that syncs every 1,000 steps. This "fixed target" approach prevents the agent from chasing a moving target, leading to much smoother loss curves and faster mastery of the game.

## 🏁 Quick Start

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

## 📊 Expected Performance
- **Linear**: 30-50 points within ~300 episodes.
- **CNN**: 25-35 points within ~1000 episodes.

> 💡 **Need more details?**
> For advanced hyperparameter tuning, reward shaping strategies, and state representations, please refer to the [Detailed Manual (DETAILS.en.md)](./DETAILS.en.md).

---
Built with ❤️ using PyTorch & Pygame.
