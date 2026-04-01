# Snake AI: DQN Architecture Details

## 1. Hyperparameters (config.py)
**AgentConfig**
- `max_memory = 100k`: Size of the Experience Replay buffer.
- `batch_size = 256`: Number of transitions sampled per training step.
- `lr = 0.0001`: Learning rate.
- `epsilon_decay = 0.999`: Exponentially decays the exploration rate down to `epsilon_min = 0.05`.
- `gamma = 0.99`: Discount factor for future rewards.
- `sync_target_frames = 1000`: Interval to copy main network weights to the target network.

## 2. State Representation
**CNN (SnakeDQN)**
- Raw Pygame 210x210 RGB frames are converted to 84x84 grayscale, normalized to `[0, 1]`, and fed as a `(1, 1, 84, 84)` tensor.

**Linear Network (SnakeDQN_linear)**
- Extracts an 11-dimensional binary vector:
  - `Index 0-2`: Danger straight, left, right (0 or 1).
  - `Index 3-6`: Current direction (One-hot).
  - `Index 7-10`: Food location relative to the head (Up/Down/Left/Right).

## 3. Reward Functions
- **CNN Model**: Eating (+15.0 + length*0.2), Collision (-20.0), Moving closer (+1.0), Moving away (-1.5), Time penalty (-0.05/step).
- **Linear Model**: Eating (+15), Collision (-15), Moving closer (+1), Survival bonus (+0.01).
