"""
config.py — Central configuration for SnakeDQN (CNN-based).

All magic numbers are defined here so they can be changed in one place
without touching algorithm logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    # Replay memory capacity (number of transitions to store)
    max_memory: int = 100_000

    # Mini-batch size drawn from replay memory each training step
    batch_size: int = 256

    # Adam optimiser learning rate
    lr: float = 0.0001

    # Starting exploration rate (1.0 = fully random)
    epsilon_start: float = 1.0

    # Minimum exploration rate after decay
    epsilon_min: float = 0.05

    # Multiplicative decay applied to epsilon after every episode
    epsilon_decay: float = 0.999

    # Discount factor for future rewards
    gamma: float = 0.99

    # How many training steps between target-network syncs
    sync_target_frames: int = 1000


@dataclass(frozen=True)
class GameConfig:
    # Grid dimensions (number of cells)
    grid_w: int = 10
    grid_h: int = 10

    # Pixel size of each grid cell
    block_size: int = 21

    # Frames per second cap (higher = faster training)
    speed: int = 100

    # Grayscale frame dimensions fed to the CNN
    frame_h: int = 84
    frame_w: int = 84

    # Window caption
    caption: str = "Snake CNN"


@dataclass(frozen=True)
class ModelConfig:
    # Number of discrete actions {straight, right, left}
    output_size: int = 3

    # Grayscale → 1 input channel
    input_channels: int = 1

    # Filename used when saving the best model
    save_filename: str = "model_cnn_best.pth"

    # Directory where model checkpoints are written
    model_dir: str = "./models"


# ---------------------------------------------------------------------------
# Colour constants (RGB tuples) used by the renderer
# ---------------------------------------------------------------------------
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (200, 0, 0)
BLUE: tuple[int, int, int] = (0, 0, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)
GRAY: tuple[int, int, int] = (200, 200, 200)


# Shared default instances — importers can use these directly or override.
AGENT_CFG = AgentConfig()
GAME_CFG = GameConfig()
MODEL_CFG = ModelConfig()
