"""
model.py — Neural network definitions for SnakeDQN.

DQNBase defines the shared interface (forward + save).
DQN is the default CNN-based implementation used by the agent.
Swapping the model architecture only requires passing a different
DQNBase subclass to Agent — no other code needs to change.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import MODEL_CFG, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base — defines the interface every model must implement
# ---------------------------------------------------------------------------

class DQNBase(nn.Module):
    """Base class for all DQN model architectures."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save(self, file_name: str | None = None, cfg: ModelConfig = MODEL_CFG) -> None:
        """Save model weights to *cfg.model_dir/file_name*."""
        target = file_name or cfg.save_filename
        os.makedirs(cfg.model_dir, exist_ok=True)
        path = os.path.join(cfg.model_dir, target)
        torch.save(self.state_dict(), path)
        logger.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Default CNN architecture (84×84 grayscale input → Q-values)
# ---------------------------------------------------------------------------

class DQN(DQNBase):
    """
    Three-layer convolutional feature extractor followed by two fully-connected
    layers.  Expects 84×84 single-channel (grayscale) input tensors normalised
    to [0, 1].

    Conv output size derivation (input 84×84):
        conv1: (84-8)/4 + 1 = 20   → 20×20
        conv2: (20-4)/2 + 1 = 9    → 9×9
        conv3: (9-3)/1  + 1 = 7    → 7×7
        flat:  64 * 7 * 7 = 3136
    """

    _FC_INPUT_SIZE: int = 64 * 7 * 7  # 3136

    def __init__(self, output_size: int = MODEL_CFG.output_size,
                 input_channels: int = MODEL_CFG.input_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._FC_INPUT_SIZE, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

class QTrainer:
    """Bellman-equation trainer using a frozen target network."""

    def __init__(self, model: DQNBase, target_model: DQNBase,
                 lr: float, gamma: float, device: torch.device) -> None:
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: "np.ndarray",
        action: "np.ndarray",
        reward: "np.ndarray",
        next_state: "np.ndarray",
        done: "np.ndarray",
    ) -> None:
        """Perform one gradient-descent step on a mini-batch of transitions."""
        import numpy as np  # local import avoids circular dependency

        state_t = torch.from_numpy(np.asarray(state)).to(self.device).float() / 255.0
        next_state_t = torch.from_numpy(np.asarray(next_state)).to(self.device).float() / 255.0
        action_t = torch.tensor(action, dtype=torch.long).to(self.device)
        reward_t = torch.tensor(reward, dtype=torch.float).to(self.device)
        done_t = torch.tensor(done, dtype=torch.bool).to(self.device)

        # Add batch dimension when called with a single transition
        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)
            next_state_t = next_state_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)
            reward_t = reward_t.unsqueeze(0)
            done_t = done_t.unsqueeze(0)

        pred = self.model(state_t)

        with torch.no_grad():
            next_pred = self.target_model(next_state_t)

        target = pred.clone()
        next_q_max = torch.max(next_pred, dim=1)[0]
        q_new = reward_t + self.gamma * next_q_max * (~done_t)

        batch_indices = torch.arange(len(state_t), dtype=torch.long).to(self.device)
        action_indices = torch.argmax(action_t, dim=1)
        target[batch_indices, action_indices] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
