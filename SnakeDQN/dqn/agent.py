"""
agent.py — DQN agent for SnakeDQN (CNN-based).

All hyperparameters are sourced from config.py.  The model architecture is
injected at construction time so it can be swapped without touching this file.
"""

import logging
import random
from collections import deque
from typing import List

import numpy as np
import torch

from config import AGENT_CFG, MODEL_CFG, AgentConfig, ModelConfig
from dqn.model import DQN, DQNBase, QTrainer

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    """DQN agent that learns to play Snake via pixel-based CNN observations."""

    def __init__(
        self,
        agent_cfg: AgentConfig = AGENT_CFG,
        model_cfg: ModelConfig = MODEL_CFG,
        model: DQNBase | None = None,
    ) -> None:
        self.cfg = agent_cfg
        self.model_cfg = model_cfg

        self.n_games: int = 0
        self.epsilon: float = agent_cfg.epsilon_start
        self.learn_step_counter: int = 0

        self.memory: deque = deque(maxlen=agent_cfg.max_memory)

        # Allow a custom model to be injected; default to DQN (CNN).
        self.model: DQNBase = (
            model if model is not None
            else DQN(output_size=model_cfg.output_size,
                     input_channels=model_cfg.input_channels)
        ).to(device)

        self.target_model: DQNBase = DQN(
            output_size=model_cfg.output_size,
            input_channels=model_cfg.input_channels,
        ).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=agent_cfg.lr,
            gamma=agent_cfg.gamma,
            device=device,
        )

    # ------------------------------------------------------------------
    # State processing
    # ------------------------------------------------------------------

    def get_state(self, frame: np.ndarray) -> np.ndarray:
        """Add a channel dimension to the grayscale frame and cast to uint8."""
        state = np.expand_dims(frame, axis=0)
        return state.astype(np.uint8)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: List[int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_replay(self) -> None:
        """Sample a random mini-batch from memory and perform a training step."""
        if len(self.memory) < self.cfg.batch_size:
            return

        mini_sample = random.sample(self.memory, self.cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

        self.learn_step_counter += 1

        if self.learn_step_counter % self.cfg.sync_target_frames == 0:
            self.update_target_network()
            logger.info("Target network synced at step %d", self.learn_step_counter)

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> List[int]:
        """Return a one-hot action vector using an epsilon-greedy policy."""
        final_move: List[int] = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_t = (
                torch.tensor(state, dtype=torch.float)
                .to(device)
                .unsqueeze(0)
                / 255.0
            )
            with torch.no_grad():
                prediction = self.model(state_t)
            move = int(torch.argmax(prediction).item())

        final_move[move] = 1
        return final_move

    # ------------------------------------------------------------------
    # Epsilon decay & target-network sync
    # ------------------------------------------------------------------

    def update_epsilon(self) -> None:
        """Decay epsilon by one step after each episode."""
        self.epsilon = max(
            self.cfg.epsilon_min,
            self.epsilon * self.cfg.epsilon_decay,
        )

    def update_target_network(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_model.load_state_dict(self.model.state_dict())
