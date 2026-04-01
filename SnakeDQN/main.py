"""
main.py — Training entry-point for SnakeDQN (CNN-based).

Run with:
    python main.py
"""

import logging
import sys

from config import MODEL_CFG
from dqn.agent import Agent
from game.snake_game import SnakeGameAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def train() -> None:
    """Main training loop."""
    record: int = 0
    agent = Agent()
    game = SnakeGameAI()

    frame = game.reset()
    state_old = agent.get_state(frame)

    n_steps: int = 0
    logger.info("Training started (CNN-based DQN).")

    while True:
        final_move = agent.get_action(state_old)

        frame_new, reward, done, score = game.play_step(final_move, render=True)
        state_new = agent.get_state(frame_new)

        agent.remember(state_old, final_move, reward, state_new, done)

        n_steps += 1
        agent.train_replay()

        state_old = state_new

        if done:
            game.reset()
            agent.n_games += 1
            agent.update_epsilon()

            frame = game.get_frame()
            state_old = agent.get_state(frame)
            n_steps = 0

            if score > record:
                record = score
                agent.model.save(MODEL_CFG.save_filename, MODEL_CFG)

            logger.info(
                "Game %d | Score: %d | Record: %d | Epsilon: %.3f",
                agent.n_games,
                score,
                record,
                agent.epsilon,
            )


if __name__ == "__main__":
    train()
