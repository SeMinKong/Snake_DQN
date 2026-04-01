"""
snake_game.py — Pygame-based Snake environment for SnakeDQN (CNN variant).

Observations are returned as 84×84 grayscale frames ready for the CNN.
"""

import logging
import random
import sys
from collections import namedtuple
from enum import Enum
from typing import Tuple

import cv2
import numpy as np
import pygame

from config import BLACK, BLUE, GRAY, RED, WHITE, GAME_CFG, GameConfig

logger = logging.getLogger(__name__)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")


class SnakeGameAI:
    """Grid-based Snake game that returns grayscale pixel frames as observations."""

    def __init__(self, cfg: GameConfig = GAME_CFG) -> None:
        self.cfg = cfg
        self.grid_w: int = cfg.grid_w
        self.grid_h: int = cfg.grid_h

        self.display_w: int = self.grid_w * cfg.block_size
        self.display_h: int = self.grid_h * cfg.block_size

        pygame.init()
        self.display = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption(cfg.caption)
        self.clock = pygame.time.Clock()

        # Game state — initialised by reset()
        self.direction: Direction = Direction.RIGHT
        self.head: Point = Point(0, 0)
        self.snake: list[Point] = []
        self.score: int = 0
        self.food: Point = Point(0, 0)
        self.frame_iteration: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset game state and return the initial frame."""
        self.direction = Direction.RIGHT
        self.head = Point(self.grid_w // 2, self.grid_h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
        ]
        self.score = 0
        self.food = Point(0, 0)
        self._place_food()
        self.frame_iteration = 0
        self._update_ui()
        return self.get_frame()

    def get_frame(self) -> np.ndarray:
        """Return the current display as an 84×84 grayscale numpy array."""
        view = pygame.surfarray.array3d(self.display)
        view = view.transpose([1, 0, 2])
        img_gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(
            img_gray,
            (self.cfg.frame_w, self.cfg.frame_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return img_resized

    def play_step(
        self, action: list[int], render: bool = True
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Advance the game by one step.

        Parameters
        ----------
        action:
            One-hot vector [straight, right, left].
        render:
            Whether to blit the frame and tick the clock.

        Returns
        -------
        frame, reward, game_over, score
        """
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.info("Quit event received — exiting.")
                pygame.quit()
                sys.exit(0)

        dist_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        reward: float = 0.0
        game_over: bool = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -20.0
            return self.get_frame(), reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 15.0 + len(self.snake) * 0.2
            self._place_food()
            self.frame_iteration = 0

            if len(self.snake) > (self.grid_w * self.grid_h) * 0.2:
                reward += 5.0
        else:
            self.snake.pop()

            dist_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            reward = 1.0 if dist_after < dist_before else -1.5
            reward -= 0.05

        if render:
            self._update_ui()
            self.clock.tick(self.cfg.speed)

        return self.get_frame(), reward, game_over, self.score

    def is_collision(self, pt: Point | None = None) -> bool:
        """Return True if *pt* (default: head) is out-of-bounds or hits the body."""
        if pt is None:
            pt = self.head
        if pt.x > self.grid_w - 1 or pt.x < 0 or pt.y > self.grid_h - 1 or pt.y < 0:
            return True
        return pt in self.snake[1:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_food(self) -> None:
        """Place food at a random cell not occupied by the snake."""
        x = random.randint(0, self.grid_w - 1)
        y = random.randint(0, self.grid_h - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self) -> None:
        """Render the current game state to the pygame display."""
        self.display.fill(BLACK)

        for i, pt in enumerate(self.snake):
            pixel_x = pt.x * self.cfg.block_size
            pixel_y = pt.y * self.cfg.block_size

            if i == 0:
                color = WHITE
            else:
                t = i / len(self.snake)
                base_val = max(80, min(220, int(220 - 140 * t)))
                color = (base_val, base_val, 255)

            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(pixel_x, pixel_y, self.cfg.block_size, self.cfg.block_size),
            )

            if i == 0:
                pygame.draw.rect(
                    self.display,
                    GRAY,
                    pygame.Rect(pixel_x + 4, pixel_y + 4, 12, 12),
                )

        food_x = self.food.x * self.cfg.block_size
        food_y = self.food.y * self.cfg.block_size
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(food_x, food_y, self.cfg.block_size, self.cfg.block_size),
        )

        pygame.display.flip()

    def _move(self, action: list[int]) -> None:
        """Update head position based on the relative action."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1

        self.head = Point(x, y)
