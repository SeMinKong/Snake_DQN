import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import cv2
import math

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 21
SPEED = 100
FRAME_H = 84
FRAME_W = 84
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, grid_w=10, grid_h=10):
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        self.display_w = self.grid_w * BLOCK_SIZE
        self.display_h = self.grid_h * BLOCK_SIZE
        
        self.display = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption('Snake CNN')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.grid_w // 2, self.grid_h // 2)
        self.snake = [self.head, 
                      Point(self.head.x - 1, self.head.y),
                      Point(self.head.x - 2, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self._update_ui()
        return self.get_frame()
        
    def _place_food(self):
        x = random.randint(0, self.grid_w - 1)
        y = random.randint(0, self.grid_h - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            
    def play_step(self, action, render=True):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
        
        dist_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -20
            return self.get_frame(), reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 15 + (len(self.snake) * 0.2)
            
            self._place_food()
            self.frame_iteration = 0
            
            if len(self.snake) > (self.grid_w * self.grid_h) * 0.2:
                 reward += 5
        else:
            self.snake.pop()
            
            dist_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            
            if dist_after < dist_before:
                reward = 1.0 
            else:
                reward = -1.5 
            
            reward -= 0.05

        if render:
            self._update_ui()
            self.clock.tick(SPEED)
            
        return self.get_frame(), reward, game_over, self.score

    def get_frame(self):
        view = pygame.surfarray.array3d(self.display)
        view = view.transpose([1, 0, 2])
        img_gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_NEAREST)
        return img_resized

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.grid_w - 1 or pt.x < 0 or pt.y > self.grid_h - 1 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for i, pt in enumerate(self.snake):
            pixel_x = pt.x * BLOCK_SIZE
            pixel_y = pt.y * BLOCK_SIZE
            
            if i == 0:
                color = WHITE
            else:
                t = i / len(self.snake)
                base_val = int(220 - 140 * t)
                base_val = max(80, min(220, base_val))
                color = (base_val, base_val, 255)

            pygame.draw.rect(self.display, color, pygame.Rect(pixel_x, pixel_y, BLOCK_SIZE, BLOCK_SIZE))
            
            if i == 0:
                pygame.draw.rect(self.display, (200, 200, 200), pygame.Rect(pixel_x+4, pixel_y+4, 12, 12))
            
        food_x = self.food.x * BLOCK_SIZE
        food_y = self.food.y * BLOCK_SIZE
        pygame.draw.rect(self.display, RED, pygame.Rect(food_x, food_y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT: x += 1
        elif self.direction == Direction.LEFT: x -= 1
        elif self.direction == Direction.DOWN: y += 1
        elif self.direction == Direction.UP: y -= 1
        
        self.head = Point(x, y)