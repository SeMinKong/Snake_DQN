import pygame
import random
from enum import Enum
from collections import namedtuple, deque
import numpy as np

pygame.init()

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 80
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

class SnakeGameAI:
    def __init__(self, grid_w=20, grid_h=20):
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        self.display_w = self.grid_w * BLOCK_SIZE
        self.display_h = self.grid_h * BLOCK_SIZE
        
        self.display = pygame.display.set_mode((self.display_w, self.display_h))
        pygame.display.set_caption('Snake AI Smart')
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
        return self.get_state_info()
        
    def _place_food(self):
        x = random.randint(0, self.grid_w - 1)
        y = random.randint(0, self.grid_h - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _get_accessible_area(self, start_point):
        queue = deque([start_point])
        visited = set()
        visited.add(start_point)
        obstacles = set(self.snake)
        count = 0
        
        limit = len(self.snake) + 2 

        while queue:
            curr = queue.popleft()
            count += 1
            if count >= limit: 
                return count
            
            x, y = curr.x, curr.y
            neighbors = [
                Point(x + 1, y), Point(x - 1, y),
                Point(x, y + 1), Point(x, y - 1)
            ]
            
            for n in neighbors:
                if (0 <= n.x < self.grid_w and 0 <= n.y < self.grid_h and
                    n not in obstacles and n not in visited):
                    visited.add(n)
                    queue.append(n)
        return count

    def _get_path_distance(self):
        start = self.head
        target = self.food
        
        queue = deque([(start, 0)])
        visited = set()
        visited.add(start)
        obstacles = set(self.snake)
        
        while queue:
            curr, dist = queue.popleft()
            if curr == target:
                return dist
            
            x, y = curr.x, curr.y
            neighbors = [
                Point(x + 1, y), Point(x - 1, y),
                Point(x, y + 1), Point(x, y - 1)
            ]
            
            for n in neighbors:
                if (0 <= n.x < self.grid_w and 0 <= n.y < self.grid_h and
                    n not in obstacles and n not in visited):
                    visited.add(n)
                    queue.append((n, dist + 1))
                    
        return 10000 

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        dist_before = self._get_path_distance()

        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 150 * len(self.snake):
            game_over = True
            reward = -20
            return reward, game_over, self.score
            
        if self.head == self.food:
            self.score += 1
            reward = 15
            self._place_food()
        else:
            self.snake.pop()
            
            dist_after = self._get_path_distance()
            
            accessible_count = self._get_accessible_area(self.head)
            if accessible_count < len(self.snake):
                reward = -15
            
            elif dist_after < dist_before:
                reward = 1
            else:
                reward = 0 
                
        reward += 0.01
        
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.grid_w - 1 or pt.x < 0 or pt.y > self.grid_h - 1 or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x * BLOCK_SIZE + 4, pt.y * BLOCK_SIZE + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
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
        
    def get_state_info(self):
        return {
            'head': self.head,
            'food': self.food,
            'snake': self.snake,
            'direction': self.direction,
            'grid_w': self.grid_w,
            'grid_h': self.grid_h
        }