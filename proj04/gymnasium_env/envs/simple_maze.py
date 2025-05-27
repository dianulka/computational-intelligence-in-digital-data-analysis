import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SimpleMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.grid_size = (20, 20)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.grid_size)-1, shape=(2,), dtype=np.int32)
        self.grid = np.zeros(self.grid_size, dtype=int)
        self._generate_maze()
        self.agent_pos = [0, 0]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _generate_maze(self):
        self.grid[:] = 0
        self.grid[19, 19] = 9
        walls = [(i, 5) for i in range(1, 15)] + [(15, j) for j in range(5, 15)] + \
                [(i, 14) for i in range(5, 19)] + [(5, j) for j in range(6, 14)] + \
                [(10, j) for j in range(0, 10)] + [(j, 10) for j in range(11, 19)] + \
                [(8, 8), (8, 9), (8, 10), (8, 11), (12, 12), (13, 12), (14, 12), (15, 12)]
        for x, y in walls:
            self.grid[x, y] = 1
        for x, y in [(2, 2), (4, 18), (10, 16), (17, 1)]:
            self.grid[x, y] = 2
        for x, y in [(6, 6), (12, 7), (18, 10), (15, 3)]:
            self.grid[x, y] = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_maze()
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos), {}

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: y = max(0, y - 1)
        elif action == 1: y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(self.grid_size[0] - 1, x + 1)
        if self.grid[x, y] == 1: x, y = self.agent_pos
        self.agent_pos = [x, y]

        tile = self.grid[x, y]
        reward, done = -0.2, False
        if tile == 2:
            reward = 5
            self.grid[x, y] = 0
        elif tile == 3:
            reward = -2
        elif tile == 9:
            reward = 20
            done = True

        return np.array(self.agent_pos), reward, done, False, {}

    def render(self):
        tile_size = 25
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.grid_size[1] * tile_size, self.grid_size[0] * tile_size))
            pygame.display.set_caption("Maze Agent")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size[1] * tile_size, self.grid_size[0] * tile_size))
        canvas.fill((255, 255, 255))
        colors = {0: (255, 255, 255), 1: (0, 0, 0), 2: (0, 255, 0), 3: (128, 0, 128), 9: (255, 0, 0)}
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pygame.draw.rect(canvas, colors.get(self.grid[i, j], (200, 200, 200)),
                                 pygame.Rect(j * tile_size, i * tile_size, tile_size, tile_size))
        ax, ay = self.agent_pos
        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(ay * tile_size, ax * tile_size, tile_size, tile_size))

        self.window.blit(canvas, (0, 0))
        pygame.display.update()

        self.clock.tick(5)

    def close(self):
        if self.window:
            pygame.quit()
