import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os


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

        tile_size = 25
        self.tile_size = tile_size

        base_path = os.path.join("gymnasium_env", "envs", "img")
        self.images = {
            2: pygame.transform.scale(pygame.image.load(os.path.join(base_path, "reward.png")), (tile_size, tile_size)),
            3: pygame.transform.scale(pygame.image.load(os.path.join(base_path, "penalty.png")),
                                      (tile_size, tile_size)),
            9: pygame.transform.scale(pygame.image.load(os.path.join(base_path, "meta.png")), (tile_size, tile_size)),
            "agent": pygame.transform.scale(pygame.image.load(os.path.join(base_path, "agent.png")),
                                            (tile_size, tile_size)),
        }

    def _generate_maze(self):
        self.grid[:] = 0
        self.grid[19, 19] = 9

        walls = (
                [(i, 3) for i in range(1, 10)] +
                [(9, j) for j in range(3, 12)] +
                [(i, 11) for i in range(10, 18)] +
                [(17, j) for j in range(6, 16)] +
                [(i, 6) for i in range(5, 17)] +
                [(5, j) for j in range(6, 10)] +
                [(13, j) for j in range(0, 5)] +
                [(j, 15) for j in range(3, 10)] +
                [(11, 13), (11, 14), (12, 14), (13, 14)]
        )

        for x, y in walls:
            self.grid[x, y] = 1

        for x, y in [(1, 1), (7, 17), (14, 2), (18, 18)]:
            self.grid[x, y] = 2

        for x, y in [(3, 7), (11, 5), (16, 12), (5, 17)]:
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
            reward = 20
            self.grid[x, y] = 0
        elif tile == 3:
            reward = -2
        elif tile == 9:
            reward = 20
            done = True

        return np.array(self.agent_pos), reward, done, False, {}

    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode(
                (self.grid_size[1] * self.tile_size, self.grid_size[0] * self.tile_size))
            pygame.display.set_caption("Maze Agent")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.grid_size[1] * self.tile_size, self.grid_size[0] * self.tile_size))
        canvas.fill((255, 255, 255))

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                tile_type = self.grid[i, j]
                if tile_type in self.images:
                    canvas.blit(self.images[tile_type], (j * self.tile_size, i * self.tile_size))
                elif tile_type == 1:
                    pygame.draw.rect(canvas, (0, 0, 0),
                                     pygame.Rect(j * self.tile_size, i * self.tile_size, self.tile_size,
                                                 self.tile_size))

        ax, ay = self.agent_pos
        canvas.blit(self.images["agent"], (ay * self.tile_size, ax * self.tile_size))

        self.window.blit(canvas, (0, 0))
        pygame.display.update()
        self.clock.tick(5)

    def close(self):
        if self.window:
            pygame.quit()
