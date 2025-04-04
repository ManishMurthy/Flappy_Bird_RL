import os
import pygame
import random
import numpy as np
import gym
from gym import spaces

class FlappyBirdEnv(gym.Env):
    """Flappy Bird Gym Environment"""

    def __init__(self):
        pygame.init()
        self.screen_width = 288
        self.screen_height = 512
        self.pipe_gap = 100
        self.gravity = 1
        self.jump_strength = -9
        self.bird_y = self.screen_height // 2
        self.bird_x = self.screen_width // 5
        self.bird_velocity = 0

        self._load_resources()
        self.reset()

        self.action_space = spaces.Discrete(2)  # Flap or do nothing
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -10]), 
            high=np.array([self.screen_width, self.screen_height, 10]), 
            dtype=np.float32
        )

    def _load_resources(self):
        """Loads game assets correctly."""
        ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "sprites")  # ✅ Fixed Path

        self.IMAGES = {
            'background': pygame.image.load(os.path.join(ASSETS_PATH, 'background-day.png')),
            'pipe': pygame.image.load(os.path.join(ASSETS_PATH, 'pipe-green.png')),
            'base': pygame.image.load(os.path.join(ASSETS_PATH, 'base.png')),
            'bird': [
                pygame.image.load(os.path.join(ASSETS_PATH, 'yellowbird-upflap.png')),
                pygame.image.load(os.path.join(ASSETS_PATH, 'yellowbird-midflap.png')),
                pygame.image.load(os.path.join(ASSETS_PATH, 'yellowbird-downflap.png'))
            ]
        }

    def reset(self):
        """Resets the environment"""
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.pipes = [{'x': self.screen_width, 'y': random.randint(50, self.screen_height - 50)}]
        return np.array([self.bird_x, self.bird_y, self.bird_velocity], dtype=np.float32)

    def step(self, action):
        """Game step logic"""
        if action == 1:
            self.bird_velocity = self.jump_strength
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity

        for pipe in self.pipes:
            pipe['x'] -= 5
        if self.pipes[0]['x'] < -50:
            self.pipes.pop(0)
            self.pipes.append({'x': self.screen_width, 'y': random.randint(50, self.screen_height - 50)})

        done = self.bird_y < 0 or self.bird_y > self.screen_height  # Collision check
        reward = 1 if not done else -10
        return np.array([self.bird_x, self.bird_y, self.bird_velocity], dtype=np.float32), reward, done, {}

    def render(self):
        """Render the game window"""
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.blit(self.IMAGES['background'], (0, 0))
        pygame.display.update()
