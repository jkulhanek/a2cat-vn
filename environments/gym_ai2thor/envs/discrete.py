import gym
import gym.spaces
import numpy as np
import ai2thor.controller
import cv2
import random

from .env import EnvBase

ACTIONS = [
    dict(action='MoveAhead'),
    dict(action='MoveBack'),
    dict(action='MoveLeft'),
    dict(action='MoveRight'),
    dict(action='RotateRight'),
    dict(action='RotateLeft'),
    dict(action='LookUp'),
    dict(action='LookDown')
]

class DiscreteEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
    
    def step(self, action):
        event = self.controller.step(ACTIONS[action])
        return self._finish_step(event)

    def browse(self):
        from .browser import KeyboardAgent
        return KeyboardAgent(self)