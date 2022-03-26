import gym
import gym.spaces
import numpy as np
import ai2thor.controller
import cv2
import random

from .env import EnvBase
from .goal import GoalEnvBase

ACTIONS = [
    lambda add_noise: dict(action='MoveAhead', magnitude = add_noise(0.6), snapToGrid = False),
    lambda add_noise: dict(action='MoveBack', magnitude = add_noise(0.6), snapToGrid = False),
    lambda add_noise: dict(action='MoveLeft', magnitude = add_noise(0.25), snapToGrid = False),
    lambda add_noise: dict(action='MoveRight', magnitude = add_noise(0.25), snapToGrid = False),
    lambda add_noise: dict(action='Rotate', angle = add_noise(30)),
    lambda add_noise: dict(action='Rotate', angle = -add_noise(30)),
    lambda add_noise: dict(action='LookUp'),
    lambda add_noise: dict(action='LookDown')
]

    
class ContinuousEnv(EnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.initialize_kwargs['continuous'] = True
    
    def step(self, action):
        event = self._controller_step(action)
        return self._finish_step(event)

    def add_noise(self, value):
        if self.enable_noise:
            std = value * 0.04
            return value + np.random.normal(0.0, std)
        return value

    def _controller_step(self, action):
        action = ACTIONS[action](self.add_noise)
        if action['action'] == 'Rotate':
            deltaangle = action['angle']
            angle = (self.state[1]['y'] + deltaangle) % 360
            return self.controller.step(dict(action = 'Rotate', rotation = angle))
        else:
            return self.controller.step(action)


class GoalContinuousEnv(GoalEnvBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.initialize_kwargs['continuous'] = True
    
    def step(self, action):
        event = self._controller_step(action)
        return self._finish_step(event)

    def add_noise(self, value):
        if self.enable_noise:
            std = value * 0.04
            return value + np.random.normal(0.0, std)
        return value

    def _controller_step(self, action):
        action = ACTIONS[action](self.add_noise)
        if action['action'] == 'Rotate':
            deltaangle = action['angle']
            angle = (self.state[1]['y'] + deltaangle) % 360
            return self.controller.step(dict(action = 'Rotate', rotation = angle))
        else:
            return self.controller.step(action)

class AuxiliaryEnv(GoalContinuousEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype=np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype=np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (1,), dtype=np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype=np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype=np.uint8)))

        self.initialize_kwargs['renderClassImage'] = True
        self.initialize_kwargs['renderDepthImage'] = True

    def _render_goal(self, scene, goal):
        result, self.goal_image_path = self.goal_source.fetch_random_with_semantic(scene, goal)
        return result

    def observe(self, event=None):
        if event is None:
            event = self._last_event
        self._last_event = event
        self.state = (event.metadata['agent']['position'], event.metadata['agent']['rotation'])
        image = cv2.resize(event.frame, self.screen_size, interpolation=cv2.INTER_CUBIC)
        segmentation = cv2.resize(event.class_segmentation_frame, self.screen_size, interpolation=cv2.INTER_NEAREST)
        depth = (event.depth_frame * 255.0 / 5000.0).astype(np.uint8)
        depth = cv2.resize(depth, self.screen_size, interpolation=cv2.INTER_CUBIC)
        depth = np.expand_dims(depth, 2)
        goal_img, goal_seg = self.goal_observation
        return (image, np.copy(goal_img), depth, segmentation, np.copy(goal_seg))

    