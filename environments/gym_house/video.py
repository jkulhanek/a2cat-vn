import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from House3D.core import Environment
from House3D.objrender import RenderAPIThread as RenderAPI
from deep_rl.configuration import configuration
from .cenv import GymHouseState
from .env import create_configuration
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import cv2
import os


class RenderVideoWrapper(gym.Wrapper):
    def __init__(self, env, path, action_frames = 10, width = 500, height = 500, renderer_config = None):
        super().__init__(env)
        self.path = path
        self.action_frames = action_frames
        self.size = (height, width)
        self.api = RenderAPI(w = width, h = height, device = 0)
        self.renderer_config = create_configuration(renderer_config)
        self.ep_states = []
        self._renderer_cache = (None, None)
        pass

    def reset(self):
        if len(self.ep_states) > 0:
            self.render_video(self.ep_states)
            self.ep_states = []

        observation = self.env.reset()
        self.ep_states = [self.unwrapped.state]
        return observation

    def step(self, action):
        obs, reward, done, stats = self.env.step(action)
        self.ep_states.append(self.unwrapped.state)
        if done:
            self.render_video(self.ep_states)
            self.ep_states = []

        return obs, reward, done, stats

    def render_video(self, states):
        house_id = states[0].house_id
        if self._renderer_cache[0] == house_id:
            renderer = self._renderer_cache[1]

        renderer = Environment(self.api, house_id, self.renderer_config)
        renderer.reset()
        video_id = len(os.listdir(self.path)) + 1
        output_filename = "vid-%s.avi" % video_id

        height, width = self.size
        writer = VideoWriter(os.path.join(self.path, output_filename), VideoWriter_fourcc(*"XVID"), 30.0, (2 * width, height))
        goal_image = '%s-render_rgb.png' % states[0].target_image
        goal_image = cv2.imread(goal_image)
        goal_image = cv2.resize(goal_image, self.size, interpolation = cv2.INTER_CUBIC)

        def render_single(position):
            renderer.reset(*position)
            frame = renderer.render()
            frame = np.concatenate([frame, goal_image], axis = 1)
            writer.write(frame)

        state = states[0]
        position = position = state.x, state.y, state.rotation
        for state in states[1:]:
            old_position = position           
            position = position = state.x, state.y, state.rotation

            for j in range(self.action_frames):
                interpolated = tuple(map(lambda a, b: a + (b - a) * j / self.action_frames, old_position, position))
                render_single(interpolated)

        for _ in range(self.action_frames):
            render_single(position)

        render_single(position)        
        writer.release()

        self._renderer_cache = (house_id, renderer)

        

    