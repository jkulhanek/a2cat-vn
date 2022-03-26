import gym
from House3D import objrender, Environment
from House3D.roomnav import RoomNavTask, n_discrete_actions
import numpy as np
import random
import os
import cv2

def create_configuration(config = None):
    if config is None:
        from configuration import configuration
        config = configuration.get('house3d')
    path = config['framework_path']
    return {
        "colorFile": os.path.join(path, "House3D/metadata/colormap_coarse.csv"),
        "roomTargetFile": os.path.join(path,"House3D/metadata/room_target_object_map.csv"),
        "modelCategoryFile": os.path.join(path,"House3D/metadata/ModelCategoryMapping.csv"),
        "prefix": os.path.join(config['dataset_path'], 'house')
    }

class GymHouseEnvOriginal(gym.Env):
    def __init__(self, scene = '2364b7dcc432c6d6dcc59dba617b5f4b', screen_size = (84,84), goals = ['kitchen'], hardness=0.3, configuration = None):
        super().__init__()

        self.screen_size = screen_size
        self.room_types = goals
        self.scene = scene
        self.configuration = create_configuration(configuration)
        self.hardness = hardness
        self._env = None

        self.action_space = gym.spaces.Discrete(n_discrete_actions)
        self.observation_space = gym.spaces.Box(0, 255, screen_size + (3,), dtype = np.uint8)

    def _initialize(self):
        h, w = self.screen_size
        api = objrender.RenderAPI(w = w, h = h, device = 0)
        env = Environment(api, self.scene, self.configuration)
        env.reset()
        env = RoomNavTask(env, discrete_action = True, depth_signal = False, segment_input = False, reward_type=None, hardness=self.hardness)
        self._env = env

    def observation(self, observation):
        return observation

    def _ensure_env_ready(self):
        if self._env is None:
            self._initialize()

    def _reset_with_target(self, target):
        return self._env.reset(target)

    def reset(self):
        self._ensure_env_ready()

        goals = set(self._env.house.all_desired_roomTypes)
        if self.room_types is not None:
            goals.intersection_update(set(self.room_types))

        target = random.choice(list(goals))

        return self.observation(self._reset_with_target(target))

    @property
    def info(self):
        self._ensure_env_ready()
        return self._env.info

    def step(self, action):
        self._ensure_env_ready()
        obs, reward, done, info = self._env.step(action)
        return self.observation(obs), reward, done, info

def GymHouseEnv2(scene = '05cac5f7fdd5f8138234164e76a97383', screen_size = (84,84), goals = ['living_room'], configuration = None):
    h, w = screen_size
    api = objrender.RenderAPI(w = w, h = h, device = 0)
    env = Environment(api, scene, create_configuration(configuration))
    env.reset()
    env = RoomNavTask2(env, discrete_action = True, depth_signal = False, segment_input = False, reward_type=None)
    env.observation_space.dtype = np.uint8
    return GymHouseWrapper(env, room_types=goals, screen_size = screen_size)