import os
import math
import numpy as np
import torch
from torch import nn

from deep_rl import register_trainer
from deep_rl.common.schedules import LinearSchedule, MultistepSchedule
from deep_rl.testing import TestingEnv, TestingVecEnv
from deep_rl.common.env import RewardCollector, TransposeImage, ScaledFloatFrame
from deep_rl.common.vec_env import DummyVecEnv, SubprocVecEnv
from deep_rl.actor_critic.unreal.utils import UnrealEnvBaseWrapper
from deep_rl.configuration import configuration
import deep_rl

import environments
from experiments.ai2_auxiliary.trainer import AuxiliaryTrainer
from models import AuxiliaryBigGoalHouseModel as Model
from utils.download import require_resource

VALIDATION_PROCESSES = 1 # note: single environment is supported at the moment

@require_resource('thor-scene-images-311')
@register_trainer(max_time_steps = 40e6, validation_period=None, validation_episodes=None,  episode_log_interval = 10, saving_period = 100000, save = True)
class Trainer(AuxiliaryTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_processes = 16
        self.max_gradient_norm = 0.5
        self.rms_alpha = 0.99
        self.rms_epsilon = 1e-5
        self.num_steps = 20
        self.gamma = .99
        self.allow_gpu = True
        self.learning_rate = LinearSchedule(7e-4, 0, self.max_time_steps)

        self.rp_weight = 1.0
        self.pc_weight = 0.05
        self.vr_weight = 1.0
        self.auxiliary_weight = 0.1
        #self.pc_cell_size = 

    def _get_input_for_pixel_control(self, inputs):
        return inputs[0][0]

    def create_env(self, kwargs):
        env, self.validation_env = create_envs(self.num_processes, kwargs)
        return env

    def create_model(self):
        model = Model(self.env.observation_space.spaces[0].spaces[0].shape[0], self.env.action_space.n)
        return model


def create_envs(num_training_processes, env_kwargs):
    def wrap(env):
        env = RewardCollector(env)
        env = TransposeImage(env)
        env = ScaledFloatFrame(env)
        env = UnrealEnvBaseWrapper(env)
        return env

    thunk = lambda: wrap(environments.make(**env_kwargs))
    env = SubprocVecEnv([thunk for _ in range(num_training_processes)])
    return env, None

def default_args():
    return dict(
        env_kwargs = dict(
            id = 'AuxiliaryThor-v1',
            scenes = ['311'], 
            screen_size=(172,172), 
            enable_noise = True),
        model_kwargs = dict()
    )
