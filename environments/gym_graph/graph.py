import gym
import numpy as np
import gym.spaces
from graph.core import GraphResize
from graph.util import load_graph, step, sample_initial_state, is_valid_state
from .download import get_graph
import random

class OrientedGraphEnv(gym.Env):
    def __init__(self, graph_name = None, graph_file = None, goals = None, screen_size = (174,174), rewards = [1.0, 0.0, 0.0]):
        if graph_name is not None:
            self.graph = get_graph(graph_name)
        elif graph_file is not None:
            from graph.util import load_graph
            self.graph = load_graph(graph_file)
        else:
            raise Exception('Not supported')

        self.graph = GraphResize(self.graph, screen_size)

        if goals is None:
            self.goals = self.graph.goals
        else:
            self.goals = goals

        if self.graph.dtype == np.float32:
            self.observation_space = gym.spaces.Box(0.0, 1.0, self.graph.observation_shape, np.float32)
        elif self.graph.dtype == np.uint8:
            self.observation_space = gym.spaces.Box(0, 255, self.graph.observation_shape, np.uint8)
        else:
            raise Exception('Unsupported observation type')
        
        self.action_space = gym.spaces.Discrete(4)
        self.state = None
        self.largest_distance = np.max(self.graph.graph)
        self.complexity = None
        self.rewards = rewards

    @property
    def unwrapped(self):
        return self

    def set_complexity(self, complexity = None):
        self.complexity = complexity

    def reset(self):
        self.goal = random.choice(self.goals) if isinstance(self.goals, list) else self.goals
        
        optimal_distance = None
        if self.complexity is not None:
            optimal_distance = self.complexity * (self.largest_distance + 4 - 1) + 1
        state = sample_initial_state(self.graph, self.goal, optimal_distance = optimal_distance)
        self.state = state
        return self.observe(self.state)

    def observe(self, state):
        observation = self.graph.render(state[:2], state[2])
        return observation

    def is_goal(self, state):
        return max(map(lambda a,b: abs(a - b), state[:2], self.goal[:2])) == 0 and state[2] == self.goal[2]

    def browse(self):
        from .browser import GoalKeyboardAgent
        return GoalKeyboardAgent(self)

    def step(self, action):
        nstate = step(self.state, action)
        if not is_valid_state(self.graph.maze, nstate):
            # We can either end the episode with failure
            # Or continue with negative reward
            return self.observe(self.state), self.rewards[2], False, dict(state = self.state)

        else:
            self.state = nstate
            if self.is_goal(self.state):
                return self.observe(self.state), self.rewards[0], True, dict(state = self.state, win = True)
            else:
                return self.observe(self.state), self.rewards[1], False, dict(state = self.state)

    def render(self, mode = 'human'):
        img = self.observe(self.state)
        if self.observation_space.dtype == np.float32:
            img = (255 * img).astype(np.uint8)

        if mode == 'human':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        elif mode == 'rgbarray':
            return img
        else:
            raise Exception("Render mode %s is not supported" % mode)


class GoalGymGraphAuxiliaryEnv(OrientedGraphEnv):
    def __init__(self, goals = None, screen_size = (174, 174,), **kwargs):
        super().__init__(goals = goals, **kwargs)

        self.screen_size = screen_size
        self.observation_space = gym.spaces.Tuple((
            self.observation_space,
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype = np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (1,), dtype = np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype = np.uint8),
            gym.spaces.Box(0, 255, self.screen_size + (3,), dtype = np.uint8)))

        self._cached_goal = (None, None)

    def render_goal(self):
        cached, value = self._cached_goal
        if cached is None or cached != self.goal:
            value = self.graph.render(self.goal[:2], self.goal[2], modes = ['rgb','segmentation'])
            self._cached_goal = (self.goal, value,)
        return value

    def observe(self, state):
        goal_rgb, goal_segmentation = self.render_goal()
        rgb, depth, segmation = self.graph.render(state[:2], state[2], modes = ['rgb','depth', 'segmentation'])
        return (rgb, goal_rgb, depth, segmation, goal_segmentation)
