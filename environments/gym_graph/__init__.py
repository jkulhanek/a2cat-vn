import gym
from .download import available_scenes

TASKS = [
    ('', (12, 3, 0)),
]

for key, name in available_scenes():
    gym.register(
        id = 'Graph' + key + '-v0',
        entry_point = 'environments.gym_graph.graph:OrientedGraphEnv',
        max_episode_steps = 100,
        kwargs = dict(
            graph_name = name
        )
    )

gym.register(
    id = 'OrientedGraph-v0',
    entry_point = 'environments.gym_graph.graph:OrientedGraphEnv',
    max_episode_steps = 900,
)

gym.register(
    id = 'AuxiliaryGraph-v0',
    entry_point = 'environments.gym_graph.graph:GoalGymGraphAuxiliaryEnv',
    max_episode_steps = 900,
)