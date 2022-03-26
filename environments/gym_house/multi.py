from gym.vector import AsyncVectorEnv, SyncVectorEnv
from House3D.core import Environment, local_create_house
import random
import time
import gym

def create_multiscene(num_processes, scenes, wrap = lambda e: e, **kwargs):
    assert len(scenes) % num_processes == 0, "The number of processes %s must devide the number of scenes %s" % (num_processes, len(scenes))
    scenes_per_process = len(scenes) // num_processes

    funcs = []
    for i in range(num_processes):
        funcs.append(lambda: wrap(gym.make(**kwargs, scene = scenes[i * scenes_per_process:(i + 1) * scenes_per_process])))

    if num_processes == 1:
        return SyncVectorEnv(funcs)

    else:
        return AsyncVectorEnv(funcs)


class MultiHouseEnv(Environment):
    def __init__(self, api, houses, config, seed=None):
        """
        Args:
            houses: a list of house id or `House` instance.
        """
        print('Generating all houses ...')
        ts = time.time()
        if not isinstance(houses, list):
            houses = [houses]
        _args = [(h, config) for h in houses]
        self.all_houses = list(map(lambda x: local_create_house(*x), _args))
        print('  >> Done! Time Elapsed = %.4f(s)' % (time.time() - ts))
        for i, h in enumerate(self.all_houses):
            h._id = i
            h._cachedLocMap = None
        super(MultiHouseEnv, self).__init__(
            api, house=self.all_houses[0], config=config, seed=seed)

    def reset_house(self, house_id=None):
        """
        Reset the scene to a different house.
        Args:
            house_id (int): a integer in range(0, self.num_house).
                If None, will choose a random one.
        """
        if house_id is None:
            self.house = random.choice(self.all_houses)
        else:
            self.house = self.all_houses[house_id]
        self._load_objects()

    def cache_shortest_distance(self):
        # TODO
        for house in self.all_houses:
            house.cache_all_target()

    @property
    def info(self):
        ret = super(MultiHouseEnv, self).info
        ret['house_id'] = self.house._id
        return ret

    @property
    def num_house(self):
        return len(self.all_houses)

    def gen_2dmap(self, x=None, y=None, resolution=None):
        # TODO move cachedLocMap to House
        self.cachedLocMap = self.house._cachedLocMap
        retLocMap = super(MultiHouseEnv, self).gen_2dmap(x, y, resolution)
        if self.house._cachedLocMap is None:
            self.house._cachedLocMap = self.cachedLocMap
        return retLocMap

