from abc import ABC, abstractproperty, abstractclassmethod
import numpy as np
import cv2
from functools import partial

class GridWorldScene:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.graph = None
        self.optimal_distances = None

    @abstractproperty
    def maze(self):
        pass

    @abstractclassmethod
    def render(self, position, rotation):
        pass

    @property
    def observation_shape(self):
        return (81, 81, 3)

    @property
    def dtype(self):
        return np.uint8


def resize(image, size):
    if isinstance(image, tuple):
        return tuple(map(partial(resize, size = size), image))
    elif isinstance(image, list):
        return list(map(partial(resize, size = size), image))
    else:
        if len(image.shape) == 3 and image.shape[:2] != size:
            result = cv2.resize(image, size)
            if len(result.shape) == 2:
                result = np.expand_dims(result, 2)
            return result
        return image

class GraphResize:
    def __init__(self, graph, screen_size):
        self._graph = graph
        self._screen_size = screen_size

    def __getattr__(self, name):
        return getattr(self._graph, name)

    def render(self, *args, **kwargs):
        value = self._graph.render(*args, **kwargs)
        return resize(value, self._screen_size)

    @property
    def observation_shape(self):
        return self._screen_size + (self._graph.observation_shape[-1],)