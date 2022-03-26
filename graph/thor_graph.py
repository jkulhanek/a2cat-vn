# from graph.core import GridWorldScene
import numpy as np
import cv2

class ThorGridWorld:
    def __init__(self, maze, observations, depths, segmentations):
        self._observations = observations
        self._depths = depths
        self._segmentations = segmentations
        self._maze = maze

    def render(self, position, direction, modes = ['rgb']):
        ret = tuple()
        if 'rgb' in modes:
            ret = ret + (self._observations[position[0], position[1], direction],)
        if 'depth' in modes:
            depth = self._depths[position[0], position[1], direction]
            ret = ret + (depth,)
        if 'segmentation' in modes:
            ret = ret + (self._segmentations[position[0], position[1], direction],)
                
        if len(ret) == 1:
            return ret[0]

        return ret

    @property
    def maze(self):
        return self._maze

    @property
    def observation_shape(self):
        return self._observations.shape[-3:]

    @property
    def dtype(self):
        return np.uint8

class GridWorldReconstructor:
    def __init__(self, scene_name = 'FloorPlan28', grid_size = 0.5, env_kwargs = dict(), screen_size = (300,300,), seed = None, cameraY = 0.675):
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.env_kwargs = env_kwargs
        self.scene_name = scene_name
        self.cameraY = cameraY
        self.seed = seed

    def _initialize(self):
        import ai2thor.controller
        self._collected_positions = set()
        self._position = (0, 0)
        self._controller = ai2thor.controller.Controller()
        self._controller.start()
        self._controller.reset(self.scene_name)

        self._frames = dict()
        self._realcoordinates = dict()
        # gridSize specifies the coarseness of the grid that the agent navigates on
        self._controller.step(dict(action='Initialize', grid_size=self.grid_size, **self.env_kwargs, renderDepthImage = True, renderClassImage = True, cameraY = self.cameraY))
        self._controller.step(dict(action = 'InitialRandomSpawn', randomSeed = self.seed, forceVisible = False, maxNumRepeats = 5))

    def _compute_new_position(self, original, direction):
        dir1, dir2 = original
        if direction == 0:
            return (dir1 + 1, dir2)
        elif direction == 1:
            return (dir1, dir2 + 1)
        elif direction == 2:
            return (dir1 - 1, dir2)
        else:
            return (dir1, dir2 - 1)

    def _collect_spot(self, position):
        if position in self._collected_positions:
            return

        self._collected_positions.add(position)
        print('collected '  + str(position))

        frames = [None, None, None, None]

        # Collect all four images in all directions
        for d in range(4):
            event = self._controller.step(dict(action='RotateRight'))
            depth = np.expand_dims((event.depth_frame * 255 / 5000).astype(np.uint8), 2)
            frames[(1 + d) % 4] = (event.frame, depth, event.class_segmentation_frame,)

        self._realcoordinates[position] = event.metadata['agent']['position']
        self._frames[position] = frames

        # Collect frames in all four dimensions
        newposition = self._compute_new_position(position, 0)
        if not newposition in self._collected_positions:
            event = self._controller.step(dict(action='MoveAhead'))
            if event.metadata.get('lastActionSuccess'):
                self._collect_spot(newposition)
                event = self._controller.step(dict(action = 'MoveBack'))

        newposition = self._compute_new_position(position, 1)
        if not newposition in self._collected_positions:
            event = self._controller.step(dict(action='MoveRight'))
            if event.metadata.get('lastActionSuccess'):
                self._collect_spot(newposition)
                event = self._controller.step(dict(action = 'MoveLeft'))

        newposition = self._compute_new_position(position, 2)
        if not newposition in self._collected_positions:
            event = self._controller.step(dict(action='MoveBack'))
            if event.metadata.get('lastActionSuccess'):
                self._collect_spot(newposition)
                event = self._controller.step(dict(action = 'MoveAhead'))

        newposition = self._compute_new_position(position, 3)
        if not newposition in self._collected_positions:
            event = self._controller.step(dict(action='MoveLeft'))
            if event.metadata.get('lastActionSuccess'):
                self._collect_spot(newposition)
                event = self._controller.step(dict(action = 'MoveRight'))

    def resize(self, image):
        if self.screen_size != (300,300):
            ret = cv2.resize(image, self.screen_size)
            if len(ret.shape) == 2:
                ret = np.expand_dims(ret, 2)
            return ret

        return image

    def _compile(self):
        minx = min(self._frames.keys(), default = 0, key = lambda x: x[0])[0]
        miny = min(self._frames.keys(), default = 0, key = lambda x: x[1])[1]
        maxx = max(self._frames.keys(), default = 0, key = lambda x: x[0])[0]
        maxy = max(self._frames.keys(), default = 0, key = lambda x: x[1])[1]

        size = (maxx - minx + 1, maxy - miny + 1)
        observations = np.zeros(size + (4,) + self.screen_size +(3,), dtype = np.uint8)
        segmentations = np.zeros(size + (4,) + self.screen_size +(3,), dtype = np.uint8)
        depths = np.zeros(size + (4,) + self.screen_size +(1,), dtype = np.uint8)
        grid = np.zeros(size, dtype = np.bool)
        for key, value in self._frames.items():
            for i in range(4):
                observations[key[0] - minx, key[1] - miny, i] = self.resize(value[i][0])
                depths[key[0] - minx, key[1] - miny, i] = self.resize(value[i][1])
                segmentations[key[0] - minx, key[1] - miny, i] = self.resize(value[i][2])
            grid[key[0] - minx, key[1] - miny] = 1

        return ThorGridWorld(grid, observations, depths, segmentations)

    def __del__(self):
        if hasattr(self, '_controller') and self._controller is not None:
            self._controller.stop()

    def reconstruct(self):
        self._initialize()
        self._controller.step(dict(action = 'RotateLeft'))
        self._collect_spot((0, 0))
        return self._compile()