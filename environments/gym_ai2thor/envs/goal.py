from .env import EnvBase
from utils.download import resource as fetch_resource
import os
import cv2
import random
import numpy as np
from utils import download

DEFAULT_GOALS = [
    "ottoman", "laptop", "vase", "sofa", "plunger", "soapbottle", "apple", "knife", "ladle", "towel", "kettle", "bowl", "watch", "chair", "window", "potato", "safe", "spatula", "bottle", "boots", "cabinet", "handtowel", "laundryhamper", "tissuebox", "microwave", "painting", "pillow", "toiletpaperroll", "candle", "box", "bread", "cup", "egg", "toiletpaper", "lettuce", "television", "wateringcan", "spoon", "toaster", "plate", "winebottle", "cloth", "dresser", "stove burner", "televisionarmchair", "toilet", "drawer", "teddybear", "statue", "fridge", "pan", "alarmclock", "dishsponge", "shelf", "baseballbat", "stove knob", "sink", "coffeemachine", "garbagecan", "pot", "desklamp", "book", "scrubbrush", "houseplant", "poster", "pillowarmchair", "tennisracket", "towelholder", "mug"
]


class GoalImageCache:
    def __init__(self, image_size):
        self.scenes = dict()
        self.cache = dict()
        self.image_size = image_size
        self.random = random.Random()
        pass

    def seed(self, seed = None):
        self.random.seed(seed)

    def sample_image(self, collection):
        return self.random.choice([x[:-len('-render_rgb.png')] for x in collection if x.endswith('-render_rgb.png')])

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = os.path.join(download.downloader.resources_path, 'thor-scene-images-' + str(scene), 'images'),
                resources = dict()

            )
            
            sceneobj['available_goals'] = os.listdir(sceneobj['path'])

        return self.scenes[scene]

    def all_goals(self, scene):
        return self.fetch_scene(scene)['available_goals']

    def read_image(self, impath):
        try:
            image = cv2.imread(impath)
            image = cv2.resize(image, self.image_size, interpolation = cv2.INTER_CUBIC)
        except Exception as e:
            print('ERROR: wrong image %s' % impath)
            raise e
        return image

    def fetch_image(self, root, scene, resource, sampled_image):
        if (scene, resource, sampled_image) in self.cache:
            return self.cache[(scene, resource, sampled_image)]

        impath = os.path.join(root, sampled_image)
        assert os.path.isfile(impath), ('Missing file %s' % impath)
        image = self.read_image(impath)
        self.cache[(scene, resource, sampled_image)] = image
        return image

    def fetch_resource(self, scene, resource):
        self.fetch_scene(scene)
        if not resource in self.scenes[scene]['resources']:
            root = os.path.join(self.scenes[scene]['path'], resource)
            images = os.listdir(root)
            self.scenes[scene]['resources'][resource] = dict(
                root = root,
                images = images
            )
        else:
            row = self.scenes[scene]['resources'][resource]
            root, images = row['root'], row['images']

        return root, images

    def enumerate_all_paths(self):
        for d in os.listdir(download.downloader.resources_path):
            if not d.startswith('thor-scene-images-'):
                continue

            pdirectory = os.path.join(download.downloader.resources_path, d)
            for d2name in os.listdir(pdirectory):
                yield os.path.join(pdirectory, d2name)

    def all_image_paths(self, modes = ['rgb']):
        for d in self.enumerate_all_paths():
            for i in range(20):
                if not os.path.isfile(os.path.join(d, 'loc_%s-render_rgb.png' % i)):
                    continue

                ret = tuple()
                if 'rgb' in modes:
                    ret = ret + (os.path.join(d, 'loc_%s-render_rgb.png' % i),)

                if 'depth' in modes:
                    ret = ret + (os.path.join(d, 'loc_%s-render_depth.png' % i),)

                if 'segmentation' in modes:
                    ret = ret + (os.path.join(d, 'loc_%s-render_semantic.png' % i),)
                yield ret

    def all_images(self, modes = ['rgb']):
        for impaths in self.all_image_paths(modes):
            yield tuple(map(self.read_image, impaths))

    def fetch_random(self, scene, resource):
        root, images = self.fetch_resource(scene, resource)       
        sampled_image = self.sample_image(images)        
        return self.fetch_image(root, scene, resource, sampled_image + '-render_rgb.png'), os.path.join(root, sampled_image)

    def fetch_random_with_semantic(self, scene, resource):
        root, images = self.fetch_resource(scene, resource)       
        sampled_image = self.sample_image(images)        
        return (
            self.fetch_image(root, scene, resource, sampled_image + '-render_rgb.png'),
            self.fetch_image(root, scene, resource, sampled_image + '-render_semantic.png')
        ), os.path.join(root, sampled_image)

class GoalEnvBase(EnvBase):
    def __init__(self, scenes, screen_size = (224, 224), goals = [], **kwargs):
        if len(goals) == 0:
            goals = list(DEFAULT_GOALS)

        self.goal_source = GoalImageCache(screen_size)
        super().__init__(scenes, screen_size=screen_size, goals=goals, **kwargs)        
    
    def _has_finished(self, event):
        for o in event.metadata['objects']:
            tp = self._get_object_type(o)
            if tp == self.goal and o['distance'] < self.treshold_distance:
                return True
        
        return False

    def _render_goal(self, scene, goal):
        return self.goal_source.fetch_random(scene, goal)

    def _pick_goal(self, event, scene):
        allgoals = set(self.goal_source.all_goals(scene))
        allgoals.intersection_update(set(self.goals))            

        # Resamples if no goals are available
        event = super()._pick_goal(event, scene)

        goals = set()
        for o in event.metadata['objects']:
            tp = self._get_object_type(o)
            if tp in allgoals:
                goals.add(tp)

        self.goal = self.random.choice(list(goals))
        self.goal_observation = self._render_goal(scene, self.goal)
        return event

    def observe(self, event = None):
        main_observation = super().observe(event)
        return (main_observation, np.copy(self.goal_observation))

    def browse(self):
        from .browser import GoalKeyboardAgent
        return GoalKeyboardAgent(self)
