import os
import random
import cv2
from itertools import count

class GoalImageCache:
    def __init__(self, image_size, dataset_path):
        self.scenes = dict()
        self.cache = dict()
        self.image_size = image_size
        self.dataset_path = os.path.join(dataset_path, 'render')
        self.random = random.Random()
        pass

    def seed(self, seed = None):
        self.random.seed(seed)

    def sample_image(self, collection):
        return self.random.choice([x[:-len('-render_rgb.png')] for x in collection if x.endswith('-render_rgb.png')])

    def fetch_scene(self, scene):
        if not scene in self.scenes:
            self.scenes[scene] = sceneobj = dict(
                path = os.path.join(self.dataset_path, scene),
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
        for d in os.listdir(self.dataset_path):
            pdirectory = os.path.join(self.dataset_path, d)
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