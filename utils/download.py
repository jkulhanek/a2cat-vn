import requests
import io
import zipfile
from collections import namedtuple
import os

def _convert_name(name, acc = ''):
    if len(name) == 0:
        return acc
    if name[0].isupper():
        acc += ('-' if len(acc) > 0 else '') + name[0].lower()
    elif name[0] == '_':
        acc += '-'
    else:
        acc += name[0]
    return _convert_name(name[1:], acc)

DownloaderContext = namedtuple('DownloaderContext', ['base_url', 'resources_path', 'store_path'])

class Downloader:
    def __init__(self):
        self.base_url = 'https://deep-rl.herokuapp.com/resources/'
        self.resources = dict()
        self._base_path = None
        self._all_requirements = []

    @property
    def base_path(self):
        if self._base_path is None:
            self._base_path = os.path.expanduser('~/.visual_navigation')
        return self._base_path

    @property
    def resources_path(self):
        return os.path.join(self.base_path, 'resources')

    def create_context(self, name):
        return DownloaderContext(self.base_url, self.resources_path, os.path.join(self.resources_path, name))

    def add_resource(self, name, fn):
        self.resources[name] = fn

    def require(self, name):
        self._all_requirements.append(name)

    def get(self, name):
        return self.resources[name](self.create_context(name))

    def download_all(self):
        for r in self._all_requirements:
            self.get(r)

downloader = Downloader()

def download_resource(name, context):
    resource_path = os.path.join(context.resources_path, name)
    if os.path.exists(resource_path):
        return resource_path

    url = context.base_url + '%s.zip' % name
    try:
        print('Downloading resource %s.' % name)  
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(resource_path)

        print('Resource %s downloaded.' %name)
        return resource_path

    except Exception as e:
        if os.path.exists(resource_path):
            os.remove(resource_path)
        raise e

def register_resource(task):
    if isinstance(task, str):
        def _thunk(res):
            downloader.add_resource(task, res)
            return res
        return _thunk

    name = _convert_name(task.__name__)
    downloader.add_resource(name, task)
    return task

def require_resource(name):
    downloader.require(name)
    return lambda x: x

def download_resource_task(name):
    def thunk(context):
        return download_resource(name, context)
    return thunk

def add_resources(downloader_instance):

    # Add test resource
    downloader_instance.add_resource('test', download_resource_task('test'))

add_resources(downloader)

def resource(name):
    return downloader.get(name)