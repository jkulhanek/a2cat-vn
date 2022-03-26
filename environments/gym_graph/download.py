import os
from os.path import expanduser
from graph.util import load_graph, dump_graph

def thor_generator(scene, screen_size, goals, seed = 1, grid_size = 0.5, cameraY = 0.675):
    def _thunk():
        import ai2thor.controller
        import graph.thor_graph
        reconstructor = graph.thor_graph.GridWorldReconstructor(scene, screen_size = screen_size, seed = seed, grid_size=grid_size, cameraY=cameraY)
        graph = reconstructor.reconstruct()
        graph.goals = goals
        return graph
    return _thunk


graph_generators = {}
'''    'kitchen-224': thor_generator('FloorPlan28', (224, 224,), (7, 0)),
    'kitchen-84': thor_generator('FloorPlan28', (84, 84,), (7, 0))
}'''

graph_generators['thor-cached-212'] = thor_generator('FloorPlan212', (300, 300), grid_size = 0.33, cameraY = 0.3, goals = [(3, 1, 2), (13, 21, 3), (10, 2, 1), (10, 14, 0)])
graph_generators['thor-cached-208'] = thor_generator('FloorPlan208', (300, 300), grid_size = 0.33, cameraY = 0.3, goals = [(6, 3, 1), (13, 3, 0), (7, 18, 2), (6, 25, 1)])
graph_generators['thor-cached-218'] = thor_generator('FloorPlan218', (300, 300), grid_size = 0.33, cameraY = 0.3, goals = [(6, 22, 1), (7, 0, 0), (18, 18, 3), (13, 31, 3)])
graph_generators['thor-cached-225'] = thor_generator('FloorPlan225', (300, 300), grid_size = 0.33, cameraY = 0.3, goals = [(3, 17, 2), (12, 17, 3), (15, 10, 0), (14, 8, 3)])

graph_generators['thor-cached-212-174'] = thor_generator('FloorPlan212', (174, 174), grid_size = 0.33, cameraY = 0.3, goals = [(3, 1, 2), (13, 21, 3), (10, 2, 1), (10, 14, 0)])
graph_generators['thor-cached-208-174'] = thor_generator('FloorPlan208', (174, 174), grid_size = 0.33, cameraY = 0.3, goals = [(6, 3, 1), (13, 3, 0), (7, 18, 2), (6, 25, 1)])
graph_generators['thor-cached-218-174'] = thor_generator('FloorPlan218', (174, 174), grid_size = 0.33, cameraY = 0.3, goals = [(6, 22, 1), (7, 0, 0), (18, 18, 3), (13, 31, 3)])
graph_generators['thor-cached-225-174'] = thor_generator('FloorPlan225', (174, 174), grid_size = 0.33, cameraY = 0.3, goals = [(3, 17, 2), (12, 17, 3), (15, 10, 0), (14, 8, 3)])

def _to_pascal(text):
    return ''.join(map(lambda x: x.capitalize(), text.split('-')))

def available_scenes():
    return [(_to_pascal(x), x) for x in graph_generators.keys()]

def get_graph(graph):
    home = expanduser("~")
    basepath = os.path.expanduser('~/.cache/visual-navigation/datasets')
    os.makedirs(basepath, exist_ok=True)

    filename = os.path.join(basepath, '%s.pkl' % graph)
    if not os.path.exists(filename):
        graph = graph_generators.get(graph)()        
        with open(filename, 'wb+') as f:
            dump_graph(graph, f)
            f.flush()

    with open(filename, 'rb') as f:
        graph = load_graph(f)

    return graph

def download_all():
    for graph in graph_generators.keys():
        get_graph(graph)
