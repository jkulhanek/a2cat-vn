import numpy as np
from operator import add

def direction_to_change(direction):
    if direction == 0:
        return (1, 0)
    elif direction == 1:
        return (0, 1)
    elif direction == 2:
        return (-1, 0)
    elif direction == 3:
        return (0, -1)
    raise Exception('Unsopported direction %s' % direction)

def step(state, action):
    if action == 0:
        change = direction_to_change(state[2])
        return (state[0] + change[0], state[1] + change[1], state[2])
    elif action == 2:
        change = direction_to_change(state[2])
        return (state[0] - change[0], state[1] - change[1], state[2])
    elif action == 1:
        return (state[0], state[1], (state[2] + 1) % 4)
    elif action == 3:
        return (state[0], state[1], (state[2] + 3) % 4)

def enumerate_positions(maze):
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y]:
                yield (x, y)

def find_state(maze):
    return next(enumerate_positions) + (0,)

def is_valid_state(maze, state):
    return state[0] >= 0 and state[1] >= 0 and state[0] < maze.shape[0] and state[1] < maze.shape[1] and maze[state[0], state[1]]


def dump_graph(graph, file):
    shortest_path_distances, actions = compute_shortest_path_data(graph.maze)
    graph.graph = shortest_path_distances
    graph.optimal_actions = actions
    import pickle
    pickle.dump(graph, file)

def load_graph(file):
    import pickle
    
    if isinstance(file, str):
        with open(file, 'rb') as f:
            graph = pickle.load(f)
    else:
        graph = pickle.load(file)
    if not hasattr(graph, 'graph') or graph.graph is None:
        graph.graph, graph.optimal_actions = compute_shortest_path_data(graph.maze)
    return graph


def compute_rotation_steps(graph, goal, state):
    optimal_action = graph.optimal_actions[state[:2] + goal[:2]]
    rot_steps = np.array(list(map(lambda x: (state[2] - (goal[2] + x)) % 4, np.where(optimal_action))))
    rot_steps[rot_steps == 3] = 1
    return np.min(rot_steps)

def sample_initial_position(graph, goal, optimal_distance = None):
    potentials = []
    distances = []
    for position in enumerate_positions(graph.maze):
        d = graph.graph[position + goal]
        if d > 0:
            potentials.append(position)
            distances.append(d)


    if optimal_distance is None:
        x = None
        while x is None or potentials[x] == goal:
            x = np.random.choice(np.arange(len(potentials)))
    else:
        distances = np.array(distances)
        positive = distances <= optimal_distance
        negative = distances > optimal_distance
        sum_negative = np.sum(negative)
        if sum_negative == 0:
            weights = positive / np.sum(positive)
        else:
            positive = 0.9 * positive / np.sum(positive)
            negative = 0.1 * negative / sum_negative
            weights = positive + negative

        x = None
        while x is None or potentials[x] == goal:
            x = np.random.choice(np.arange(len(potentials)), p = weights)
    return potentials[x]

def sample_initial_state(graph, goal, optimal_distance = None):
    potentials = []
    distances = []
    for position in enumerate_positions(graph.maze):
        d = graph.graph[position + goal[:2]]
        if d > 0:
            for i in range(4):
                state = position + (i,)
                potentials.append(state)
                distances.append(d + compute_rotation_steps(graph, goal, state))


    if optimal_distance is None:
        x = np.random.choice(np.arange(len(potentials)))
    else:
        distances = np.array(distances)
        positive = distances <= optimal_distance
        #negative = distances > optimal_distance
        #positive = 0.9 * positive / np.sum(positive)
        #negative = 0.1 * negative / np.sum(negative)
        #weights = positive + negative
        weights = positive / np.sum(positive)

        x = np.random.choice(np.arange(len(potentials)), p = weights)
    return potentials[x]


def compute_shortest_path_data(maze):
    distances = np.ndarray(maze.shape + maze.shape, dtype = np.int32)
    actions = np.ndarray(maze.shape + maze.shape + (4,), dtype = np.bool)
    distances.fill(-1)
    actions.fill(False)
    def fill_shortest_path(goal, position, distance, from_direction):
        if not is_valid_state(maze, position):
            return 

        if distances[position + goal] != -1 and distances[position + goal] < distance:
            return

        if distances[position + goal] == distance:
            actions[position + goal + ((from_direction + 2) % 4,)] = True
            return            
        
        actions[position + goal] = False
        actions[position + goal + ((from_direction + 2) % 4,)] = True
        distances[position + goal] = distance

        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(0))), distance + 1, 0)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(1))), distance + 1, 1)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(2))), distance + 1, 2)
        fill_shortest_path(goal, tuple(map(add, position, direction_to_change(3))), distance + 1, 3)
        

    for goal in enumerate_positions(maze):
        fill_shortest_path(goal, goal, 0, 0)
        actions[goal + goal] = False

    return distances, actions


def create_resnet():
    from torchvision.models.resnet import resnet50
    import torch
    import cv2
    model = resnet50(pretrained=True)
    self = model
    def forward(x):
        x = cv2.resize(x, (224, 224), interpolation = cv2.INTER_CUBIC)
        x = torch.from_numpy(np.transpose(x, [2, 0, 1]).astype(np.float32) / 255.0).unsqueeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(-1).detach().numpy()
    model.eval()
    return forward

def save_graph_as_h5(graph, path):
    import h5py

    if not hasattr(graph, 'graph') or graph.graph is None:
        graph.graph, graph.optimal_actions = compute_shortest_path_data(graph.maze)

    locations = list(enumerate_positions(graph.maze))
    locations_lookup = { key: i for i, key in enumerate(locations) }
    num_locations = len(locations) * 4

    def compute_graph_line(point, rotation):
        def add_rotation(point):
            if point == -1:
                return point
            return point * 4 + rotation
        return [add_rotation(locations_lookup.get(tuple(map(lambda x, y: x+y, direction_to_change(rotation), point)), -1)),
            add_rotation(locations_lookup.get(tuple(map(lambda x, y: x+y, direction_to_change((rotation + 2) % 4), point)), -1))]

    resnet = create_resnet()

    with h5py.File(path, 'w') as file:
        graph_dataset = file.create_dataset('graph', (num_locations, 4), np.int)
        location_dataset = file.create_dataset('location', (num_locations, 2), np.float)
        observation_dataset = file.create_dataset('observation', (num_locations,) + graph.observation_shape, np.uint8)
        resnet_feature_dataset = file.create_dataset('resnet_feature', (num_locations, 2048), np.float32)
        shortest_path_distance_dataset = file.create_dataset('shortest_path_distance', (num_locations, num_locations), np.int64)

        for point, location in enumerate(locations):
            for rotation in range(4):
                actions = compute_graph_line(location, rotation) + \
                    [point * 4 + (rotation + 1) % 4, point * 4 + (rotation - 1) % 4]

                observation = graph.render(location, rotation)
                graph_dataset[point * 4 + rotation,:] = np.array(actions, dtype=np.int)
                location_dataset[point * 4 + rotation,...] = location
                observation_dataset[point * 4 + rotation,...] = observation
                resnet_feature_dataset[point * 4 + rotation, ...] = resnet(observation)

            for point2, location2 in enumerate(locations):
                base_distance = graph.graph[location + location2]
                for rotation in range(4):
                    for rotation2 in range(4):
                        rotation_diff = abs(rotation - rotation2)
                        if rotation_diff == 3: 
                            rotation_diff = 1
                        shortest_path_distance_dataset[point * 4 + rotation, point2 * 4 + rotation2] = base_distance + rotation_diff


            



