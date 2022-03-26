from utils.download import register_resource
from functools import partial

def class_dataset_images_for_scene(context, scene_name):
    base_path = context.store_path

    import ai2thor.controller
    from itertools import product
    from collections import defaultdict
    import numpy as np
    import os
    import cv2
    import hashlib
    import json
    import shutil

    if os.path.exists(base_path):
        if os.path.isfile(os.path.join(base_path, '.complete')):
            return
        else:
            shutil.rmtree(base_path)
            os.makedirs(base_path)

    env = ai2thor.controller.Controller(quality='Low')
    player_size = 300
    zoom_size = 1000
    target_size = 256
    rotations = [0, 90, 180, 270]
    horizons = [330,  0, 30]
    buffer = 15
    # object must be at least 40% in view
    min_size = ((target_size * 0.4)/zoom_size) * player_size

    env.start(player_screen_width=player_size, player_screen_height=player_size)
    env.reset(scene_name)
    event = env.step(dict(action='Initialize', gridSize=0.25, renderObjectImage=True, renderClassImage=False, renderImage=False))

    for o in event.metadata['objects']:
        if o['receptacle'] and o['receptacleObjectIds'] and o['openable']:
            print("opening %s" % o['objectId'])
            env.step(dict(action='OpenObject', objectId=o['objectId'], forceAction=True))

    event = env.step(dict(action='GetReachablePositions', gridSize=0.25))
    
    visible_object_locations = []
    for point in event.metadata['actionReturn']:
        for rot, hor in product(rotations, horizons):
            exclude_colors = set(map(tuple, np.unique(event.instance_segmentation_frame[0], axis=0)))
            exclude_colors.update(set(map(tuple, np.unique(event.instance_segmentation_frame[:, -1, :], axis=0))))
            exclude_colors.update(set(map(tuple, np.unique(event.instance_segmentation_frame[-1], axis=0))))
            exclude_colors.update(set(map(tuple, np.unique(event.instance_segmentation_frame[:, 0, :], axis=0))))

            event = env.step(dict( action='TeleportFull', x=point['x'], y=point['y'], z=point['z'], rotation=rot, horizon=hor, forceAction=True), raise_for_failure=True) 

            visible_objects = []

            for o in event.metadata['objects']:

                if o['visible'] and o['objectId'] and o['pickupable']:
                    color = event.object_id_to_color[o['objectId']]
                    mask = (event.instance_segmentation_frame[:,:,0] == color[0]) & (event.instance_segmentation_frame[:,:,1] == color[1]) &\
                        (event.instance_segmentation_frame[:,:,2] == color[2])
                    points = np.argwhere(mask)

                    if len(points) > 0:
                        min_y = int(np.min(points[:,0]))
                        max_y = int(np.max(points[:,0]))
                        min_x = int(np.min(points[:,1]))
                        max_x = int(np.max(points[:,1]))
                        max_dim = max((max_y - min_y), (max_x - min_x))
                        if max_dim > min_size and min_y > buffer and min_x > buffer and max_x < (player_size - buffer) and max_y < (player_size - buffer):
                            visible_objects.append(dict(objectId=o['objectId'],min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y))
                            print("[%s] including object id %s %s" % (scene_name, o['objectId'], max_dim))




            if visible_objects:
                visible_object_locations.append(dict(point=point, rot=rot, hor=hor, visible_objects=visible_objects))

    env.stop()
    env = ai2thor.controller.Controller()
    env.start(player_screen_width=zoom_size, player_screen_height=zoom_size)
    env.reset(scene_name)
    event = env.step(dict(action='Initialize', gridSize=0.25, renderClassImage = True, renderDepthImage = True))

    for o in event.metadata['objects']:
        if o['receptacle'] and o['receptacleObjectIds'] and o['openable']:
            print("opening %s" % o['objectId'])
            env.step(dict(action='OpenObject', objectId=o['objectId'], forceAction=True))

    for vol in visible_object_locations:
        point = vol['point']
        
        event = env.step(dict( action='TeleportFull', x=point['x'], y=point['y'], z=point['z'],rotation=vol['rot'], horizon=vol['hor'], forceAction=True), raise_for_failure=True)
        for v in vol['visible_objects']:
            object_id = v['objectId']
            min_y = int(round(v['min_y'] * (zoom_size/player_size)))
            max_y = int(round(v['max_y'] * (zoom_size/player_size)))
            max_x = int(round(v['max_x'] * (zoom_size/player_size)))
            min_x = int(round(v['min_x'] * (zoom_size/player_size)))
            delta_y = max_y - min_y
            delta_x = max_x - min_x
            scaled_target_size = max(delta_x, delta_y, target_size) + buffer * 2
            if min_x > (zoom_size - max_x):
                start_x = min_x - (scaled_target_size - delta_x)
                end_x = max_x + buffer
            else:
                end_x = max_x + (scaled_target_size - delta_x )
                start_x = min_x - buffer

            if min_y > (zoom_size - max_y):
                start_y = min_y - (scaled_target_size - delta_y)
                end_y = max_y + buffer
            else:
                end_y = max_y + (scaled_target_size - delta_y)
                start_y = min_y - buffer

            #print("max x %s max y %s min x %s  min y %s" % (max_x, max_y, min_x, min_y))
            #print("start x %s start_y %s end_x %s end y %s" % (start_x, start_y, end_x, end_y))
            print("storing %s " % object_id)
            img = event.cv2img[start_y: end_y, start_x:end_x, :]
            seg_img = event.cv2img[min_y: max_y, min_x:max_x, :]
            dst = cv2.resize(img, (target_size, target_size), interpolation = cv2.INTER_LANCZOS4)
            dst_depth = (event.depth_frame[start_y: end_y, start_x:end_x] * 255.0 / 5000.0).astype(np.uint8)
            dst_depth = cv2.resize(dst_depth,
                (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            dst_semantic = cv2.resize(\
                event.class_segmentation_frame[start_y: end_y, start_x:end_x, :], \
                (target_size, target_size), interpolation=cv2.INTER_NEAREST)

            object_type = object_id.split('|')[0].lower()
            target_dir = os.path.join(base_path, 'images', object_type)
            h = hashlib.md5()
            h.update(json.dumps(point, sort_keys=True).encode('utf8'))
            h.update(json.dumps(v, sort_keys=True).encode('utf8'))

            os.makedirs(target_dir, exist_ok=True)
            cv2.imwrite(os.path.join(target_dir, h.hexdigest() + "-render_rgb.png"), dst)
            cv2.imwrite(os.path.join(target_dir, h.hexdigest() + "-render_depth.png"), dst_depth)
            cv2.imwrite(os.path.join(target_dir, h.hexdigest() + "-render_semantic.png"), dst_semantic)

    env.stop()
    open(os.path.join(base_path, '.complete'), 'a').close()
    return scene_name

for ordinal in [0, 200, 300, 400]:
    for i in range(1, 31):
        name = 'FloorPlan' + str(ordinal + i)
        register_resource('thor-scene-images-' + str(ordinal + i))(
            partial(class_dataset_images_for_scene, scene_name=name)
        )
        
