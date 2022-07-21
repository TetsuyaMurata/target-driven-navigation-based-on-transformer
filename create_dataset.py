import argparse
import gc
import json
import os
import re
from collections import namedtuple

import ai2thor.controller
import h5py
import numpy as np
import spacy
# import tensorflow as tf # before
import tensorflow.compat.v1 as tf # after
import torch
# from keras.applications import resnet50 #before
from tensorflow.keras.applications import resnet50 #after
# from keras.backend.tensorflow_backend import set_session #before
from tensorflow.python.keras.backend import set_session #after
from tensorflow.python.keras import backend as K # add
from PIL import Image
from tqdm import tqdm
import random #add

import torchvision.models as models
import torchvision.transforms as transforms
from pytorchyolo3.darknet import Darknet
from pytorchyolo3.models.tiny_yolo import TinyYoloNet
from pytorchyolo3.utils import *

scene_type = []
SCENES = [0, 200, 300, 400]
TRAIN_SPLIT = (1, 22)
TEST_SPLIT = (22, 27)

KITCHEN_OBJECT_CLASS_LIST = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMachine",
    "GarbageCan",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Bowl",
]

BEDROOM_OBJECT_CLASS_LIST = ["HousePlant", "Lamp", "Book", "AlarmClock"]


BATHROOM_OBJECT_CLASS_LIST = [
    "Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]
SCENE_TASKS = [KITCHEN_OBJECT_CLASS_LIST, LIVING_ROOM_OBJECT_CLASS_LIST,
               BEDROOM_OBJECT_CLASS_LIST, BATHROOM_OBJECT_CLASS_LIST]


def construct_scene_names():
    names = []
    for idx, scene in enumerate(SCENES):
        for t in range(*TRAIN_SPLIT):
            names.append("FloorPlan" + str(scene + t))
            scene_type.append(idx)
        for t in range(*TEST_SPLIT):
            names.append("FloorPlan" + str(scene + t))
            scene_type.append(idx)
    return names, scene_type


grid_size = 0.5

actions = ["MoveAhead", "RotateRight", "RotateLeft",
           "MoveBack", "LookUp", "LookDown", "MoveRight", "MoveLeft"]
rotation_possible_inplace = 8 #4(90degree), 8(45degree)
ACTION_SIZE = len(actions)
StateStruct = namedtuple(
    "StateStruct", "id pos rot obs semantic_obs feat feat_place bbox obj_visible")

# Extracted from unity/Assets/Scripts/SimObjType.cs
OBJECT_IDS = {
    "Undefined": 0,
    "Apple": 1,
    "AppleSliced": 2,
    "Tomato": 3,
    "TomatoSliced": 4,
    "Bread": 5,
    "BreadSliced": 6,
    "Sink": 7,
    "Pot": 8,
    "Pan": 9,
    "Knife": 10,
    "Fork": 11,
    "Spoon": 12,
    "Bowl": 13,
    "Toaster": 14,
    "CoffeeMachine": 15,
    "Microwave": 16,
    "StoveBurner": 17,
    "Fridge": 18,
    "Cabinet": 19,
    "Egg": 20,
    "Chair": 21,
    "Lettuce": 22,
    "Potato": 23,
    "Mug": 24,
    "Plate": 25,
    "TableTop": 26,
    "CounterTop": 27,
    "GarbageCan": 28,
    "Omelette": 29,
    "EggShell": 30,
    "EggCracked": 31,
    "StoveKnob": 32,
    "Container": 33,
    "Cup": 34,
    "ButterKnife": 35,
    "PotatoSliced": 36,
    "MugFilled": 37,
    "BowlFilled": 38,
    "Statue": 39,
    "LettuceSliced": 40,
    "ContainerFull": 41,
    "BowlDirty": 42,
    "Sandwich": 43,
    "Television": 44,
    "HousePlant": 45,
    "TissueBox": 46,
    "VacuumCleaner": 47,
    "Painting": 48,
    "WateringCan": 49,
    "Laptop": 50,
    "RemoteControl": 51,
    "Box": 52,
    "Newspaper": 53,
    "TissueBoxEmpty": 54,
    "PaintingHanger": 55,
    "KeyChain": 56,
    "Dirt": 57,
    "CellPhone": 58,
    "CreditCard": 59,
    "Cloth": 60,
    "Candle": 61,
    "Toilet": 62,
    "Plunger": 63,
    "Bathtub": 64,
    "ToiletPaper": 65,
    "ToiletPaperHanger": 66,
    "SoapBottle": 67,
    "SoapBottleFilled": 68,
    "SoapBar": 69,
    "ShowerDoor": 70,
    "SprayBottle": 71,
    "ScrubBrush": 72,
    "ToiletPaperRoll": 73,
    "Lamp": 74,
    "LightSwitch": 75,
    "Bed": 76,
    "Book": 77,
    "AlarmClock": 78,
    "SportsEquipment": 79,
    "Pen": 80,
    "Pencil": 81,
    "Blinds": 82,
    "Mirror": 83,
    "TowelHolder": 84,
    "Towel": 85,
    "Watch": 86,
    "MiscTableObject": 87,
    "ArmChair": 88,
    "BaseballBat": 89,
    "BasketBall": 90,
    "Faucet": 91,
    "Boots": 92,
    "Glassbottle": 93,
    "DishSponge": 94,
    "Drawer": 95,
    "FloorLamp": 96,
    "Kettle": 97,
    "LaundryHamper": 98,
    "LaundryHamperLid": 99,
    "Lighter": 100,
    "Ottoman": 101,
    "PaintingSmall": 102,
    "PaintingMedium": 103,
    "PaintingLarge": 104,
    "PaintingHangerSmall": 105,
    "PaintingHangerMedium": 106,
    "PaintingHangerLarge": 107,
    "PanLid": 108,
    "PaperTowelRoll": 109,
    "PepperShaker": 110,
    "PotLid": 111,
    "SaltShaker": 112,
    "Safe": 113,
    "SmallMirror": 114,
    "Sofa": 115,
    "SoapContainer": 116,
    "Spatula": 117,
    "TeddyBear": 118,
    "TennisRacket": 119,
    "Tissue": 120,
    "Vase": 121,
    "WallMirror": 122,
    "MassObjectSpawner": 123,
    "MassScale": 124,
    "Footstool": 125,
    "Shelf": 126,
    "Dresser": 127,
    "Desk": 128,
    "NightStand": 129,
    "Pillow": 130,
    "Bench": 131,
    "Cart": 132,
    "ShowerGlass": 133,
    "DeskLamp": 134,
    "Window": 135,
    "BathtubBasin": 136,
    "SinkBasin": 137,
    "CD": 138,
    "Curtains": 139,
    "Poster": 140,
    "HandTowel": 141,
    "HandTowelHolder": 142,
    "Ladle": 143,
    "WineBottle": 144,
    "ShowerCurtain": 145,
    "ShowerHead": 146

}


def equal(s1, s2):
    if s1.pos["x"] == s2.pos["x"] and s1.pos["z"] == s2.pos["z"]:
        if s1.rot == s2.rot:
            return True
    return False


def search_namedtuple(list_states, search_state):
    for s in list_states:
        if equal(s, search_state):
            return s
    return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def create_states(h5_file, resnet_trained, resnet_places, controller, name, args, scene_type):
    # Reset the environnment
    controller.reset(name)

    # gridSize specifies the coarseness of the grid that the agent navigates on
    state = controller.step(
        dict(action='Initialize', gridSize=grid_size, renderObjectImage=True, renderClassImage=True))

    it = 0
    while it < 5:
        if args['eval']:
            state = controller.step(dict(action='InitialRandomSpawn',
                                         randomSeed=100, forceVisible=True, maxNumRepeats=30))
        else:
            state = controller.step(dict(action='InitialRandomSpawn',
                                        #  randomSeed=200, forceVisible=True, maxNumRepeats=30)) #origin
                                         randomSeed=random.randint(0, int(1e9)), forceVisible=True, maxNumRepeats=30)) #random object position

        # Check that every object is in scene
        scene_task = SCENE_TASKS[scene_type]
        obj_present = [False for i in scene_task]
        for obj in state.metadata['objects']:
            objectId = obj['objectId']
            obj_name = objectId.split('|')[0]
            # print("objectId : {0}, obj_name : {1}".format(objectId, obj_name)) #test
            for idx, _ in enumerate(obj_present):
                if obj_name == scene_task[idx]:
                    obj_present[idx] = True

        if np.all(obj_present):
            break
        else:
            it = it + 1
    # print(it)
    # Store available objects
    available_obj = set()
    for obj in state.metadata['objects']:
        objectId = obj['objectId']
        obj_name = objectId.split('|')[0]
        available_obj.add(obj_name)
    available_obj = list(available_obj)
    # print("Obj available", available_obj)

    h5_file.attrs["task_present"] = np.string_(
        json.dumps(available_obj, cls=NumpyEncoder))

    reachable_pos = controller.step(dict(
        action='GetReachablePositions', gridSize=grid_size)).metadata['reachablePositions']

    states = []
    obss = []
    idx = 0
    # Does not redo if already existing
    if args['force'] or \
        ('resnet_feature' not in h5_file.keys() and not args['view']) or \
        'observation' not in h5_file.keys() or \
        'location' not in h5_file.keys() or \
        'rotation' not in h5_file.keys() or \
        'bbox' not in h5_file.keys() or \
            ('semantic_obs' not in h5_file.keys() and not args['view']):
        for pos in tqdm(reachable_pos, desc="Feature extraction", position=1):
            state = controller.step(dict(action='Teleport', **pos))
            # Normal/Up/Down view
            for i in range(3):
                # Up view
                if i == 1:
                    state = controller.step(dict(action="LookUp"))
                # Down view
                elif i == 2:
                    state = controller.step(dict(action="LookDown"))
                # Rotate
                for a in range(rotation_possible_inplace):
                    state = controller.step(dict(action="RotateLeft"))
                    state.metadata['agent']['rotation']['z'] = state.metadata['agent']['cameraHorizon']
                    feature = None
                    feature_place = None
                    if ('resnet_feature' not in h5_file.keys() or args['force']) and not args['view']:
                        obs_process = resnet50.preprocess_input(state.frame)
                        obs_process = obs_process[np.newaxis, ...]

                        # Extract resnet feature from observation
                        feature = resnet_trained.predict(obs_process)

                        # Extract resnet place feature from observation
                        input_place = torch.from_numpy(
                            state.frame.copy()/255.0)
                        input_place = input_place.to(
                            "cuda", dtype=torch.float32)
                        input_place = input_place/255
                        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                        input_place = input_place.unsqueeze(0)
                        input_place = input_place.permute(0, 3, 2, 1)
                        feature_place = resnet_places(input_place)
                        feature_place = feature_place.cpu().detach().numpy()
                        feature_place = feature_place.squeeze()[
                            np.newaxis, ...]

                    # Store visible objects from the agent (visible = 1m away from the agent)
                    obj_visible = [obj['objectId']
                                   for obj in state.metadata['objects'] if obj['visible']]
                    state_struct = StateStruct(
                        idx,
                        state.metadata['agent']['position'],
                        state.metadata['agent']['rotation'],
                        obs=state.frame,
                        semantic_obs=state.class_segmentation_frame,
                        feat=feature,
                        feat_place=feature_place,
                        bbox=json.dumps(
                            state.instance_detections2D, cls=NumpyEncoder),
                        obj_visible=json.dumps(obj_visible))

                    if search_namedtuple(states, state_struct):
                        print("Already exists")
                        # exit()
                    else:
                        states.append(state_struct)
                        idx = idx + 1

                # Reset camera
                if i == 1:
                    state = controller.step(dict(action="LookDown"))
                elif i == 2:
                    state = controller.step(dict(action="LookUp"))

        # Save it to h5 file
        if args['force'] or 'resnet_feature' not in h5_file.keys() and not args['view']:
            if 'resnet_feature' in h5_file.keys():
                del h5_file['resnet_feature']
            h5_file.create_dataset(
                'resnet_feature', data=[s.feat for s in states])

        if args['force'] or 'observation' not in h5_file.keys():
            if 'observation' in h5_file.keys():
                del h5_file['observation']
            h5_file.create_dataset(
                'observation', data=[s.obs for s in states])

        if args['force'] or 'location' not in h5_file.keys():
            if 'location' in h5_file.keys():
                del h5_file['location']
            h5_file.create_dataset(
                'location', data=[list(s.pos.values()) for s in states])

        if args['force'] or 'rotation' not in h5_file.keys():
            if 'rotation' in h5_file.keys():
                del h5_file['rotation']
            h5_file.create_dataset(
                'rotation', data=[list(s.rot.values()) for s in states])

        if args['force'] or 'bbox' not in h5_file.keys():
            if 'bbox' in h5_file.keys():
                del h5_file['bbox']
            h5_file.create_dataset(
                'bbox', data=[s.bbox.encode("ascii", "ignore") for s in states])

        if args['force'] or 'object_visibility' not in h5_file.keys():
            if 'object_visibility' in h5_file.keys():
                del h5_file['object_visibility']
            h5_file.create_dataset(
                'object_visibility', data=[s.obj_visible.encode("ascii", "ignore") for s in states])

        if args['force'] or 'semantic_obs' not in h5_file.keys() and not args['view']:
            if 'semantic_obs' in h5_file.keys():
                del h5_file['semantic_obs']
            h5_file.create_dataset(
                'semantic_obs', data=[s.semantic_obs for s in states])

        return states
    else:
        ind_axis = ['x', 'y', 'z']
        for idx, _ in enumerate(h5_file['location']):
            pos = dict()
            rot = dict()
            for i, _ in enumerate(h5_file['location'][idx]):
                pos[ind_axis[i]] = h5_file['location'][idx][i]
                rot[ind_axis[i]] = h5_file['rotation'][idx][i]
            state_struct = StateStruct(
                idx,
                pos=pos,
                rot=rot,
                obs=None,
                semantic_obs=None,
                feat=None,
                feat_place=None,
                bbox=None,
                obj_visible=None)
            states.append(state_struct)
        return states


def create_graph(h5_file, states, controller, args):
    num_states = len(states)
    graph = np.full((num_states, ACTION_SIZE), -1)
    # Speed improvement
    state = controller.step(
        dict(action='Initialize', gridSize=grid_size, renderObjectImage=False))
    # Populate graph
    if args['force'] or 'graph' not in h5_file.keys():
        for state in tqdm(states, desc="Graph construction", position=1):
            for i, a in enumerate(actions):
                controller.step(dict(action='TeleportFull', **state.pos,
                                     rotation=state.rot['y'], horizon=state.rot['z']))
                state_controller = controller.step(dict(action=a))
                state_controller.metadata['agent']['rotation']['z'] = state_controller.metadata['agent']['cameraHorizon']
                # Convert to search
                state_controller_named = StateStruct(-1,
                                                     state_controller.metadata['agent']['position'],
                                                     state_controller.metadata['agent']['rotation'],
                                                     obs=None,
                                                     semantic_obs=None,
                                                     feat=None,
                                                     feat_place=None,
                                                     bbox=None,
                                                     obj_visible=None)

                if not equal(state, state_controller_named) and not round(state_controller.metadata['agent']['cameraHorizon']) == 60:
                    found = search_namedtuple(
                        states, state_controller_named)
                    if found is None:
                        # print([(s.pos, s.rot) for s in states])
                        # print(state_controller_named)
                        print("Error, state not found")
                        continue
                    graph[state.id][i] = found.id
        if 'graph' in h5_file.keys():
            del h5_file['graph']

        h5_file.create_dataset(
            'graph', data=graph)
        return graph
    else:
        return h5_file['graph']


def write_object_feature(h5_file, object_feature, object_vector, object_vector_visualgenome):
    # Write object_feature (resnet features)
    if 'object_feature' in h5_file.keys():
        del h5_file['object_feature']
    h5_file.create_dataset(
        'object_feature', data=object_feature)

    # Write object_vector (word embedding features)
    if 'object_vector' in h5_file.keys():
        del h5_file['object_vector']
    h5_file.create_dataset(
        'object_vector', data=object_vector)
    # Write object_vector (word embedding features)
    if 'object_vector_visualgenome' in h5_file.keys():
        del h5_file['object_vector_visualgenome']
    h5_file.create_dataset(
        'object_vector_visualgenome', data=object_vector_visualgenome)

    h5_file.attrs["object_ids"] = np.string_(json.dumps(OBJECT_IDS))


def extract_word_emb_vector(nlp, word_name):
    # Usee scapy to extract word embedding vector
    word_vec = nlp(word_name.lower())

    # If words don't exist in dataset
    # cut them using uppercase letter (SoapBottle -> Soap Bottle)
    if word_vec.vector_norm == 0:
        word = re.sub(r"(?<=\w)([A-Z])", r" \1", word_name)
        word_vec = nlp(word.lower())

        # If no embedding found try to cut word to find embedding (SoapBottle -> [Soap, Bottle])
        if word_vec.vector_norm == 0:
            word_split = re.findall('[A-Z][^A-Z]*', word)
            for word in word_split:
                word_vec = nlp(word.lower())
                if word_vec.has_vector:
                    break
            if word_vec.vector_norm == 0:
                print('ERROR: %s not found' % word_name)
                return None
    norm_word_vec = word_vec.vector / word_vec.vector_norm  # Normalize vector size
    return norm_word_vec


def extract_object_feature(resnet_trained, h, w):
    # Use scapy to extract vector from word embeddings
    nlp = spacy.load('en_core_web_lg')  # Use en_core_web_lg for more words
    nlp_visual = spacy.load('./word2vec_visualgenome/visualgenome_spacy')

    # Use glob to list object image
    import glob

    # 2048 is the resnet feature size
    object_feature = np.zeros((len(OBJECT_IDS), 2048), dtype=np.float32)
    # 300 is the word embeddings feature size
    object_vector = np.zeros((len(OBJECT_IDS), 300), dtype=np.float32)
    object_vector_visualgenome = np.zeros((len(OBJECT_IDS), 300), dtype=np.float32)
    # List all jpg files in data/objects/
    for filepath in glob.glob('data/objects/*.jpg'):

        # Resize image to be the same as observation (300x400)
        frame = Image.open(filepath)
        frame = frame.resize((w, h))
        frame = np.asarray(frame, dtype="int32")

        # Use resnet to extract object features
        obj_process = resnet50.preprocess_input(frame)
        obj_process = obj_process[np.newaxis, ...]
        feature = resnet_trained.predict(obj_process)

        filename = os.path.splitext(os.path.basename(filepath))[0]
        object_feature[OBJECT_IDS[filename]] = feature

    for object_name, object_id in OBJECT_IDS.items():
        norm_word_vec = extract_word_emb_vector(nlp, object_name)
        if norm_word_vec is None:
            print("Spacy no we for", object_name)
        object_vector[object_id] = norm_word_vec

        norm_word_vec_vg = extract_word_emb_vector(nlp_visual, object_name)
        if norm_word_vec_vg is None:
            print("Visual genome no we for", object_name)
        object_vector_visualgenome[object_id] = norm_word_vec_vg

    return object_feature, object_vector, object_vector_visualgenome


def create_shortest_path(h5_file, states, graph):
    # Usee network to compute shortest path
    import networkx as nx
    from networkx.readwrite import json_graph
    num_states = len(states)
    G = nx.Graph()
    shortest_dist_graph = np.full((num_states, num_states), -1)

    for state in states:
        G.add_node(state.id)
    for state in states:
        for i, a in enumerate(actions):
            if graph[state.id][i] != -1:
                G.add_edge(state.id, graph[state.id][i])
    shortest_path = nx.shortest_path(G)

    for state_id_src in range(num_states):
        for state_id_dst in range(num_states):
            try:
                shortest_dist_graph[state_id_src][state_id_dst] = len(
                    shortest_path[state_id_src][state_id_dst]) - 1
            except KeyError:
                # No path between states
                print(state_id_src, state_id_dst)
                shortest_dist_graph[state_id_src][state_id_dst] = -1

    if 'shortest_path_distance' in h5_file.keys():
        del h5_file['shortest_path_distance']
    h5_file.create_dataset('shortest_path_distance',
                           data=shortest_dist_graph)
    if 'networkx_graph' in h5_file.keys():
        del h5_file['networkx_graph']
    h5_file.create_dataset("networkx_graph", data=np.array(
        [json.dumps(json_graph.node_link_data(G), cls=NumpyEncoder)], dtype='S'))


def extract_yolobbox(m, h5_file):

    if 'yolo_bbox' not in h5_file.keys():
        # print("###### EXTRACTING YOLO #######")
        yolo_bbox = []

        namesfile = "yolo_dataset/obj.names"
        class_names = load_class_names(namesfile)

        for obs in h5_file['observation']:
            img = Image.fromarray(obs).convert('RGB')
            sized = img.resize((m.width, m.height))

            current_bbox = dict()
            boxes = do_detect(m, sized, 0.5, 0.4, 1)
            width, height = img.size
            for box in boxes:
                x1 = int(round(float((box[0] - box[2]/2.0) * width)))
                y1 = int(round(float((box[1] - box[3]/2.0) * height)))
                x2 = int(round(float((box[0] + box[2]/2.0) * width)))
                y2 = int(round(float((box[1] + box[3]/2.0) * height)))

                cls_conf = box[5]
                cls_id = box[6]

                obj_name = class_names[cls_id] + '|'
                current_bbox[obj_name] = [x1, y1, x2, y2]
            yolo_bbox.append(json.dumps(
                current_bbox, cls=NumpyEncoder))

        h5_file.create_dataset('yolo_bbox',
                               data=[y.encode("ascii", "ignore") for y in yolo_bbox])


def main():
    argparse.ArgumentParser(description="")
    parser = argparse.ArgumentParser(description='Dataset creation.')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--scene', type=str, default=None)
    parser.add_argument('--view', action='store_true')
    args = vars(parser.parse_args())
    controller = ai2thor.controller.Controller()

    w, h = 400, 300
    if args['view']:
        w, h = 800, 600
    controller.start(player_screen_width=w, player_screen_height=h)

    # Use resnet from Keras to compute features
    # config = tf.ConfigProto() # before
    config = tf.compat.v1.ConfigProto() # after
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # set_session(tf.Session(config=config)) # before
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config = config)) # after
    resnet_trained = resnet50.ResNet50(
        include_top=False, weights='imagenet', pooling='avg', input_shape=(h, w, 3))
    # Freeze all layers
    for layer in resnet_trained.layers:
        layer.trainable = False

    # Use resnet places
    resnet_places = models.resnet50(num_classes=365)
    checkpoint = torch.load("agent/resnet/resnet50_places365.pth.tar",
                            map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k,
                  v in checkpoint['state_dict'].items()}
    resnet_places.load_state_dict(state_dict)
    resnet_places = torch.nn.Sequential(*list(resnet_places.children())[:-1])
    resnet_places.eval()
    resnet_places = resnet_places.to("cuda")

    object_feature, object_vector_spacy, object_vector_visualgenome = extract_object_feature(
        resnet_trained, h, w)

    custom_scene = False
    if args['scene'] is not None:
        names = [args['scene']]
        custom_scene = True
        scene_id = int(names[0].split("FloorPlan")[1])
        scene_type = -1
        if scene_id > 0 and scene_id < 100:
            scene_type = 0
        elif scene_id > 200 and scene_id < 300:
            scene_type = 1
        elif scene_id > 300 and scene_id < 400:
            scene_type = 2
        elif scene_id > 400 and scene_id < 500:
            scene_type = 3
    else:
        names, scene_type = construct_scene_names()

    pbar_names = tqdm(names, position=0)


    m = Darknet("yolo_dataset/yolov3_ai2thor.cfg")
    # m.load_weights("yolo_dataset/backup/yolov3_ai2thor_best.weights") # before
    m.load_weights("yolo_dataset/backup/yolov3_ai2thor_last.weights") # after
    m.print_network()
    m.cuda()

    for idx, name in enumerate(pbar_names):
        pbar_names.set_description("%s" % name)

        # Eval dataset
        if args['eval']:
            if args['view']:
                if not os.path.exists("data_eval_view/"):
                    os.makedirs("data_eval_view/")
                h5_file = h5py.File("data_eval_view/" + name + '.h5', 'a')
            else:
                if not os.path.exists("data_eval/"):
                    os.makedirs("data_eval/")
                h5_file = h5py.File("data_eval/" + name + '.h5', 'a')
        else:
            if args['view']:
                if not os.path.exists("data_view/"):
                    os.makedirs("data_view/")
                h5_file = h5py.File("data_view/" + name + '.h5', 'a')
            else:
                if not os.path.exists("data/"):
                    os.makedirs("data/")
                h5_file = h5py.File("data/" + name + '.h5', 'a')

        write_object_feature(h5_file,
                             object_feature, object_vector_spacy, object_vector_visualgenome)

        # Construct all possible states
        if custom_scene:
            states = create_states(h5_file, resnet_trained, resnet_places,
                                   controller, name, args, scene_type)
        else:
            states = create_states(h5_file, resnet_trained, resnet_places,
                                   controller, name, args, scene_type[idx])
        # Create action-state graph
        graph = create_graph(h5_file, states, controller, args)

        # Create shortest path from all state
        create_shortest_path(h5_file, states, graph)

        # Extract yolo bbox
        extract_yolobbox(m, h5_file)

        h5_file.close()

        gc.collect()


if __name__ == '__main__':
    main()