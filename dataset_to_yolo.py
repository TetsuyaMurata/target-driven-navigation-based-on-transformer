import argparse
import json
import re

import h5py
import numpy as np
import spacy
from PIL import Image
from scipy import spatial
from tqdm import tqdm, trange

names = []
SCENES = [0, 200, 300, 400]
TRAIN_SPLIT = (1, 22)
TEST_SPLIT = (21, 26)


KITCHEN_OBJECT_CLASS_LIST_TRAIN = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMachine",
    "GarbageCan",
    "Bowl",
]

KITCHEN_OBJECT_CLASS_LIST_EVAL = [
    "StoveBurner",
    "Mug",
    "Pot",
    "Cup"
]

LIVING_ROOM_OBJECT_CLASS_LIST_TRAIN = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST_EVAL = [
    "Sofa",
    "RemoteControl",
    "Box",
    "TableTop"
]

BEDROOM_OBJECT_CLASS_LIST_TRAIN = ["HousePlant", "Lamp", "Book", "AlarmClock"]

BEDROOM_OBJECT_CLASS_LIST_EVAL = ["Mirror", "CD", "CellPhone"]


BATHROOM_OBJECT_CLASS_LIST_TRAIN = [
    "Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]

BATHROOM_OBJECT_CLASS_LIST_EVAL = [
    "Toilet", "Towel", "SprayBottle"]

scene_id_name = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create param.json experiment file')
    parser.add_argument('--train_range', nargs=2, default=TRAIN_SPLIT,
                        help='train scene range Ex : 1 12')

    parser.add_argument('--eval_range', nargs=2, default=TEST_SPLIT,
                        help='train scene range Ex : 22 27')
    parser.add_argument('--method', type=str, default="word2vec",
                        help='Method to use Ex : word2vec')
    parser.add_argument('--reward', type=str, default="soft_goal",
                        help='Method to use Ex : soft_goal')

    parser.add_argument('--eval', action="store_true")

    args = vars(parser.parse_args())
    str_range = list(args["train_range"])
    for i, s in enumerate(str_range):
        str_range[i] = int(s)
    args["train_range"] = str_range

    str_range = list(args["eval_range"])
    for i, s in enumerate(str_range):
        str_range[i] = int(s)
    args["eval_range"] = str_range
    data = {}

    scene_tasks = [KITCHEN_OBJECT_CLASS_LIST_TRAIN, LIVING_ROOM_OBJECT_CLASS_LIST_TRAIN,
                   BEDROOM_OBJECT_CLASS_LIST_TRAIN, BATHROOM_OBJECT_CLASS_LIST_TRAIN]

    training = {}
    set_obj = None

    obs_id = 0
    if args["eval"]:
        range_scene = args['eval_range']
        base_path = "yolo_dataset/obs_label_eval/"
    else:
        range_scene = args['train_range']
        base_path = "yolo_dataset/obs_label/"

    for idx_scene, scene in enumerate(tqdm(SCENES)):
        for t in trange(*range_scene, position=1):
            name = "FloorPlan" + str(scene + t)
            f = h5py.File("data/"+name+".h5", 'r')
            # Use h5py object available
            obs = f['observation']
            bbox = [json.loads(bb) for bb in f['bbox']]
            for curr_obs, curr_bbox in tqdm(zip(obs, bbox), total=len(obs), position=2):

                im = Image.fromarray(curr_obs)
                img_name = "img_{:010d}".format(obs_id)
                im.save(base_path + img_name + '.jpeg')

                bboxs = []
                for key, value in curr_bbox.items():
                    keys = key.split('|')

                    # Convert to x center
                    x = (value[0] + value[2]) / 2

                    # Convert to y center
                    y = (value[1] + value[3]) / 2

                    width = value[2] - value[0]
                    height = value[3] - value[1]

                    # Normalize
                    x = x / 400
                    width = width / 400
                    y = y / 300
                    height = height / 300
                    if width < 0.025 or height < 0.025:
                        continue

                    bbox_str = str(OBJECT_IDS[keys[0]]) + \
                        " {} {} {} {}".format(x, y, width, height)
                    bboxs.append(bbox_str)
                with open(base_path + img_name + '.txt', 'w') as bbox_f:
                    for bb in bboxs:
                        bbox_f.write(bb + '\n')
                obs_id = obs_id + 1

        # if args['eval_objects']:
        #     scene_tasks = [KITCHEN_OBJECT_CLASS_LIST_EVAL, LIVING_ROOM_OBJECT_CLASS_LIST_EVAL,
        #                    BEDROOM_OBJECT_CLASS_LIST_EVAL, BATHROOM_OBJECT_CLASS_LIST_EVAL]

        # evaluation = {}

        # evaluation_set = dict()
        # for idx_scene, scene in enumerate(SCENES):
        #     evaluation_set[scene] = list()
        #     for t in range(*args['eval_range']):
        #         name = "FloorPlan" + str(scene + t)
        #         # Use h5py object available
        #         f = h5py.File("data/"+name+".h5", 'r')
