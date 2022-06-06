import argparse
import json
import random
import time

import ai2thor.controller
import cv2
import evdev
import h5py
import numpy as np
from evdev import InputDevice, categorize, ecodes
from tqdm import tqdm

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.utils import populate_config

joystick_tolerance = 20000

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


def find_shortest_path(name, start_id, end_id):
    f = h5py.File("data/" + name + ".h5", "r")
    shortest_path = f["shortest_path_distance"][start_id][end_id]
    f.close()
    return shortest_path


def check_visibility(env):
    for objectId in env.object_visibility[env.current_state_id]:
        obj = objectId.split('|')
        if obj[0] == env.terminal_state['object']:
            return True
    return False


def reset_env(name, target):
    # Reset to new env
    def h5_file_lambda(s): return "data_view/{scene}.h5".replace("{scene}", s)
    env = THORDiscreteEnvironmentFile(
        scene_name=name, terminal_state=target, method="random", reward="soft_goal", h5_file_path=h5_file_lambda, action_size=9)
    env.reset()
    return env


def display_target(target, wait=False, success=None):
    font = cv2.FONT_HERSHEY_SIMPLEX

    bottomLeftCornerOfText = (5, 50)
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    img = np.zeros((100, 300, 3), np.uint8) + 255
    if wait:
        cv2.putText(img, "Please wait",
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    (0, 0, 255),
                    lineType)
    else:
        if success != None:
            if success:
                cv2.putText(img, "Success",
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            (0, 255, 0),
                            lineType)
            else:
                cv2.putText(img, "Failed",
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            (0, 0, 255),
                            lineType)
        else:
            cv2.putText(img, target['object'],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    # Display the image
    for i in range(10):
        cv2.imshow("Target", img)
        cv2.waitKey(1)


def display_obs(obs):
    cv2.imshow("View", obs[:, :, ::-1])


def default(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def main():
    parser = argparse.ArgumentParser(
        description='Episode evaluation using human agent')

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Experiment parameters.json file', required=True)
    args = vars(parser.parse_args())
    config = populate_config(args, mode="eval")

    # Choose first controller
    device = InputDevice(evdev.list_devices()[0])
    print("Using", device.name)

    # Set random seed
    random.seed()
    seed = int(random.randint(0, 200))

    # Init random with seed
    random.seed(seed)

    js_right = False
    js_left = False
    js_up = False
    js_down = False

    # Init evaluation
    # Load tasks
    tasks = []
    current_task = 0
    for scene in config["task_list"].keys():
        it = 0
        for target in config["task_list"].get(scene):
            target['id'] = it
            it = it + 1
            tasks.append((scene, target))

    # Shuffle tasks
    random.shuffle(tasks)

    # Store episode results
    episodes_param = []
    current_ep_action = []
    scene_name, current_target = "FloorPlan201", {"object": "Television"}
    # Display target
    display_target(current_target, wait=True)

    # Training start

    env = reset_env(scene_name, current_target)
    training = True
    display_target(current_target)
    display_obs(env.observation)
    cv2.waitKey(100)

    # Display evaluation progress
    pbar = tqdm(total=len(tasks))
    for event in device.read_loop():
        # Ignore all event
        # if event.timestamp()-time.time() > 1e-2:
        #     pass
        action = None
        if event.type == ecodes.EV_KEY:
            c = categorize(event)

            # X is pressed
            if event.code == ecodes.BTN_A:
                if c.keystate == c.key_down:

                    if not training:
                        # Store current episode
                        # Get current agent position
                        agent_pos = env.locations[env.current_state_id]
                        agent_rot = env.rotations[env.current_state_id]
                        end_pose = {"x": agent_pos[0], "y": agent_pos[1], "z": agent_pos[2],
                                    "rotation": agent_rot[1], "horizon": agent_rot[2]}

                        # Convert to dataset id
                        end_id = env.current_state_id

                        # Check if object is visible
                        success = check_visibility(env)
                        episode = {
                            "actions": current_ep_action,
                            "scene_name": scene_name,
                            "target": current_target,
                            "success": success,
                            "start_pose": start_pose,
                            "start_id": start_id,
                            "end_pose": end_pose,
                            "end_id": end_id,
                            "shortest_path": env.shortest_path_terminal(start_id)}
                        episodes_param.append(episode)

                        # Start a new task
                        current_task = current_task + 1

                        # Check if all tasks are done
                        if current_task == len(tasks):
                            break
                        scene_name, current_target = tasks[current_task]

                        # Reset var for new episode
                        display_target(current_target, wait=True)
                        env = reset_env(scene_name, current_target)
                        agent_pos = env.locations[env.current_state_id]
                        agent_rot = env.rotations[env.current_state_id]
                        start_pose = {"x": agent_pos[0], "y": agent_pos[1], "z": agent_pos[2],
                                      "rotation": agent_rot[1], "horizon": agent_rot[2]}
                        current_ep_action = []
                        start_id = env.current_state_id
                        display_target(current_target)
                        display_obs(env.observation)

                        # Update tqdm
                        pbar.update(1)

                        cv2.waitKey(100)
                    else:
                        success = check_visibility(env)
                        env.reset()
                        display_target(current_target, success=success)
                        display_obs(env.observation)
                        cv2.waitKey(100)

            # Start button run the evaluation
            if event.code == ecodes.BTN_START:
                if training:
                    training = False
                    scene_name, current_target = tasks[current_task]
                    display_target(current_target, wait=True)
                    env = reset_env(scene_name, current_target)
                    agent_pos = env.locations[env.current_state_id]
                    agent_rot = env.rotations[env.current_state_id]
                    start_pose = {"x": agent_pos[0], "y": agent_pos[1], "z": agent_pos[2],
                                  "rotation": agent_rot[1], "horizon": agent_rot[2]}
                    start_id = env.current_state_id
                    display_target(current_target)
                    display_obs(env.observation)
                    cv2.waitKey(100)
            if event.code == ecodes.BTN_SELECT:
                break

        if event.type == ecodes.EV_ABS:
            c = categorize(event)
            # D-PAD
            if event.code == ecodes.ABS_HAT0X:
                if event.value == 1:
                    action = "MoveRight"
                elif event.value == -1:
                    action = "MoveLeft"

            elif event.code == ecodes.ABS_HAT0Y:
                if event.value == -1:
                    action = "MoveAhead"
                elif event.value == 1:
                    action = "MoveBack"

            # Right joystick
            if event.code == ecodes.ABS_RX:
                val = event.value
                if abs(val) > joystick_tolerance:
                    if val > 0 and not js_right:
                        action = "RotateRight"
                        js_right = True

                    if val < 0 and not js_left:
                        action = "RotateLeft"
                        js_left = True
                else:
                    js_right = False
                    js_left = False

            if event.code == ecodes.ABS_RY:
                val = event.value
                if abs(val) > joystick_tolerance:
                    if val > 0 and not js_down:
                        action = "LookDown"
                        js_down = True

                    if val < 0 and not js_up:
                        action = "LookUp"
                        js_up = True
                else:
                    js_up = False
                    js_down = False

        if action is not None:
            current_ep_action.append(action)
            env.step(env.actions.index(action))
            display_obs(env.observation)
            display_target(current_target)
            cv2.waitKey(100)

    from datetime import datetime
    with open("EXPERIMENTS/SPL_HUMAN/human_agent" + datetime.now().strftime('%b%d_%H-%M-%S') + ".json", "w") as f:
        f.write(json.dumps(
            {"seed": seed, "data": episodes_param}, default=default, indent=4))


if __name__ == '__main__':
    main()
