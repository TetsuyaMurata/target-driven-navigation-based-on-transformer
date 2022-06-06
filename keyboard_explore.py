# TODO keyboard exploration to create terminal state
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import signal
import sys

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.utils import populate_config
import numpy as np

from agent.tools import SimpleImageViewer

#
# Navigate the scene using your keyboard
#
actions = ["MoveAhead", "RotateRight",
           "RotateLeft", "MoveBack", "LookUp", "LookDown", "MoveLeft", "MoveRight"]

def reset_env(name, target):
    # Reset to new env
    def h5_file_lambda(s): return "data_old/{scene}.h5".replace("{scene}", s)
    env = THORDiscreteEnvironmentFile(
        scene_name=name, terminal_state=target, method="random", reward="soft_goal", h5_file_path=h5_file_lambda, action_size=9)
    env.reset()
    return env
    
def key_press(key, mod):

    global human_agent_action, human_wants_restart, stop_requested, info
    if key == ord('R') or key == ord('r'):  # r/R
        human_wants_restart = True
    if key == ord('Q') or key == ord('q'):  # q/Q
        stop_requested = True
    if key == 0xFF52:  # up
        human_agent_action = 0
    if key == 0xFF53:  # right
        human_agent_action = 7
    if key == 0xFF51:  # left
        human_agent_action = 6
    if key == 0xFF54:  # down
        human_agent_action = 3
    if key == 105:  # i key LookUp
        human_agent_action = 4
    if key == 107:  # k key LookDown
        human_agent_action = 5
    if key == 106:  # j key RotateRight
        human_agent_action = 2
    if key == 108:  # l key RotateLeft
        human_agent_action = 1
    if key == 32:
        info = True
    print(key)


def rollout(env):

    global human_agent_action, human_wants_restart, stop_requested, info
    human_agent_action = None
    human_wants_restart = False
    state = None
    # env.reset()
    while True:
        # waiting for keyboard input
        if human_agent_action is not None:
            # move actions
            state = env.step(env.acts.index(actions[human_agent_action]))
            human_agent_action = None

        # waiting for reset command
        if human_wants_restart:
            # reset agent to random location
            env.reset()
            human_wants_restart = False
        if info:
            # print(env.get_state.instance_detections2D)
            for bbox in env.boudingbox.keys():
                print(bbox.split('|')[0])
            info = False
        # check quit command
        if stop_requested:
            break
        viewer.imshow(env.observation)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene_name", type=str, default="FloorPlan1",
                        help="AI2THOR scene name")
    args = parser.parse_args()

    print("Loading scene dump {}".format(args.scene_name))
    
    env = reset_env(args.scene_name, {"object":"GarbageCan"})
    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    info = False

    viewer = SimpleImageViewer()
    viewer.imshow(np.zeros((300, 400, 3), dtype=np.uint8))
    viewer.window.on_key_press = key_press

    print("Use arrow keys to move the agent.")
    print("Press R to reset agent\'s location.")
    print("Press Q to quit.")

    rollout(env)

    print("Goodbye.")
