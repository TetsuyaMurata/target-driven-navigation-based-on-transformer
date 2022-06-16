# TODO keyboard exploration to create terminal state
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import signal
import sys

import ai2thor.controller
import numpy as np

from agent.tools import SimpleImageViewer

#
# Navigate the scene using your keyboard
#
actions = ["MoveAhead", "RotateRight",
           "RotateLeft", "MoveBack", "LookUp", "LookDown"]


def key_press(key, mod):

    global human_agent_action, human_wants_restart, stop_requested, info
    if key == ord('R') or key == ord('r'):  # r/R
        human_wants_restart = True
    if key == ord('Q') or key == ord('q'):  # q/Q
        stop_requested = True
    if key == 0xFF52:  # up
        human_agent_action = 0
    if key == 0xFF53:  # right
        human_agent_action = 1
    if key == 0xFF51:  # left
        human_agent_action = 2
    if key == 0xFF54:  # down
        human_agent_action = 3
    if key == 105:  # i key LookUp
        human_agent_action = 4
    if key == 107:  # k key LookDown
        human_agent_action = 5
    if key == 32:
        info = True


def rollout(env, state, scene_name):

    global human_agent_action, human_wants_restart, stop_requested, info
    human_agent_action = None
    human_wants_restart = False
    # env.reset()
    while True:
        # waiting for keyboard input
        if human_agent_action is not None:
            # move actions
            state = env.step(dict(action=actions[human_agent_action]))
            human_agent_action = None

        # waiting for reset command
        if human_wants_restart:
            # reset agent to random location
            # env.reset(scene_name)
            import time
            env.step(dict(action='Initialize', gridSize=0.25))
            env.step(dict(action='InitialRandomSpawn',
                                     randomSeed=time.time(), forceVisible=False, maxNumRepeats=30))
            
            human_wants_restart = False
        if info:
            # print(env.get_state.instance_detections2D)
            for bbox in state.instance_detections2D.keys():
                print(bbox.split('|')[0])
            obj_visible = [obj['objectId']
                           for obj in state.metadata['objects'] if obj['visible']]
            print(obj_visible)
            print(state.metadata['agent']['position'],
                  state.metadata['agent']['rotation'])
            info = False
        # check quit command
        if stop_requested:
            break
        viewer.imshow(np.zeros((30, 30, 3), dtype=np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scene_name", type=str, default="FloorPlan1",
                        help="AI2THOR scene name")
    args = parser.parse_args()

    print("Loading scene dump {}".format(args.scene_name))
    controller = ai2thor.controller.Controller()
    controller.start(player_screen_width=400, player_screen_height=300)
    controller.reset(args.scene_name)
    for i in range(5):
        state = controller.step(dict(action='InitialRandomSpawn',
                                     randomSeed=200, forceVisible=True, maxNumRepeats=30))
    state = controller.step(
        dict(action='Initialize', gridSize=0.5, renderObjectImage=True))
    # state = controller.step(dict(action='ToggleMapView'))

    human_agent_action = None
    human_wants_restart = False
    stop_requested = False
    info = False

    viewer = SimpleImageViewer()
    viewer.imshow(np.zeros((30, 30, 3), dtype=np.uint8))
    viewer.window.on_key_press = key_press

    print("Use arrow keys to move the agent.")
    print("Press R to reset agent\'s location.")
    print("Press Q to quit.")

    rollout(controller, state, args.scene_name)

    print("Goodbye.")
