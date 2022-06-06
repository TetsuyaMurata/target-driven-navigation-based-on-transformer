import argparse
import functools
import json
from operator import attrgetter

import numpy as np


class Episode():
    def __init__(self, task, actions, shortest_path_length, success):
        self.task = task
        self.actions = actions
        self.shortest_path_length = shortest_path_length
        self.success = success

    def is_success(self):
        return 1 if self.success else 0

    def SPL(self):
        return self.shortest_path_length / max(self.shortest_path_length, len(self.actions))

    def __repr__(self):
        return self.task


def get_scene_type(scene_name):
    scene_id = int(scene_name.split("FloorPlan")[1])
    if scene_id < 200:
        return "Kitchen"
    elif scene_id < 300:
        return "Livingroom"
    elif scene_id < 400:
        return "Bedroom"
    elif scene_id < 500:
        return "Bathroom"


def comparator(key1, key2):
    key1 = key1.split("FloorPlan")[1]
    key2 = key2.split("FloorPlan")[1]
    return int(key1) - int(key2)


def main():
    parser = argparse.ArgumentParser(
        description='Compute SPL and success rate from human agent')

    # Use experiment.json
    parser.add_argument('--exp', '-e', type=str,
                        help='Results from human.json file', required=True)
    parser.add_argument('--latex',
                        help='Output latex style table', action='store_true')
    args = vars(parser.parse_args())
    with open(args['exp']) as json_file:
        results = json.load(json_file)

    scenes_results = dict()
    for res in results['data']:
        actions = res['actions']
        scene_name = res['scene_name']
        target = res['target']['object']
        success = res['success']
        shortest_path_length = res['shortest_path']

        episode = Episode(target, actions, shortest_path_length, success)
        if scene_name in scenes_results:
            scenes_results[scene_name].append(episode)
        else:
            scenes_results[scene_name] = [episode]

    scenes_stats = dict()
    for scene in scenes_results.keys():
        scene_success = []
        scene_spl = []
        targets = []
        for episode in scenes_results[scene]:
            scene_success.append(episode.is_success())
            scene_spl.append(round(episode.is_success() * episode.SPL() * 100, 2))
            targets.append(episode.task)

        scenes_stats[scene] = {"success": scene_success,
                               "SPL": scene_spl, "targets": targets}
    print("Results:")
    for scene in sorted(scenes_stats, key=functools.cmp_to_key(comparator)):
        for ep_target, ep_success, ep_spl in sorted(zip(scenes_stats[scene]['targets'], scenes_stats[scene]['success'], scenes_stats[scene]['SPL'])):
            if args['latex']:
                print(scene, '\t&', ep_target, '\t&', ep_spl,
                      '\t&', 'Yes' if ep_success else 'No', '\\\\')
            else:
                print("Scene:", scene, '| Target:', ep_target, '| SPL:', ep_spl,
                      '| Success', 'Yes' if ep_success else 'No')

    print("Summary results:")
    scenetype_stats = dict()
    for scene in sorted(scenes_stats, key=functools.cmp_to_key(comparator)):
        success_mean = round(np.mean(scenes_stats[scene]['success'])*100, 2)
        spl_mean = round(np.mean(scenes_stats[scene]['SPL']), 2)
        if args['latex']:
            print(scene, '\t&', spl_mean, '\t&', success_mean, '\\\\')
        else:
            print(scene, "| SPL:", spl_mean, '| Success:', success_mean)

        scene_type = get_scene_type(scene)
        if scene_type in scenetype_stats:
            scenetype_stats[scene_type]['success'].append(success_mean)
            scenetype_stats[scene_type]['SPL'].append(spl_mean)
        else:
            scenetype_stats[scene_type] = {
                'success': [success_mean], 'SPL': [spl_mean]}
    print("Summary per scene type")
    for scene_type in scenetype_stats:
        success_mean = round(np.mean(scenetype_stats[scene_type]['success']), 2)
        spl_mean = round(np.mean(scenetype_stats[scene_type]['SPL']), 2)
        if args['latex']:
            print(scene_type, '\t&', spl_mean, '\t&', success_mean, '\\\\')
        else:
            print(scene_type, "| SPL:", spl_mean, '| Success:', success_mean)


if __name__ == '__main__':
    main()
