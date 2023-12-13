from multiprocessing import Process, Manager
from typing import List
from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task

from dataset_generator_closedloop_control import (save_demo, get_obs_config_front_wrist_cam_only, 
                                                  get_obs_config_all_sensors, keypoint_discovery)

import os
import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np
import pprint

from absl import app
from absl import flags

from rlbench.demo import Demo

FLAGS = flags.FLAGS

# flags.DEFINE_string('save_path',
#                     '/tmp/rlbench_data/',
#                     'Where to save the demos.')
# flags.DEFINE_list('tasks', [],
#                   'The tasks to collect. If empty, all tasks are collected.')
# flags.DEFINE_list('image_size', [256, 256],
#                   'The size of the images tp save.')
# flags.DEFINE_enum('renderer',  'opengl3', ['opengl', 'opengl3'],
#                   'The renderer to use. opengl does not include shadows, '
#                   'but is faster.')
# flags.DEFINE_integer('processes', 1,
#                      'The number of parallel processes during collection.')
# flags.DEFINE_integer('episodes_per_task', 10,
#                      'The number of episodes to collect per task.')
# flags.DEFINE_integer('variations', -1,
#                      'Number of variations to collect per task. -1 for all.')
# flags.DEFINE_integer('offset', 0,
#                      'First variation id.')
# flags.DEFINE_boolean('state', False,
#                      'Record the state (not available for all tasks).')


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


from pyquaternion import Quaternion

class LinearTrajectoryGenerator:

    def __init__(self):
        pass

    def are_points_collinear(self, points: np.ndarray) -> bool:
        eps = 1e-6
        dy_dx = np.arctan2((points[1:, 1] - points[0, 1]), (points[1:, 0] - points[0, 0]))
        dz_dx = np.arctan2((points[1:, 2] - points[0, 2]), (points[1:, 0] - points[0, 0]))
        dy_dx_median = np.median(dy_dx)
        dz_dx_median = np.median(dz_dx)
        off_median_threshold = 0.08
        dy_dx_within_bounds = np.abs(dy_dx - dy_dx_median) < off_median_threshold
        dz_dx_within_bounds = np.abs(dz_dx - dz_dx_median) < off_median_threshold
        
        dy_dx_within_bounds_percent = dy_dx_within_bounds.sum() / len(dy_dx_within_bounds)
        dz_dx_within_bounds_percent = dz_dx_within_bounds.sum() / len(dz_dx_within_bounds)

        # Percent threshold for collinearity
        percent_threshold = 0.60
        return dy_dx_within_bounds_percent > percent_threshold and dz_dx_within_bounds_percent > percent_threshold

    
    def check_demo_paths(self, demo: Demo):
        """
        Fetch the desired state and action based on the provided demo.
            :param demo: fetch each demo and save key-point observations
            :param normalise_rgb: normalise rgb to (-1, 1)
            :return: a list of obs and action
        """
        key_frame = keypoint_discovery(demo)
        action_ls = []
        last_frame = 0
        are_keyframe_paths_linear = []
        for f_idx, f in enumerate(key_frame):
            gripper_pos_path = []
            gripper_quat_path = []
            gripper_quats = []
            for i in range(last_frame, f + 1):
                gripper_pose = demo._observations[i].gripper_pose
                gripper_pos_path.append(gripper_pose[:3])
                gripper_quat_path.append(gripper_pose[3:])
                x, y, z, w = gripper_pose[3:]
                gripper_quats.append(Quaternion(w, x, y, z))

            gripper_pos_arr = np.c_[gripper_pos_path]
            # breakpoint()
            is_linear_path = self.are_points_collinear(gripper_pos_arr)
            are_keyframe_paths_linear.append(is_linear_path)
            gripper_quat_path = np.c_[gripper_quat_path]

        return {
            'keyframe_pos_linear': are_keyframe_paths_linear,
        }

    

def run(i, lock, task_index, variation_count, results, file_lock, tasks):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, FLAGS.image_size))

    # obs_config = get_obs_config_all_sensors(img_size)
    obs_config = get_obs_config_front_wrist_cam_only(img_size)

    rlbench_env = Environment(
        # action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning(), Discrete()),
        obs_config=obs_config,
        headless=True)
    # rlbench_env.launch('task_design_small_square_insert.ttt')
    # rlbench_env.launch('task_design_pick_and_lift_small.ttt')
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if FLAGS.variations >= 0:
                var_target = np.minimum(FLAGS.variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = FLAGS.offset
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        obs, descriptions = task_env.reset()

        variation_path = os.path.join(
            FLAGS.save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)
        print(variation_path)

        check_and_make(variation_path)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(FLAGS.episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            attempts = 10
            while attempts > 0:
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                if os.path.exists(episode_path):
                    break

                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True,
                        use_primitives=True)

                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break

                # TODO(Mohit): Should we ask user input to see if we should save demo or not.
                print('Will save demo')
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()


def main(argv):

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]

    if len(FLAGS.tasks) > 0:
        for t in FLAGS.tasks:
            if t not in task_files:
                raise ValueError('Task %s not recognised!.' % t)
        task_files = FLAGS.tasks

    tasks = [task_file_to_task_class(t) for t in task_files]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', FLAGS.offset)
    lock = manager.Lock()

    check_and_make(FLAGS.save_path)

    multiprocess = False
    if not multiprocess:
        run(0, lock, task_index, variation_count, result_dict, file_lock, tasks)
    else:
        processes = [Process(
            target=run, args=(
                i, lock, task_index, variation_count, result_dict, file_lock,
                tasks))
            for i in range(FLAGS.processes)]
        [t.start() for t in processes]
        [t.join() for t in processes]

    print('Data collection done!')
    for i in range(FLAGS.processes):
        print(result_dict[i])


if __name__ == '__main__':
  app.run(main)
