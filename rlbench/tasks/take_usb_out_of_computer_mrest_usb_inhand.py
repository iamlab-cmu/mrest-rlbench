import numpy as np

from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, GraspedCondition
from rlbench.tasks.controllers import TakeUSBOutReachPController


class TakeUsbOutOfComputerMrestUsbInhand(Task):

    def init_task(self) -> None:
        usb = Shape('usb')
        self.register_graspable_objects([usb])
        self.register_success_conditions(
            [DetectedCondition(usb, ProximitySensor('success'), negated=True),
             GraspedCondition(self.robot.gripper, usb)])

    def init_episode(self, index: int) -> List[str]:
        return ['take usb out of computer',
                'remove the usb stick from its port',
                'retrieve the usb stick',
                'grasp the usb stick and slide it out of the pc',
                'get a hold of the usb stick and pull it out of the desktop '
                'computer']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, 0], [0, 0, 0]

    def get_noise_in_waypoints(self, waypoints) -> List[np.ndarray]:
        waypoint_noise = []
        for i in range(len(waypoints)):
            pos_noise = np.random.uniform(-0.01, 0.01, 3)
            if i == len(waypoints) - 2:
                pos_noise[:2] = np.random.uniform(-0.004, 0.004, 2)
                # Not too below
                pos_noise[2] = np.random.uniform(-0.01, 0.005)
            elif i == len(waypoints) - 1:
                pass
            else:
                pass

            waypoint_noise.append(pos_noise)

        return waypoint_noise

    def demo_primitive_class(self):
        return TakeUSBOutReachPController
