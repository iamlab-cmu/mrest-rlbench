import numpy as np

from typing import List, Tuple
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary

from rlbench.tasks.controllers import ShapeSorterReachPController


# SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'moon']
SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star']


class PickAndLiftSmallMrest(Task):

    def init_task(self) -> None:
        self._shapes = [Shape(ob.replace(' ', '_')) for ob in SHAPE_NAMES]
        self._grasp_points = [
            Dummy('%s_grasp_point' % ob.replace(' ', '_'))
            for ob in SHAPE_NAMES]
        self._w1 = Dummy('waypoint1')

        self.register_graspable_objects(self._shapes)
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')

    def init_episode(self, index: int) -> List[str]:
        shape = self._shapes[index]
        self.register_success_conditions([
            GraspedCondition(self.robot.gripper, shape),
            DetectedCondition(shape, self.success_detector)
        ])
        self.boundary.clear()
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for sh in self._shapes:
            self.boundary.sample(sh,
                                 min_distance=0.05,
                                 min_rotation=(0.0, 0.0, 0.0),
                                 max_rotation=(0.0, 0.0, 0.0),
                                 ignore_collisions=False)

        self._w1.set_pose(self._grasp_points[index].get_pose())

        return ['pick up the %s and lift it up to the target' %
                SHAPE_NAMES[index],
                'grasp the blue %s to the target' % SHAPE_NAMES[index],
                'lift the blue %s up to the target' % SHAPE_NAMES[index]]

    def variation_count(self) -> int:
        return len(SHAPE_NAMES)

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Defines how much the task base can rotate during episode placement.

        Default is set such that it can rotate any amount on the z axis.

        :return: A tuple containing the min and max (x, y, z) rotation bounds
            (in radians).
        """
        return (0.0, 0.0, np.pi/2.0), (0.0, 0.0, np.pi/2.0+0.0001)
    
    def get_noise_in_waypoints(self, waypoints) -> List[np.ndarray]:
        waypoint_noise = []
        for i in range(len(waypoints)):
            pos_noise = np.random.uniform(-0.01, 0.01, 3)
            if i == 0:
                # Large amount of noise when far above (so that we don't commit to one object prematurely)
                pos_noise[:2] = np.random.uniform(-0.08, 0.08, 2)
            elif i == 1:
                pos_noise[2] = 0.0
            elif i == len(waypoints) - 2:
                # Not too below
                pos_noise[2] = np.random.uniform(-0.003, 0.01)
            elif i == len(waypoints) - 1:
                pos_noise = np.copy(waypoint_noise[1])
                # No noise in Y and Z axes and same noise along X-as during grasping
                pos_noise[1:] = 0.0
            else:
                pass

            waypoint_noise.append(pos_noise)

        return waypoint_noise
 
    def demo_primitive_class(self):
        return ShapeSorterReachPController
