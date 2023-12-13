import numpy as np

from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition

from rlbench.tasks.controllers import ShapeSorterReachPController


SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'moon']


class PlaceShapeInShapeSorterMrest(Task):

    def init_task(self) -> None:
        self.shape_sorter = Shape('shape_sorter')
        self.success_sensor = ProximitySensor('success')
        self.shapes = [Shape(ob.replace(' ', '_')) for ob in SHAPE_NAMES]
        self.drop_points = [
            Dummy('%s_drop_point' % ob.replace(' ', '_'))
            for ob in SHAPE_NAMES]
        self.grasp_points = [
            Dummy('%s_grasp_point' % ob.replace(' ', '_'))
            for ob in SHAPE_NAMES]
        self.waypoint1 = Dummy('waypoint1')
        self.waypoint4 = Dummy('waypoint4')
        self.register_graspable_objects(self.shapes)

        self.register_waypoint_ability_start(0, self._set_grasp)
        self.register_waypoint_ability_start(3, self._set_drop)
        self.boundary = SpawnBoundary([Shape('boundary')])

    def init_episode(self, index) -> List[str]:
        self.variation_index = index
        shape = SHAPE_NAMES[index]
        self.register_success_conditions(
            [DetectedCondition(self.shapes[index], self.success_sensor)])

        self.boundary.clear()
        # [self.boundary.sample(s, min_distance=0.05) for s in self.shapes]
        [self.boundary.sample(s, min_distance=0.05, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
         for s in self.shapes]

        return ['put the %s in the shape sorter' % shape,
                'pick up the %s and put it in the sorter' % shape,
                'place the %s into its slot in the shape sorter' % shape,
                'slot the %s into the shape sorter' % shape]

    def variation_count(self) -> int:
        return len(SHAPE_NAMES)

    def _set_grasp(self, _):
        gp = self.grasp_points[self.variation_index]
        self.waypoint1.set_pose(gp.get_pose())

    def _set_drop(self, _):
        dp = self.drop_points[self.variation_index]
        self.waypoint4.set_pose(dp.get_pose())

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
            if i == 1:
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
