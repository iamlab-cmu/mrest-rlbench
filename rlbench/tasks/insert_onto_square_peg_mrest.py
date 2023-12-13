from typing import List, Tuple
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors

from rlbench.tasks.controllers import ReachPController


class InsertOntoSquarePegMrest(Task):

    def init_task(self) -> None:
        self._square_ring = Shape('square_ring')
        self._success_centre = Dummy('success_centre')
        success_detectors = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        self.register_graspable_objects([self._square_ring])
        success_condition = ConditionSet([DetectedCondition(
            self._square_ring, sd) for sd in success_detectors])
        self.register_success_conditions([success_condition])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]

        fixed_colors = False
        # Fix the pillar we use for insertion as well as the other pillar colors.
        if fixed_colors:
            chosen_pillar = spokes[0]
        else:
            chosen_pillar = np.random.choice(spokes)
        chosen_pillar.set_color(color_rgb)

        _, _, z = self._success_centre.get_position()
        x, y, _ = chosen_pillar.get_position()
        self._success_centre.set_position([x, y, z])

        if fixed_colors:
            color_choices = [1, 2]
        else:
            color_choices = np.random.choice(
                list(range(index)) + list(range(index + 1, len(colors))),
                size=2, replace=False)
        spokes.remove(chosen_pillar)
        for spoke, i in zip(spokes, color_choices):
            name, rgb = colors[i]
            spoke.set_color(rgb)
        b = SpawnBoundary([Shape('boundary0')])
        b.sample(self._square_ring, min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        return ['put the ring on the %s spoke' % color_name,
                'slide the ring onto the %s colored spoke' % color_name,
                'place the ring onto the %s spoke' % color_name]

    def variation_count(self) -> int:
        return len(colors)

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
        return ReachPController
