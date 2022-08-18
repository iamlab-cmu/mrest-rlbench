from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class TakeUmbrellaOutOfUmbrellaStand(Task):
    def init_task(self):
        success_sensor = ProximitySensor("success")
        umbrella = Shape("umbrella")
        self.register_graspable_objects([umbrella])
        self.register_success_conditions(
            [DetectedCondition(umbrella, success_sensor, negated=True)]
        )

    def init_episode(self, index: int) -> List[str]:
        return [
            "take umbrella out of umbrella stand",
            "grasping the umbrella by its handle, lift it up and out of the" " stand",
            "remove the umbrella from the stand",
            "retrieve the umbrella from the stand",
            "get the umbrella",
            "lift the umbrella out of the stand",
        ]

    def variation_count(self) -> int:
        return 1

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        shape = Shape("umbrella")

        info = _get_shape_pose(shape)

        state = np.zeros(28)
        state[: info.size] = info

        return state


def _get_shape_pose(shape: Shape) -> np.ndarray:
    return np.concatenate([shape.get_position(), shape.get_quaternion()])
