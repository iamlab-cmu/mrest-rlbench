from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import state_size


class SlideBlockToTarget(Task):
    def init_task(self) -> None:
        self.register_success_conditions(
            [DetectedCondition(Shape("block"), ProximitySensor("success"))]
        )

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        return [
            "slide the block to target",
            "slide the block onto the target",
            "push the block until it is sitting on top of the target",
            "slide the block towards the green target",
            "cover the target with the block by pushing the block in its" " direction",
        ]

    def variation_count(self) -> int:
        return 1

    @property
    def state(self) -> np.ndarray:
        """
        Return a vector containing information for all objects in the scene
        """
        shapes = [Shape("block"), ProximitySensor("success")]

        info = np.concatenate([_get_shape_pose(shape) for shape in shapes])

        state = np.zeros(state_size)
        state[: info.size] = info

        return state


def _get_shape_pose(shape: Shape) -> np.ndarray:
    return np.concatenate([shape.get_position(), shape.get_quaternion()])
