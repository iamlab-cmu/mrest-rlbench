import numpy as np
from typing import Optional

from rlbench.backend.waypoints import Point 


class Reach:
    def __init__(self, task, target: Point) -> None:
        self.task = task
        self.target = target
        self.target_pos = target._waypoint.get_position()
        self.target_quat = target._waypoint.get_quaternion()

        self.action_mag = 0.03
        self.noise_ratio = 0.2

        self.action_space_low = np.array([-self.action_mag, -self.action_mag, -self.action_mag])
        self.action_space_high = np.array([self.action_mag,  self.action_mag,  self.action_mag])

        self.norm_action_low = np.array([-1., -1., -1.])
        self.norm_action_high = np.array([1., 1., 1.])
    
    def current_position(self, robot):
        tip = robot.arm.get_tip()
        pose = tip.get_pose()
        return pose[:3], pose[3:]

    def step(self, robot):
        curr_pos, curr_quat = self.current_position(robot)
        action_delta = self.target_pos - curr_pos
        action_delta = np.clip(action_delta, self.action_space_low, self.action_space_high)
        final_pos = curr_pos + action_delta
        # Do not change the orientation
        # breakpoint()

        # add_noise = np.linalg.norm(self.target_pos - final_pos) > 0.06
        add_noise = np.linalg.norm(self.target_pos - final_pos) > (self.action_mag + 0.333 * self.action_mag)
        if add_noise:
            action_noise = np.random.uniform(-0.005, 0.005, 3)
            final_pos += action_noise
        # elif np.linalg.norm(self.target_pos - final_pos) > 0.02:
        elif np.linalg.norm(self.target_pos - final_pos) > (self.action_mag + 0.1 * self.action_mag):
            action_noise = np.random.uniform(-0.005, 0.005, 3)
            final_pos += action_noise

        action_norm = self.normalize_action(action_delta)

        path = self.target._robot.arm.get_linear_path(
            final_pos,
            # euler=self.target._waypoint.get_orientation(),
            quaternion=curr_quat,
            ignore_collisions=False,
            add_noise=False,
            save_no_noise_path=False,)

        return path, {
            'final_pos': final_pos,
            'action_norm': action_norm,
            'action_delta': action_delta,
            'action_low': self.action_space_low,
            'action_high': self.action_space_high,
            }
    
    def normalize_action(self, action_abs):
        # This assumes that norm action
        norm_action = (action_abs - self.action_space_low) / (self.action_space_high - self.action_space_low)
        assert np.all(norm_action >= 0.0) and np.all(norm_action <= 1.0)
        return norm_action

    def done(self, robot, err_th: float = 0.01):
        curr_pos, _ = self.current_position(robot)
        return np.linalg.norm(self.target_pos - curr_pos) < err_th


class ReachPController:
    def __init__(self, task, target: Point, fixed_noise_in_target: Optional[np.ndarray] = None) -> None:
        self.task = task
        self.target = target
        self.target_pos = np.copy(target._waypoint.get_position())
        self.target_quat = target._waypoint.get_quaternion()
        if fixed_noise_in_target is not None:
            self.target_pos += fixed_noise_in_target

        self.Kp = 0.5
        self.noise_ratio = 0.2

        self.action_mag = 0.03
        self.action_space_low = np.array([-self.action_mag, -self.action_mag, -self.action_mag])
        self.action_space_high = np.array([self.action_mag,  self.action_mag,  self.action_mag])

        # Set action_low to 0.0 for reach tasks
        # self.norm_action_low = np.array([-1., -1., -1.])
        self.norm_action_low = np.array([0., 0., 0.])
        self.norm_action_high = np.array([1., 1., 1.])
    
    def current_position(self, robot):
        tip = robot.arm.get_tip()
        pose = tip.get_pose()
        return pose[:3], pose[3:]

    def step(self, robot):
        curr_pos, curr_quat = self.current_position(robot)
        action_delta = self.Kp * (self.target_pos - curr_pos)
        action_delta = np.clip(action_delta, self.action_space_low, self.action_space_high)
        final_pos = curr_pos + action_delta
        # Do not change the orientation
        # breakpoint()

        is_ee_far = np.linalg.norm(self.target_pos - final_pos) > (self.action_mag + 1.0 * self.action_mag)
        if is_ee_far:
            # Add large noise if end-effector is far from target
            # action_noise = np.random.uniform(-0.008, 0.008, 3)
            action_noise = np.random.uniform(-0.005, 0.005, 3)
            final_pos += action_noise
        else:
            # Add small noise (Don't add if we already have noise in waypoint position)
            # action_noise = np.random.uniform(-0.002, 0.002, 3)
            # final_pos += action_noise
            pass

        # action_noise = np.random.uniform(-0.002, 0.002, 3)
        # final_pos += action_noise

        action_norm = self.normalize_action(action_delta)

        path = self.target._robot.arm.get_linear_path(
            final_pos,
            # euler=self.target._waypoint.get_orientation(),
            quaternion=curr_quat,
            ignore_collisions=False,
            add_noise=False,
            save_no_noise_path=False,)

        return path, {
            'final_pos': final_pos,
            'action_norm': action_norm,
            'action_delta': action_delta,
            'action_low': self.action_space_low,
            'action_high': self.action_space_high,
            }
    
    def normalize_action(self, action_abs):
        # This assumes that norm action
        norm_action = (action_abs - self.action_space_low) / (self.action_space_high - self.action_space_low)
        assert np.all(norm_action >= 0.0) and np.all(norm_action <= 1.0)
        return norm_action

    def done(self, robot, err_th: float = 0.01):
        curr_pos, _ = self.current_position(robot)
        return np.linalg.norm(self.target_pos - curr_pos) < err_th


class ShapeSorterReachPController:
    def __init__(self, task, target: Point, fixed_noise_in_target: Optional[np.ndarray] = None) -> None:
        self.task = task
        self.target = target
        self.target_pos = np.copy(target._waypoint.get_position())
        self.target_quat = target._waypoint.get_quaternion()
        if fixed_noise_in_target is not None:
            self.target_pos += fixed_noise_in_target

        self.Kp = 0.5
        self.noise_ratio = 0.2

        self.action_mag = 0.03
        self.action_space_low = np.array([-self.action_mag, -self.action_mag, -self.action_mag])
        self.action_space_high = np.array([self.action_mag,  self.action_mag,  self.action_mag])

        # Set action_low to 0.0 for reach tasks
        # self.norm_action_low = np.array([-1., -1., -1.])
        self.norm_action_low = np.array([0., 0., 0.])
        self.norm_action_high = np.array([1., 1., 1.])
    
    def current_position(self, robot):
        tip = robot.arm.get_tip()
        pose = tip.get_pose()
        return pose[:3], pose[3:]

    def step(self, robot):
        curr_pos, curr_quat = self.current_position(robot)
        action_delta = self.Kp * (self.target_pos - curr_pos)
        action_delta = np.clip(action_delta, self.action_space_low, self.action_space_high)
        final_pos = curr_pos + action_delta
        # Do not change the orientation
        # breakpoint()

        is_ee_far = np.linalg.norm(self.target_pos - final_pos) > (self.action_mag + 1.0 * self.action_mag)
        if is_ee_far:
            # Add large noise if end-effector is far from target
            action_noise = np.random.uniform(-0.005, 0.005, 3)
            final_pos += action_noise
            pass
        else:
            # Add small noise (Don't add if we already have noise in waypoint position)
            # action_noise = np.random.uniform(-0.002, 0.002, 3)
            # final_pos += action_noise
            pass

        # action_noise = np.random.uniform(-0.002, 0.002, 3)
        # final_pos += action_noise

        action_norm = self.normalize_action(action_delta)

        path = self.target._robot.arm.get_linear_path(
            final_pos,
            # euler=self.target._waypoint.get_orientation(),
            quaternion=curr_quat,
            ignore_collisions=False,
            add_noise=False,
            save_no_noise_path=False,)

        return path, {
            'final_pos': final_pos,
            'action_norm': action_norm,
            'action_delta': action_delta,
            'action_low': self.action_space_low,
            'action_high': self.action_space_high,
            }
    
    def normalize_action(self, action_abs):
        # This assumes that norm action
        norm_action = (action_abs - self.action_space_low) / (self.action_space_high - self.action_space_low)
        assert np.all(norm_action >= 0.0) and np.all(norm_action <= 1.0)
        return norm_action

    def done(self, robot, err_th: float = 0.01):
        curr_pos, _ = self.current_position(robot)
        return np.linalg.norm(self.target_pos - curr_pos) < err_th


class TakeUSBOutReachPController:
    def __init__(self, task, target: Point, fixed_noise_in_target: Optional[np.ndarray] = None) -> None:
        self.task = task
        self.target = target
        self.target_pos = np.copy(target._waypoint.get_position())
        self.target_quat = target._waypoint.get_quaternion()
        if fixed_noise_in_target is not None:
            self.target_pos += fixed_noise_in_target

        self.Kp = 0.5
        self.noise_ratio = 0.2

        self.action_mag = 0.03
        self.action_space_low = np.array([-self.action_mag, -self.action_mag, -self.action_mag])
        self.action_space_high = np.array([self.action_mag,  self.action_mag,  self.action_mag])

        # Set action_low to 0.0 for reach tasks
        # self.norm_action_low = np.array([-1., -1., -1.])
        self.norm_action_low = np.array([0., 0., 0.])
        self.norm_action_high = np.array([1., 1., 1.])
    
    def current_position(self, robot):
        tip = robot.arm.get_tip()
        pose = tip.get_pose()
        return pose[:3], pose[3:]

    def step(self, robot):
        curr_pos, curr_quat = self.current_position(robot)
        action_delta = self.Kp * (self.target_pos - curr_pos)
        action_delta = np.clip(action_delta, self.action_space_low, self.action_space_high)
        final_pos = curr_pos + action_delta
        # Do not change the orientation
        # breakpoint()

        is_ee_far = np.linalg.norm(self.target_pos - final_pos) > (self.action_mag + 1.0 * self.action_mag)
        if is_ee_far:
            # Add large noise if end-effector is far from target
            action_noise = np.random.uniform(-0.005, 0.005, 3)
            final_pos += action_noise
            pass
        else:
            # Add small noise (Don't add if we already have noise in waypoint position)
            # action_noise = np.random.uniform(-0.002, 0.002, 3)
            # final_pos += action_noise
            pass

        # action_noise = np.random.uniform(-0.002, 0.002, 3)
        # final_pos += action_noise

        action_norm = self.normalize_action(action_delta)

        path = self.target._robot.arm.get_linear_path(
            final_pos,
            # euler=self.target._waypoint.get_orientation(),
            quaternion=curr_quat,
            ignore_collisions=False,
            add_noise=False,
            save_no_noise_path=False,)

        return path, {
            'final_pos': final_pos,
            'action_norm': action_norm,
            'action_delta': action_delta,
            'action_low': self.action_space_low,
            'action_high': self.action_space_high,
            }
    
    def normalize_action(self, action_abs):
        # This assumes that norm action
        norm_action = (action_abs - self.action_space_low) / (self.action_space_high - self.action_space_low)
        assert np.all(norm_action >= 0.0) and np.all(norm_action <= 1.0)
        return norm_action

    def done(self, robot, err_th: float = 0.01):
        curr_pos, _ = self.current_position(robot)
        return np.linalg.norm(self.target_pos - curr_pos) < err_th

