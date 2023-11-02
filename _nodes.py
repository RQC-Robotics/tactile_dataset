"""Auxiliary nodes for the task."""
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from ur_env.scene.nodes import base
from ur_env.scene.nodes.robot.robotiq_gripper import RobotiqGripper

Observation = dict[str, np.ndarray]


class RealSense(base.Node):
    """Intel RealSense D455."""

    def __init__(self,
                 resolution: tuple[int, int],
                 output_shape: tuple[int, int] | None = None
                 ) -> None:
        config = rs.config()
        config.disable_all_streams()
        config.enable_stream(rs.stream.color,
                             width=resolution[0],
                             height=resolution[1],
                             framerate=30)
        self._pipeline = rs.pipeline()
        self._pipeline.start(config)
        self.output_shape = output_shape

    def get_observation(self) -> Observation:
        frame = self._pipeline.wait_for_frames()
        img = frame.get_color_frame()
        img = np.asanyarray(img.get_data()).copy()
        if (shape := self.output_shape) is not None:
            img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        return {'image': img}

    def observation_spec(self):
        raise RuntimeError

    def close(self) -> None:
        self._pipeline.stop()


class Robotiq2f85(base.Node):
    """Robotiq gripper."""

    def __init__(self, address: tuple[str, int]) -> None:
        gripper = RobotiqGripper()
        gripper.connect(*address)
        gripper.activate(auto_calibrate=False)
        self._gripper = gripper

    def move(self, pos: int, speed: int, force: int) -> tuple[bool, int]:
        ack, cmd_pos = self._gripper.move(pos, speed, force)
        while self._get_var(self._gripper.PRE) != cmd_pos:
            time.sleep(0.001)
        return ack, cmd_pos

    def get_observation(self) -> Observation:
        pos, obj_stat = self.get_pos_and_obj()
        Status = RobotiqGripper.ObjectStatus
        obs = {
            'pos': pos,
            'object_detected': Status(obj_stat) == Status.STOPPED_INNER_OBJECT
            }
        return {k: np.atleast_1d(v) for k, v in obs.items()}

    def get_pos_and_obj(self):
        return (
            self._get_var(RobotiqGripper.POS),
            self._get_var(RobotiqGripper.OBJ)
        )

    def observation_spec(self):
        raise RuntimeError

    def __getattr__(self, name):
        return getattr(self._gripper, name)
