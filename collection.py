import time
import pathlib
import itertools
import dataclasses
from enum import IntEnum
from typing import ClassVar, Literal, TypedDict

import tree
import numpy as np
from ur_env.scene import nodes, Scene
import _nodes


class HardnessClass(IntEnum):
    Soft = 0
    Hard = 1


# TODO: object description is mostly missing. Consider adding suitable fields.
class DatasetItem(TypedDict, total=False):
    """The following data is recorded on touch.

    Types can be misleading since resulting item may have an additional
      leading dim that comes from series of touches w/ different forces.
      Sensor fields:
        pos: gripper position in range [0, 255].
        object_detected: rq_is_object_detected() output.
        (left|right)_sensor: Digit RGB output.
        wrist_camera: RealSense D455 RGB image.

      Object fields:
        obj_hardness: subjective hardness class of an object.
    """

    pos: int
    object_detected: bool
    left_sensor: np.uint8
    right_sensor: np.uint8
    wrist_camera: np.uint8

    obj_hardness: HardnessClass
    # category, material, slippery?


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
    """Fix related variables and hparams."""

    CONFIG_NAME: ClassVar[str] = 'config.npz'
    ITEMS_DIR: ClassVar[str] = 'items/'

    relaxation_time: float = 0.5
    dataset_dir: str = 'dataset/'
    seed: int = 1
    # Robotiq_2f85
    address: tuple[str, int] = ('192.168.1.179', 63352)
    force: int = 5
    pos: int = 255
    speed: int = 10
    fps: int = 30
    # Digits
    left_digit_serial: str = 'D20591'
    right_digit_serial: str = 'D20589'
    digit_resolution: Literal['VGA', 'QVGA'] = 'QVGA'
    # RealSense
    wrist_camera_resolution: tuple[int, int] = (640, 480)
    resize_image: tuple[int, int] | None = (128, 128)


def make_scene(cfg: Config) -> Scene:
    """Connect all required sensors."""
    gripper = _nodes.Robotiq2f85(address=cfg.address)
    left_sensor = nodes.Digit(serial=cfg.left_digit_serial,
                              resolution=cfg.digit_resolution)
    right_sensor = nodes.Digit(serial=cfg.right_digit_serial,
                               resolution=cfg.digit_resolution)
    wrist_camera = _nodes.RealSense(resolution=cfg.wrist_camera_resolution,
                                    output_shape=cfg.resize_image)
    return Scene(gripper=gripper,
                 left_sensor=left_sensor,
                 right_sensor=right_sensor,
                 wrist_camera=wrist_camera
                 )


def describe_object(cfg: Config) -> DatasetItem:
    """Input info about the object."""
    del cfg  # may use config later to ignore some fields.
    print('Describe the object.')
    print('Hardness: Soft=0, Hard=1.')
    hardness = HardnessClass(int(input()))
    return DatasetItem(
        obj_hardness=hardness
    )


def scan_object(cfg: Config, scene: Scene) -> DatasetItem:
    scene.initialize_episode(np.random.default_rng(cfg.seed))
    g = scene.gripper
    g.move_and_wait_for_pos(3, 255, 255)
    g.move(cfg.pos, cfg.speed, cfg.force)
    obj_status = g.ObjectStatus.MOVING
    spf = 1. / cfg.fps
    obss = []
    while obj_status == g.ObjectStatus.MOVING:
        pos, obj_status = g.get_pos_and_obj()
        obs = scene.get_observation()
        obss.append(obs)
        time.sleep(spf)
    obs = tree.map_structure(lambda *t: np.stack(t), *obss)
    return DatasetItem(
            pos=obs['gripper/pos'].astype(np.uint8),
            object_detected=obs['gripper/object_detected'].astype(np.bool_),
            left_sensor=obs['left_sensor/sensor'],
            right_sensor=obs['right_sensor/sensor'],
            wrist_camera=obs['wrist_camera/image']
           )


def run(cfg: Config) -> None:
    """Main data collection loop."""
    dataset_dir = pathlib.Path(cfg.dataset_dir).absolute()
    items_dir = dataset_dir / cfg.ITEMS_DIR
    if not items_dir.exists():
        items_dir.mkdir(parents=True)
    if not (path := dataset_dir / cfg.CONFIG_NAME).exists():
        np.savez(path, **dataclasses.asdict(cfg))
    scene = make_scene(cfg)
    cont_idx = items_dir.iterdir()
    cont_idx = len(list(cont_idx))
    try:
        for idx in itertools.count(cont_idx):
            obj_desc = describe_object(cfg)
            sensor_data = scan_object(cfg, scene)
            data = sensor_data | obj_desc
            np.savez_compressed(items_dir / f'{idx:04d}', **data)
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    run(Config())
