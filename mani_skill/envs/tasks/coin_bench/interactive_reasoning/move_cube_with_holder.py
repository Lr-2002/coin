from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d import euler
from transforms3d.euler import euler2quat
import random
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views

from mani_skill.utils.building import actors

from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv


@register_env("Tabletop-Move-Cube-WithHolder-v1", max_episode_steps=5000)
class MoveCubeWithHolderEnv(UniversalTabletopEnv):
    description = "move the cube to the marker and put the holder on the cube "

    workflow = [
        "pull the holder",
        "pick the cube and put it on the holder",
        "pick the holder and put it on the cube",
    ]

    def __init__(self, *args, **kwargs):
        print("===== args ", args, kwargs)
        # object_friction = random.uniform(0.1, 1)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.SCALE, Obj.GEOMETRY, Obj.SEQ_NAV, Obj.SPATIALRELATE],
            "rob": [],
            "iter": [Inter.PLAN]
        }
        self.holder_config = "configs/pen_holder.json"
        super().__init__(*args, **kwargs)
        self.query_query = "What's the best sequence of actions for this task?"
        self.query_selection = {
            "A": "Move the cube to the marker first, then place the holder on top",
            "B": "Move the holder first, then place the cube to the marker, finally put the holder on the cube",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cube = self._create_default_object(size=0.015)
        self.holder = self.load_from_config(
            self.holder_config, "holder", convex=False, scale_override=0.4
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=0.05,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.0, 0, 0], q=(euler2quat(*np.deg2rad([0, 90, 0])))
            ),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # self.cube.set_mass(self.object_mass)
        self.cube.set_pose(
            sapien.Pose(p=[0.0, 0.2, 0.015], q=(euler2quat(*np.deg2rad([0, 0, 0]))))
        )
        self.holder.set_pose(
            sapien.Pose(p=[0.0, 0.0, 0.02], q=(euler2quat(*np.deg2rad([-90, 0, 0]))))
        )
        # self.holder.set_mass(0.4)
        # self.cube.set_friction(self.object_friction)

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        self.show_two_objects(self.holder, self.cube)
        print(
            self.calculate_object_distance(self.cube, self.goal_region) <= 0.05,
            self.calculate_object_distance(self.cube, self.holder, axis=[0, 1]) <= 0.05,
            self.is_stable(self.cube),
            self.is_stable(self.holder),
        )
        if (
            self.calculate_object_distance(self.cube, self.goal_region) <= 0.05
            and self.is_stable(self.cube)
            and self.calculate_object_distance(self.cube, self.holder, axis=[0, 1])
            <= 0.05
            and self.is_stable(self.holder)
        ):
            success = torch.ones_like(success)
        return success

    # self.object_mass = randomization.uniform(0.1, 100)
