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


@register_env("Tabletop-Move-Cube-WithPivot-v1", max_episode_steps=5000)
class MoveCubeWithPivotEnv(UniversalTabletopEnv):
    description = "move the cube with the pivot to the marker "
    workflow = [
        "pick the stick and push it between two cylinder on the desk",
        "rotate the stick till the cube on the marker",
    ]

    def __init__(self, *args, **kwargs):
        print("===== args ", args, kwargs)
        # object_friction = random.uniform(0.1, 1)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.MASS],
            "rob": [Robot.ACT_NAV, Robot.DYN_TUNE],
            "iter": [Inter.PLAN, Inter.TOOL, Inter.FAIL_ADAPT],
        }
        self.object_friction = 10
        self.object_mass = 20.0

        super().__init__(*args, **kwargs)
        self.query_query = "Why we need the pivot to move the cube?"
        self.query_selection = {
            "A": "The cube is too heavy to lift",
            "B": "The cube is too far to grasp",
            "C": "The pivot is decorative and serves no functional purpose",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.cube = self._create_default_object(friction=self.object_friction)
        self.pivot = actors.build_twocolor_peg(
            self.scene,
            length=0.4,
            width=0.02,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="pivot",
            body_type="dynamic",
            initial_pose=sapien.Pose(
                p=[0, -0.1, 0], q=(euler2quat(*np.deg2rad([0, 0, 90])))
            ),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=0.05,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.2, 0, 0], q=(euler2quat(*np.deg2rad([0, 90, 0])))
            ),
        )
        self.pivot_base = actors.build_cylinder(
            self.scene,
            radius=0.02,
            half_length=0.05,
            color=np.array([0.1, 0.1, 0.1, 1]),
            name="pivot_base",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[0, 0.05, 0.0], q=(euler2quat(*np.deg2rad([0, 90, 0])))
            ),
        )

        self.pivot_base_2 = actors.build_cylinder(
            self.scene,
            radius=0.02,
            half_length=0.05,
            color=np.array([0.1, 0.1, 0.1, 1]),
            name="pivot_base_2",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[0.0, 0.2, 0.0], q=(euler2quat(*np.deg2rad([0, 90, 0])))
            ),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.cube.set_mass(self.object_mass)
        self.cube.set_pose(
            sapien.Pose(p=[0.2, 0.2, 0.015], q=(euler2quat(*np.deg2rad([0, 0, 0]))))
        )
        self.pivot.set_mass(0.2)

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(
            self.cube, self.goal_region
        ) <= 0.04 and self.is_stable(self.cube):
            success = torch.ones_like(success)
        return success

    # self.object_mass = randomization.uniform(0.1, 100)
