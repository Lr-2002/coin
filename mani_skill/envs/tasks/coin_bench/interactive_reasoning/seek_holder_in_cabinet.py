from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
)

"""
Task: Find and pick up an apple that is inside a cabinet.

The robot needs to open the cabinet to reach and pick up the apple.
"""


@register_env("Tabletop-Seek-Holder-InCabinet-v1", max_episode_steps=5000)
class SeekHolderInCabinetEnv(UniversalTabletopEnv):
    description = (
        "Find the holder containing an eraser the cabinet, put it to the marker"
    )
    workflow = [
        "open cabinet door",
        "pick left holder",
        "pick right holder",
        "pick the holder containing the eraser",
        "put the eraser to the marker",
    ]

    def __init__(self, *args, **kwargs):
        # Set default friction for the apple
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.OBSTACLE, Obj.GEOMETRY, Obj.SEQ_NAV, Obj.MESH, Obj.SPATIALRELATE],
            "rob": [Robot.ACT_NAV, Robot.MORPH, Robot.PERSPECTIVE],
            "iter": [Inter.PLAN, Inter.HISTORY],
        }
        self.object_friction = random.uniform(0.1, 1)
        super().__init__(*args, **kwargs)
        self.query_query = "Which holder has a eraser?"
        self.query_selection = {"A": "A", "B": "B"}
        self.query_answer = "A"
        # Did it open the cabinet?
        # Did it go near to the holder with a eraser?
        # Did it pick up the eraser?
        # Dit it put the eraser to the marker

    def _load_scene(self, options: dict):
        """Load the scene with table, cabinet, and apple"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the cabinet using articulation loader
        self.cabinet = self._load_cabinet("1054")

        self.goal = self._create_goal_area()
        self.holder_1 = self.load_from_config("configs/pen_holder.json", "holder_1")
        self.holder_2 = self.load_from_config("configs/pen_holder.json", "holder_2")
        self.eraser = self.load_from_config(
            "configs/eraser.json", "eraser", convex=True
        )
        # Load the apple from config file

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with cabinet and apple"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Position the cabinet on the table
        # cabinet_position = torch.tensor([0.3, 0.0, 0.01], device=self.device)
        self.cabinet.set_pose(
            sapien.Pose(
                p=[-0.15, -0.5, 0.35],
                # p=[, 0.5, 0.35],
                q=euler2quat(0, 0, -np.pi / 2),  # No rotation
            )
        )
        self.set_articulation_joint(self.cabinet, "joint_0", 0.1)
        cabinet_position = self.cabinet.pose.p[0]
        self.holder_1.set_pose(sapien.Pose(p=[-0.1, -0.5, 0.1], q=[0.707, 0.707, 0, 0]))
        self.holder_2.set_pose(
            sapien.Pose(p=[-0.25, -0.5, 0.1], q=[0.707, 0.707, 0, 0])
        )
        self.eraser.set_pose(sapien.Pose(p=[-0.1, -0.5, 0.15], q=[0.707, 0.707, 0, 0]))

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(
            self.eraser, self.goal
        ) <= 0.05 and self.is_stable(self.eraser):
            success = torch.ones_like(success)
        #
        return success
