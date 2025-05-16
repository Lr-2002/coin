from typing import Any, Dict, List, Optional, Tuple, Union
import os
from cv2 import borderInterpolate
import numpy as np
import sapien
import torch
import json
from transforms3d import euler
from transforms3d.euler import euler2quat
import random
from mani_skill.envs import sapien_env
from mani_skill.utils.building import actors
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.envs.tasks.coin_bench.primitive_actions.pick_place import PickPlaceEnv


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Pull-Pivot-v1", max_episode_steps=5000)
class PullPivotEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to pull the pivot to target area.
    
    **Randomizations:**
    - The pen holder position is randomized on the table
    
    **Success Conditions:**
    - Success is measured by the pen holder opening up
    - The robot is static (velocity < 0.2)
    """
    
    description = "pull the pivot to the target area"
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args, **kwargs
        )
    
    def _load_scene(self, options: dict):
        """Load the scene with table, container, and balls"""
        # Load the basic scene with table and container
        super()._load_scene(options)
        
        # Create the pivot
        self.pivot = actors.build_twocolor_peg(
            self.scene,
            length=0.2,
            width=0.01,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="pivot",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.3, -0.1, 0]),
        )
        self.target_area = self._create_goal_area()

   
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized container and ball positions"""
        # Initialize the table scene, robot, and container
        super()._initialize_episode(env_idx, options)
        
        table_range = 0.1  # Range for ball placement on table
        min_distance = 0.05  # Minimum distance between balls
        
    
   
    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        # Count balls in container
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(self.pivot, self.target_area) <= 0.05 and self.is_static(self.pivot):
            success = torch.ones_like(success) 
        return {"success": success}
