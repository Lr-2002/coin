from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random
from mani_skill.envs import sapien_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
import numpy as np
from mani_skill.utils import sapien_utils


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Insert-WithOrientation-v1", max_episode_steps=5000)
class InsertWithOrientationEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to insert three different shaped objects (cube, cylinder, triangle)
    into a container with matching holes.

    **Randomizations:**
    - The objects' positions are randomized on the table
    - The objects' orientations are randomized around the z-axis
    - The container position is randomized on the table

    **Success Conditions:**
    - All three objects are inserted into their corresponding holes in the container
    - The robot is static (velocity < 0.2)
    """

    description = "insert the board on the wall"
    workflow = ["rotate the stick along y-axis", "insert it to the hole"]

    def __init__(
        self,
        *args,
        object_mass=0.5,
        object_friction=1.0,
        insertion_threshold=0.02,
        **kwargs,
    ):
        # Only need mass/friction for the insertable rectangle
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT],
            "rob": [],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT]
        }
        self.object_mass = object_mass
        self.object_friction = object_friction
        self.insertion_threshold = insertion_threshold
        super().__init__(*args, **kwargs)
        self.query_query = (
            "What is the correct orientation for inserting the blue object?"
        )
        self.query_selection = {
            "A": "Rotate the object vertically to match the slot height",
            "B": "Keep the object flat on the table and push it in",
        }
        self.query_answer = "A"

    def _create_cuboid(self, name, half_size, mass, color, body_type="dynamic"):
        """Create a cuboid object (wall or rectangle)"""
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        try:
            material = sapien.render.RenderMaterial()
            material.set_base_color(color)
            builder.add_box_visual(half_size=half_size, material=material)
        except Exception as e:
            print(f"Warning: Could not set material: {e}")
            builder.add_box_visual(half_size=half_size)
        from mani_skill.utils.building.actors.common import _build_by_type

        cuboid = _build_by_type(builder, name, body_type=body_type)
        return cuboid

    def _load_scene(self, options):
        super()._load_scene(options)
        # Wall with a rectangular hole (made of 3 static cuboids)
        wall_center = [0.0, 0.2, 0.09]  # Raise wall so hole is above desk (z=0.09)
        wall_thickness = 0.1
        wall_height = 0.12
        wall_width = 0.24
        hole_width = 0.08
        hole_height = 0.06
        color_wall = [0.7, 0.7, 0.7, 1.0]
        color_rect = [0.2, 0.6, 0.9, 1.0]
        # Left segment
        left_half = [
            (wall_width - hole_width) / 2 / 2,
            wall_thickness / 2,
            wall_height / 2,
        ]
        left_center = [
            wall_center[0] - (hole_width + left_half[0] * 2) / 2,
            wall_center[1],
            wall_center[2],
        ]
        self.wall_left = self._create_cuboid(
            "wall_left", left_half, 2.0, color_wall, body_type="static"
        )
        self.wall_left.set_pose(sapien.Pose(left_center))
        # Right segment
        right_half = left_half
        right_center = [
            wall_center[0] + (hole_width + right_half[0] * 2) / 2,
            wall_center[1],
            wall_center[2],
        ]
        self.wall_right = self._create_cuboid(
            "wall_right", right_half, 2.0, color_wall, body_type="static"
        )
        self.wall_right.set_pose(sapien.Pose(right_center))
        # Top segment
        top_half = [
            hole_width / 2,
            wall_thickness / 2,
            (wall_height - hole_height) / 2 / 2,
        ]
        top_center = [
            wall_center[0],
            wall_center[1],
            wall_center[2] + (hole_height + top_half[2] * 2) / 2,
        ]
        self.wall_top = self._create_cuboid(
            "wall_top", top_half, 2.0, color_wall, body_type="static"
        )
        self.wall_top.set_pose(sapien.Pose(top_center))
        # Bottom segment (down wall)
        bottom_half = [
            hole_width / 2,
            wall_thickness / 2,
            (wall_height - hole_height) / 2 / 2,
        ]
        bottom_center = [
            wall_center[0],
            wall_center[1],
            wall_center[2] - (hole_height + bottom_half[2] * 2) / 2,
        ]
        self.wall_bottom = self._create_cuboid(
            "wall_bottom", bottom_half, 2.0, color_wall, body_type="static"
        )
        self.wall_bottom.set_pose(sapien.Pose(bottom_center))
        # Insertable rectangle (cube)
        rect_half = [
            hole_width / 2 * 0.8,
            wall_thickness / 2 * 3,
            hole_height / 2 * 0.8,
        ]
        self.insert_rect = self._create_cuboid(
            "insert_rect", rect_half, self.object_mass, color_rect, body_type="dynamic"
        )
        # Save for initialization
        self._wall_params = dict(
            wall_center=wall_center,
            wall_thickness=wall_thickness,
            wall_height=wall_height,
            wall_width=wall_width,
            hole_width=hole_width,
            hole_height=hole_height,
            rect_half=rect_half,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Place the rectangle at a random position on the table, away from the hole
        table_z = 0.02
        rect_start_pos = [0.0, -0.1, table_z + self._wall_params["rect_half"][2]]
        self.insert_rect.set_pose(sapien.Pose(rect_start_pos, [0.707, 0, 0.707, 0]))

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Check if all objects are in their target positions
        all_inserted = True
        inserted_objects = []

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        a, b = self.get_aabb(self.insert_rect)
        a = a[2]
        b = b[2]
        print(min(a, b), self.is_stable(self.insert_rect))
        if min(a, b) >= 0.05 and self.is_stable(self.insert_rect):
            success = torch.ones_like(success)
        return success
