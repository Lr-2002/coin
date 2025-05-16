from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
    load_articulation_from_urdf,
)


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Clean-For-Dinner-v1", max_episode_steps=5000)
class CleanForDinnerEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to arrange dinnerware items (bowl, fork, spoon) onto a plate.

    **Randomizations:**
    - The positions of the bowl, fork, and spoon are randomized on the table
    - The plate is fixed at the center of the table

    **Success Conditions:**
    - All dinnerware items (bowl, fork, spoon) are placed on the plate
    - The robot is static (velocity < 0.2)
    """

    description = "Arrange the bowl, fork onto the plate, clean for dinner "
    workflow = [
        "pick the bowl and put it on the plate",
        "pick the fork and put it on the bowl",
    ]

    def __init__(
        self,
        *args,
        success_threshold=0.05,  # Distance threshold for successful completion
        **kwargs,
    ):
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.SPATIALRELATE],
            "rob": [],
            "iter": []
        }

        # Set success threshold
        self.success_threshold = success_threshold

        # Set task description
        # Initialize objects
        self.plate = None
        self.bowl = None
        self.fork = None
        self.spoon = None

        super().__init__(*args, **kwargs)



        # Tags for object types


        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter


        self.tags = {


            "obj": [],


            "rob": [],


            "iter": []


        }
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [
            ],
            "rob": [
            ],
            "iter": [
            ]
        }

        self.query_query = "Which item should be placed on the plate first?"
        self.query_selection = {"A": "The bowl", "B": "The fork"}
        self.query_answer = "A"

    def _create_cuboid(self, name, half_size, mass, color, body_type="dynamic"):
        """Create a cuboid object (fork, spoon)"""
        # Create a builder for the cuboid
        builder = self.scene.create_actor_builder()

        # Add collision component
        builder.add_box_collision(half_size=half_size)

        # Add visual component with color
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color(color)
            builder.add_box_visual(half_size=half_size, material=material)
        except Exception as e:
            print(f"Warning: Could not set material: {e}")
            # Fallback with no material
            builder.add_box_visual(half_size=half_size)

        # Build the actor
        cuboid = _build_by_type(builder, name, body_type=body_type)

        # Set mass
        if body_type == "dynamic":
            cuboid.set_mass(mass)

        return cuboid

    def _create_cylinder(
        self, name, radius, half_length, mass, color, body_type="dynamic"
    ):
        """Create a cylindrical object (plate, bowl)"""
        # Create a builder for the cylinder
        builder = self.scene.create_actor_builder()

        # Add collision component
        builder.add_capsule_collision(radius=radius, half_length=half_length)

        # Add visual component with color
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color(color)
            builder.add_capsule_visual(
                radius=radius, half_length=half_length, material=material
            )
        except Exception as e:
            print(f"Warning: Could not set material: {e}")
            # Fallback with no material
            builder.add_capsule_visual(radius=radius, half_length=half_length)

        # Build the actor
        cylinder = _build_by_type(builder, name, body_type=body_type)

        # Set mass
        if body_type == "dynamic":
            cylinder.set_mass(mass)

        return cylinder

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Create a plate (static)
        self.plate = self.load_from_config(
            "configs/plate_new.json", "plate", convex=True
        )

        # Create a bowl (dynamic, non-convex)
        self.bowl = self.load_articulation_from_json("configs/bowl.json")
        # Create a fork (dynamic)
        self.fork = self.load_from_config("configs/fork.json", "fork", convex=True)
        # Create a spoon (dynamic)
        # self.spoon = self.load_from_config("configs/spoon.json", "spoon")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with positions for objects"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_height

        # Place the plate at the center of the table
        if self.plate is not None:
            plate_pos_x = 0.0
            plate_pos_y = 0.0
            plate_pos_z = (
                table_height + 0.03
            )  # Slightly above the table to avoid z-fighting

            self.plate.set_pose(
                sapien.Pose(
                    p=[plate_pos_x, plate_pos_y, plate_pos_z], q=[0.707, -0.707, 0, 0]
                )
            )

        # Place the bowl randomly on the table
        if self.bowl is not None:
            bowl_pos_x = self.np_random.uniform(-0.3, -0.1)
            bowl_pos_y = self.np_random.uniform(-0.1, -0.1)
            bowl_pos_z = table_height + 0.01  # Slightly above the table

            self.bowl.set_pose(
                sapien.Pose(
                    p=[bowl_pos_x, bowl_pos_y, bowl_pos_z], q=[0.707, 0.707, 0, 0]
                )
            )

        # Place the fork randomly on the table
        if self.fork is not None:
            fork_pos_x = self.np_random.uniform(0.1, 0.1)
            fork_pos_y = self.np_random.uniform(-0.2, 0.3)
            fork_pos_z = table_height + 0.01  # Slightly above the table

            # Random orientation for the fork
            fork_quat = euler2quat(0, 0, self.np_random.uniform(0, 2 * np.pi)).tolist()

            self.fork.set_pose(
                sapien.Pose(p=[fork_pos_x, fork_pos_y, fork_pos_z], q=fork_quat)
            )

        # Place the spoon randomly on the table
        if self.spoon is not None:
            spoon_pos_x = self.np_random.uniform(-0.3, -0.1)
            spoon_pos_y = self.np_random.uniform(0.0, 0.3)
            spoon_pos_z = table_height + 0.01  # Slightly above the table

            # Random orientation for the spoon
            spoon_quat = euler2quat(0, 0, self.np_random.uniform(0, 2 * np.pi)).tolist()

            self.spoon.set_pose(
                sapien.Pose(p=[spoon_pos_x, spoon_pos_y, spoon_pos_z], q=spoon_quat)
            )

    # def _get_obs_extra(self, info: Dict):
    #     """Get extra observations specific to this task"""
    #     obs = {}
    #
    #     # Add object positions to observations
    #     if self.plate is not None:
    #         obs["plate_position"] = np.array(self.plate.pose.p)
    #
    #     if self.bowl is not None:
    #         obs["bowl_position"] = np.array(self.bowl.pose.p)
    #
    #     if self.fork is not None:
    #         obs["fork_position"] = np.array(self.fork.pose.p)
    #
    #     if self.spoon is not None:
    #         obs["spoon_position"] = np.array(self.spoon.pose.p)
    #
    #     return obs
    #
    def _are_items_on_plate(self):
        """Check if all items are on the plate"""
        # if (
        #     self.plate is None
        #     or self.bowl is None
        #     or self.fork is None
        #     or self.spoon is None
        # ):
        #     return False

        plate_pos = np.array(self.plate.pose.p)
        plate_radius = 0.12  # Same as defined in _load_scene

        # Check bowl
        bowl_pos = np.array(self.bowl.pose.p)
        bowl_to_plate_dist = np.linalg.norm(bowl_pos[:2] - plate_pos[:2])
        bowl_on_plate = bowl_to_plate_dist < plate_radius * 0.5

        # Check fork
        fork_pos = np.array(self.fork.pose.p)
        fork_to_plate_dist = np.linalg.norm(fork_pos[:2] - plate_pos[:2])
        fork_on_plate = fork_to_plate_dist < plate_radius * 1

        # # Check spoon
        # spoon_pos = np.array(self.spoon.pose.p)
        # spoon_to_plate_dist = np.linalg.norm(spoon_pos[:2] - plate_pos[:2])
        # spoon_on_plate = spoon_to_plate_dist < plate_radius * 0.8
        #
        # # All items must be on the plate
        return bowl_on_plate and fork_on_plate

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        self.show_two_objects(self.bowl, self.plate)
        # Initialize success tensor
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # Check if all items are on the plate
        items_on_plate = self._are_items_on_plate()

        # # Check if the robot is static
        # robot_static = self.agent.is_static(0.2)
        #
        # Set success if both conditions are met
        # print(self.is_static(self.bowl), self.is_static(self.fork))
        static = self.is_static(self.bowl) and self.is_static(self.fork)
        if items_on_plate and static:
            success = torch.ones(len(env_idx), dtype=torch.bool, device=self.device)

        return success
