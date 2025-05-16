from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
import json
from transforms3d.euler import euler2quat
import random
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


@register_env("Tabletop-Stack-LongObjects-v1", max_episode_steps=5000)
class StackLongObjectsEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to stack three long objects to reach the maximum height.

    **Randomizations:**
    - The objects' positions are randomized on the table
    - The objects' orientations are randomized around the z-axis

    **Success Conditions:**
    - The objects are stacked to reach the maximum height (0.05 + 0.2 + 0.4 = 0.65)
    - The stack is stable (objects are not falling)
    - The robot is static (velocity < 0.2)
    """

    description = "stack all the objects to make it most high"
    workflow = [
        "stack the mid cube and the longest cube",
        "stack the shortest cube and the mid one ",
    ]

    def __init__(
        self,
        *args,
        cube_config="configs/yellow_cube.json",
        **kwargs,
    ):
        # Load the base cube configuration
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.SCALE, Obj.ORIENT, Obj.GEOMETRY, Obj.OBSTACLE],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.FAIL_ADAPT, Inter.PLAN],
        }
        self.cube_config = cube_config
        self.cube_config_data = None

        # Define the three objects with different heights
        self.object_sizes = [
            (0.05, 0.05, 0.05),  # Small cube: 0.05 x 0.05 x 0.05
            (0.05, 0.05, 0.07),  # Medium object: 0.05 x 0.05 x 0.2
            (0.05, 0.05, 0.1),  # Long object: 0.05 x 0.05 x 0.4
        ]

        # Calculate the maximum possible stack height
        self.max_stack_height = sum(size[2] for size in self.object_sizes)
        # List to store the objects
        self.objects = []

        # Initialize with the first object as the main object
        super().__init__(*args, **kwargs)
        self.query_query = "Which cube is the highest?"
        self.query_selection = {"A": "A", "B": "B", "C": "C"}
        self.query_answer = "C"
        # Did it pick up one of the cubes
        # Did it stack the cubes?

    def _load_scene(self, options: dict):
        """Load the scene with table and objects"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Create the three objects with different heights
        self.objects = []

        # Create objects with different heights
        for i, size in enumerate(self.object_sizes):
            # Create a custom-sized object
            obj = self._create_custom_object(size, f"cube_{i}")
            self.objects.append(obj)

        # Set the first object as the main object for the parent class
        self.object = self.objects[0]

        # Create target marker for stacking

    def _create_custom_object(self, size, name):
        """Create a custom-sized object with the specified dimensions"""
        builder = self.scene.create_actor_builder()

        # Get half sizes for SAPIEN
        half_size = [s / 2 for s in size]

        # Add collision component
        builder.add_box_collision(half_size=half_size)

        # Add visual component (yellow box)
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color([1, 1, 0, 1])  # Yellow color with alpha=1
            builder.add_box_visual(half_size=half_size, material=material)
        except TypeError:
            # Fallback for older SAPIEN versions
            try:
                # Try with color parameter
                builder.add_box_visual(half_size=half_size, color=[1, 1, 0, 1])
            except TypeError:
                # Fallback with no color
                builder.add_box_visual(half_size=half_size)

        # Create the actor
        actor = builder.build(name=name)

        # Set physical properties from the cube config
        mass = 0.5
        friction = 1.0
        if self.cube_config_data:
            mass = self.cube_config_data.get("mass", 0.5)
            friction = self.cube_config_data.get("friction", 1.0)

        # Set mass proportional to volume
        volume = size[0] * size[1] * size[2]
        base_volume = 0.05 * 0.05 * 0.05
        scaled_mass = mass * (volume / base_volume)

        try:
            actor.set_mass(scaled_mass)
            actor.set_damping(linear=0.5, angular=0.5)
        except Exception as e:
            print(f"Warning: Could not set some physical properties: {e}")

        # Set friction
        try:
            for collision_shape in actor.get_collision_shapes():
                collision_shape.set_friction(friction)
        except Exception as e:
            print(f"Warning: Could not set friction: {e}")

        return actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode with randomized object positions"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_scene.table_height

        # Randomize object positions on table
        xy_range = 0.1
        min_distance = 0.07  # Minimum distance between objects

        # Place objects randomly on the table
        object_positions = []
        for i, obj in enumerate(self.objects):
            # Try to find a valid position for the object
            max_attempts = 20
            valid_position = False

            for _ in range(max_attempts):
                # Random position on table
                pos_x = torch.rand(1).item() * xy_range * 2 - xy_range
                pos_y = torch.rand(1).item() * xy_range * 2 - xy_range
                pos_z = 0  # Place on table

                # Check if too close to other objects
                too_close = False
                for other_pos in object_positions:
                    dist = np.sqrt(
                        (pos_x - other_pos[0]) ** 2 + (pos_y - other_pos[1]) ** 2
                    )
                    if dist < min_distance:
                        too_close = True
                        break

                if not too_close:
                    # Valid position found
                    object_positions.append([pos_x, pos_y, pos_z])
                    valid_position = True
                    break

            if not valid_position:
                # If no valid position found after max attempts, just place it somewhere
                pos_x = (i - 1) * 0.15  # Spread objects along x-axis
                pos_y = 0
                pos_z = 0.0
                object_positions.append([pos_x, pos_y, pos_z])

            # Randomize orientation (only around z-axis)
            z_rotation = torch.rand(1).item() * 2 * np.pi
            orientation = [0, 0, z_rotation]

            # Set object pose
            obj_pose = sapien.Pose(
                p=[pos_x, pos_y, pos_z],
                q=euler2quat(0, 0, z_rotation),
            )
            obj.set_pose(obj_pose)

        # Set target marker at a random position on the table
        target_x = torch.rand(1).item() * xy_range * 2 - xy_range
        target_y = torch.rand(1).item() * xy_range * 2 - xy_range
        target_z = table_height + 0.001  # Just above table

        target_pose = sapien.Pose(
            p=[target_x, target_y, target_z],
            q=[1, 0, 0, 0],
        )
        self.target_position = [target_x, target_y, target_z]

    def _get_obs_extra(self, info: Dict):
        """Get task-specific observations"""
        obs = super()._get_obs_extra(info)

        # Add positions and orientations of all objects
        object_positions = []
        object_orientations = []
        object_velocities = []

        # for obj in self.objects:
        #     obj_pose = obj.get_pose()
        #     obj_vel = obj.get_velocity()
        #     obj_ang_vel = obj.get_angular_velocity()
        #
        #     object_positions.append(obj_pose.p)
        #     object_orientations.append(obj_pose.q)
        #     object_velocities.append(np.concatenate([obj_vel, obj_ang_vel]))
        #
        # # Flatten and add to observations
        # obs["object_positions"] = np.array(object_positions, dtype=np.float32).flatten()
        # obs["object_orientations"] = np.array(object_orientations, dtype=np.float32).flatten()
        # obs["object_velocities"] = np.array(object_velocities, dtype=np.float32).flatten()
        #
        # Add target position
        obs["target_position"] = np.array(self.target_position, dtype=np.float32)

        # Add current stack height
        obs["stack_height"] = np.array(
            [self._calculate_stack_height()], dtype=np.float32
        )

        return obs

    def _calculate_stack_height(self):
        """Calculate the current height of the stack"""
        # Get table height
        stack_height = 0
        for cube in self.objects:
            aabb = self.get_aabb(cube)
            # breakpoint()
            stack_height = max(aabb[1][2], stack_height)
        return stack_height

    def _get_success(self, env_idx=None):
        """Evaluate task success"""
        #
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        # Calculate current stack height
        stack_height = self._calculate_stack_height()

        # Success rate is proportional to stack height
        success_rate = stack_height / self.max_stack_height

        # Check if stack is stable
        is_stable = True
        for obj in self.objects:
            if not self.is_stable(obj):
                is_stable = False
        # # Check if robot is static
        # robot_velocity = torch.norm(self.robot.get_velocity()).item()
        # is_robot_static = robot_velocity < 0.2
        # Success if stack height is at least 90% of maximum and stack is stable
        suc = success_rate >= 0.8 and is_stable
        print(self.max_stack_height, stack_height, success_rate, is_stable, suc)

        if suc:
            success = torch.ones_like(success)

        return success
