from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random
import math

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv


@register_env("Tabletop-Pick-Cylinder-WithObstacle-v1", max_episode_steps=5000)
class PickCylinderWithObstacleEnv(UniversalTabletopEnv):
    """
    Pick Cylinder With Obstacle Environment

    Task: Pick up the center cylinder surrounded by obstacle cylinders

    Features:
    1. Six cylinders in total
    2. Five cylinders arranged in a circle around one center cylinder
    3. Center cylinder has low friction and low mass
    4. Outer cylinders: 3 fixed, 2 dynamic with high friction and high mass
    5. All cylinders are positioned with their circular faces on the horizontal plane
    6. The center cylinder is positioned lower than the others
    """

    description = "pick up the center cylinder"
    workflow = ["push left every cylinder", "pick the center cylinder"]

    def __init__(self, *args, **kwargs):
        # Properties for the center cylinder
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.LOCK, Obj.MOVEABLE],
            "rob": [Robot.JOINT_AWARE],
            "iter": [Inter.PLAN, Inter.FAIL_ADAPT, Inter.HISTORY]
        }
        self.center_cylinder_friction = 0.3
        self.center_cylinder_mass = 0.2

        # Properties for the dynamic obstacle cylinders
        self.obstacle_cylinder_friction = 1.0
        self.obstacle_cylinder_mass = 1.0

        # Arrangement parameters
        self.cylinder_height = 0.15
        self.circle_radius = 0.05  # Radius of the circle for outer cylinders
        self.cylinder_radius = 0.02

        # Success criteria
        self.success_height_threshold = 0.1  # Height threshold for successful pickup

        # Initialize cylinders
        self.center_cylinder = None
        self.obstacle_cylinders = []
        super().__init__(*args, **kwargs)
        self.query_query = "How to move the center cylinder to the goal?"
        self.query_selection = {
            "A": "Pick up the center cylinder directly and move it to the goal",
            "B": "Find the dynamic obstacle cylinders and push them away to move the center cylinder to the goal",
            "C": "Put one of the dynamic cylinder to the goal",
        }
        self.query_answer = "B"

    def _load_scene(self, options: dict):
        """Load the scene with six cylinders"""
        super()._load_scene(options)

        # Create the center cylinder (low friction, low mass, positioned lower)
        self.center_cylinder = self._create_cylinder(
            name="center_cylinder",
            position=[0.0, 0.0, 0.01],  # Lower position
            is_dynamic=True,
            mass=self.center_cylinder_mass,
            friction=self.center_cylinder_friction,
        )

        # Create 5 obstacle cylinders arranged in a circle
        self.obstacle_cylinders = []

        # Calculate positions for the 5 cylinders in a circle
        for i in range(5):
            angle = 2 * math.pi * i / 5
            x = self.circle_radius * math.cos(angle)
            y = self.circle_radius * math.sin(angle)

            # Determine if this cylinder should be fixed or dynamic
            is_dynamic = i >= 3  # First 3 are fixed, last 2 are dynamic

            # Create the cylinder with a unique name
            cylinder = self._create_cylinder(
                name=f"obstacle_cylinder_{i}",
                position=[x, y, 0.06],  # Higher position
                is_dynamic=is_dynamic,
                mass=self.obstacle_cylinder_mass if is_dynamic else None,
                friction=self.obstacle_cylinder_friction if is_dynamic else None,
            )

            self.obstacle_cylinders.append(cylinder)
        self.goal = self._create_goal_area(position=[0, 0.2, 0])

    def _create_cylinder(
        self, name, position, is_dynamic=True, mass=None, friction=None
    ):
        """Create a cylinder object"""
        # Create a cylinder using the actor builder
        builder = self.scene.create_actor_builder()

        # Add a cylinder shape
        material = sapien.render.RenderMaterial()
        material.set_base_color([0.0, 0.8, 0.8, 1.0])

        builder.add_cylinder_visual(
            radius=self.cylinder_radius,
            half_length=self.cylinder_height / 2,
            material=material,
        )

        # Add collision shape
        builder.add_cylinder_collision(
            radius=self.cylinder_radius, half_length=self.cylinder_height / 2
        )

        # Build the actor with the name
        if is_dynamic:
            cylinder = builder.build(name=name)
        else:
            cylinder = builder.build_kinematic(name=name)

        # Set physical properties
        if is_dynamic:
            # Set mass if provided
            if mass is not None:
                cylinder.set_mass(mass)

            # Set damping
            try:
                cylinder.set_damping(linear=0.5, angular=0.5)
            except Exception as e:
                print(f"Warning: Could not set damping for {name}: {e}")

            # Set friction if provided
            if friction is not None:
                try:
                    for collision_shape in cylinder.get_collision_shapes():
                        collision_shape.set_friction(friction)
                except Exception as e:
                    print(f"Warning: Could not set friction for {name}: {e}")

        # Set initial pose (cylinder standing upright on the table)
        # Rotate 90 degrees around X-axis to make the circular faces horizontal
        # cylinder.set_pose(
        #     sapien.Pose(
        #         p=position,
        #         q=euler2quat(0, 0, 0).tolist()
        #     )
        # )

        return cylinder

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode"""
        super()._initialize_episode(env_idx, options)

        # Slightly randomize the position of the center cylinder
        center_pos = [
            0.0 + random.uniform(-0.02, 0.02),
            0.0 + random.uniform(-0.02, 0.02),
            0.00,  # Keep it lower
        ]

        if self.center_cylinder is not None:
            self.center_cylinder.set_pose(
                sapien.Pose(p=center_pos, q=euler2quat(0, np.pi / 2, 0).tolist())
            )

        # Slightly randomize positions of obstacle cylinders
        for i, cylinder in enumerate(self.obstacle_cylinders):
            if cylinder is not None:
                angle = 2 * math.pi * i / 5
                rand_radius = self.circle_radius + random.uniform(-0.01, 0.01)
                x = rand_radius * math.cos(angle)
                y = rand_radius * math.sin(angle)

                cylinder.set_pose(
                    sapien.Pose(
                        p=[x, y, 0.07],  # Keep them higher
                        q=euler2quat(0, np.pi / 2, 0).tolist(),
                    )
                )

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(
            self.center_cylinder, self.goal, axis=[0, 1]
        ) <= 0.04 and self.is_stable(self.center_cylinder):
            success = torch.ones_like(success)
        # Check if the center cylinder is picked up (height above threshold)
        # center_cylinder_pos = self.center_cylinder.pose.p
        #
        # # Task is successful if the center cylinder is lifted above the threshold
        # success = center_cylinder_pos[2] > self.success_height_threshold

        return {"success": success}
