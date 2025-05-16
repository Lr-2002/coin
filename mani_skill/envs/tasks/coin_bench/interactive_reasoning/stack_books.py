from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.coin_bench import UniversalTabletopEnv


def degree2rad(angle):
    return angle / 180 * np.pi


@register_env("Tabletop-Stack-Books-v1", max_episode_steps=5000)
class StackBooksEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to stack all books on the table.
    The books are initially scattered on the table, and the robot needs to stack them.

    **Randomizations:**
    - The initial positions of the books are randomized on the table

    **Success Conditions:**
    - All books are stacked on top of each other
    - The robot is static (velocity < 0.2)
    """

    description = "Stack all things on the table"
    workflow = [
        "pikc the green book on the table",
        "pick the red book on the green one",
    ]

    def __init__(
        self,
        *args,
        success_threshold=0.05,  # Distance threshold for successful completion
        num_books=3,  # Number of books to stack
        **kwargs,
    ):
        # Set success threshold
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.SCALE, Obj.ORIENT],
            "rob": [Robot.PERSPECTIVE],
            "iter": [Inter.PLAN]
        }
        self.success_threshold = success_threshold

        # Set task description

        # Number of books to stack
        self.num_books = num_books

        # Initialize objects
        self.books = []

        # Book configurations
        self.book_configs = [
            "configs/book1.json",
            "configs/book2.json",
            # 1 "configs/eraser.json",
        ]
        self.box_config = "configs/box.json"
        super().__init__(*args, **kwargs)
        self.query_query = "Which book should be stacked on top of the other?"
        self.query_selection = {
            "A": "The red book should be placed on top of the green book",
            "B": "The green book should be placed on top of the red book",
        }
        self.query_answer = "A"

    def _load_scene(self, options: dict):
        """Load the scene with table and books"""
        # Load the basic scene with table
        super()._load_scene(options)

        # Load the books (dynamic)
        self.books = []
        for i in range(min(self.num_books, len(self.book_configs))):
            book = self.load_from_config(
                self.book_configs[i % len(self.book_configs)],
                f"book_{i}",
                body_type="dynamic",
                convex=True,
                # scale_override=0.03
            )
            self.books.append(book)

        self.box = self.load_from_config(self.box_config, "box", "static", convex=True)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode with positions for books"""
        # Initialize the table scene and robot
        super()._initialize_episode(env_idx, options)

        # Get table height
        table_height = self.table_height + 0.13

        self.box.set_pose(sapien.Pose(p=(0, 0, 0.03), q=(0.7070, 0, 0, 0.707)))
        # Place books randomly on the table
        if self.books:
            # Define a grid on the table for initial book placement
            grid_size = max(2, int(np.ceil(np.sqrt(len(self.books)))))
            grid_spacing = 0.18

            # Calculate the starting position for the grid
            start_x = -grid_spacing * (grid_size - 1) / 2
            start_y = -grid_spacing * (grid_size - 1) / 2

            for i, book in enumerate(self.books):
                # Calculate grid position
                grid_x = i % grid_size
                grid_y = i // grid_size

                # Calculate position with some randomness
                pos_x = (
                    start_x
                    + grid_x * grid_spacing
                    + self.np_random.uniform(-0.03, 0.03)
                )
                pos_y = (
                    start_y
                    + grid_y * grid_spacing
                    + self.np_random.uniform(-0.03, 0.03)
                )
                pos_z = table_height + 0.03  # Slightly above the table

                # Random orientation around z-axis
                orientation = list(euler2quat(0, np.pi / 2, 0))

                # Set book position and orientation
                book.set_pose(sapien.Pose(p=[pos_x, pos_y, pos_z], q=orientation))

    def pairwise_combinations(self, lst):
        return [
            [lst[i], lst[j]] for i in range(len(lst)) for j in range(i + 1, len(lst))
        ]

    def _check_books_stacked(self):
        """Check if the books are stacked on top of each other"""
        if not self.books:
            return False
        lst = self.pairwise_combinations(self.books)
        is_stacked = True
        for a, b in lst:
            print(a.name, b.name)
            if self.calculate_object_distance(a, b) >= 0.08:
                is_stacked = False
                break

        return is_stacked

    def _get_success(self, env_idx=None):
        """Check if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Initialize success tensor
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

        # Check if books are stacked
        books_stacked = self._check_books_stacked()

        # Check if the robot is static
        robot_static = self.agent.is_static(0.2)

        # Set success if both conditions are met
        if books_stacked and robot_static:
            success = torch.ones(len(env_idx), dtype=torch.bool, device=self.device)

        return success
