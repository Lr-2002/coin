from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import sapien
import torch
from transforms3d import euler
from transforms3d.euler import euler2quat
import random

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import UniversalTabletopEnv
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
    load_articulation_from_urdf,
)

"""
This task requires the robot to find a specific book from a bookshelf placed on the table.
The bookshelf contains multiple books with different colors and sizes.
"""


@register_env("Tabletop-Pick-Book-FromShelf-v1", max_episode_steps=5000)
class PickBookFromShelfEnv(UniversalTabletopEnv):
    description = "Find and pick the book from the bookshelf and put it on the marker "

    def __init__(self, *args, **kwargs):
        self.bookshelf_config = "configs/Bookshelf.json"
        self.book_configs = [
            "configs/book1.json",
        ]
        self.cnt = 0
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # Load the bookshelf
        self.bookshelf = self.load_from_config(
            self.bookshelf_config,
            "bookshelf",
            body_type="static",  # Make the bookshelf static so it doesn't move
        )
        # Load books
        self.books = []
        for i, book_config in enumerate(self.book_configs):
            book = self.load_from_config(
                book_config,
                f"book_{i}",
                body_type="dynamic",  # Books can be picked up
                convex=True,
                # scale_override=0.03,
            )
            self.books.append(book)

        # Set the target book (will be randomly selected during initialization)
        self.target_book_idx = 0
        self.target_book = self.books[self.target_book_idx]
        print(self.target_book)
        # input()
        self.target_area = self._create_goal_area(
            sapien.Pose(p=(-0.15, 0, 0), q=(euler2quat(*np.deg2rad([0, 90, 0]))))
        )
        self.update_obj_register(self.bookshelf)
        self.update_obj_register(self.books)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        # Position the bookshelf on the table
        if self.bookshelf is not None:
            self.bookshelf.set_pose(
                sapien.Pose(
                    p=[0.35, 0.0, self.table_height + 0.3],  # Center of the table
                    q=euler2quat(
                        *np.deg2rad([90, 0, -90])
                    ),  # No rotation, using quaternion [w, x, y, z]
                )
            )

        # Get bookshelf dimensions to place books properly
        try:
            bookshelf_size = self.get_actor_size(self.bookshelf)
            shelf_width = bookshelf_size[1]
            shelf_depth = bookshelf_size[0]
            shelf_height = bookshelf_size[2]
            print("---- bookshelf size is ", bookshelf_size)
        except:
            # Default dimensions if we can't get the actual size
            shelf_width = 0.3
            shelf_depth = 0.2
            shelf_height = 0.4

        # Calculate shelf positions (assuming 3 shelves)
        shelf_positions = [
            self.table_height + shelf_height * 0.25,  # Bottom shelf
            self.table_height + shelf_height * 0.5,  # Middle shelf
            self.table_height + shelf_height * 0.75,  # Top shelf
        ]

        # Place books on the shelves
        for i, book in enumerate(self.books):
            # Determine which shelf to place the book on
            shelf_idx = i % len(shelf_positions)
            shelf_z = shelf_positions[shelf_idx]

            # Calculate position on the shelf
            # Distribute books evenly along the shelf width
            position_x = 0.25 + 0.0 * i
            position_y = -0.1 + 0.03 * i  # Place books toward the front of the shelf

            # Set book position and orientation
            book.set_pose(
                sapien.Pose(
                    p=[position_x, position_y, 0.01],
                    q=euler2quat(
                        *np.deg2rad([90, 0, -90])
                    ),  # Books standing upright, using quaternion [w, x, y, z]
                )
            )

        # Randomly select a target book
        # self.target_book_idx = random.randint(0, len(self.books) - 1)
        # self.target_book = self.books[self.target_book_idx]

    def _get_success(self, env_idx=None):
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        self.cnt += 1
        # markers = self.mark_aabb_corners(self.target_book, iter=self.cnt)
        # Success criteria: The target book is lifted above a certain height
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        # print(self.calculate_object_distance(self.target_book, self.target_area))
        # print(self.is_static(self.target_book))
        if self.calculate_object_distance(
            self.target_book, self.target_area
        ) <= 0.15 and self.is_static(self.target_book, 0.1):
            success = torch.ones_like(success)

        return success
