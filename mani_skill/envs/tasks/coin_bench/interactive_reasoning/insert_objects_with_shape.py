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


@register_env("Tabletop-Insert-Objects-WithShape-v1", max_episode_steps=5000)
class InsertObjectsWithShapeEnv(UniversalTabletopEnv):
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

    description = "insert all the stick on the table into corresponding holes"
    workflow = [
        "pick the triangle and put it on the left hole",
        "pick the round one and put it to the round hole",
        "pick the square one and put it to the central hole",
    ]

    def __init__(
        self,
        *args,
        object_mass=0.5,  # Mass of the objects in kg
        object_friction=1.0,  # Friction coefficient of the objects
        insertion_threshold=0.02,  # Distance threshold for successful insertion
        container_scale=0.5,  # Scale of the container
        **kwargs,
    ):
        print("===== args ", args, kwargs)
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY],
            "rob": [],
            "iter": [Inter.PLAN]
        }
        object_mass = random.uniform(0.1, 10)

        # Object properties
        self.object_mass = object_mass
        self.object_friction = object_friction
        self.insertion_threshold = insertion_threshold
        self.container_scale = container_scale

        # Object paths
        self.cube_path = "configs/plugin_cube.json"
        self.cylinder_path = "configs/plugin_cylinder.json"
        self.triangle_path = "configs/plugin_triangle.json"
        self.container_path = "configs/plugin_container.json"
        # Check if files exist
        for path in [
            self.cube_path,
            self.cylinder_path,
            self.triangle_path,
            self.container_path,
        ]:
            if not os.path.exists(path):
                print(f"Warning: Asset file not found: {path}")

        # Object scales
        self.cube_scale = 0.5
        self.cylinder_scale = 0.5
        self.triangle_scale = 0.5

        # Store objects and their target positions
        self.objects = {}
        self.object_initial_poses = {}
        self.target_positions = {}

        super().__init__(*args, **kwargs)
        self.query_query = "Which hole is for the cylinder?"
        self.query_selection = {"A": "A hole", "B": "B hole", "C": "C hole"}
        self.query_answer = "A"

    def _load_scene(self, options):
        super()._load_scene(options)
        self.container = self.load_from_config(
            self.container_path, "container", body_type="static"
        )

        # Randomize container position on the table
        # table_size = self.table_size
        # container_x = randomization.uniform(
        #     -table_size[0] * 0.3, table_size[0] * 0.3
        # )
        # container_y = randomization.uniform(
        #     -table_size[1] * 0.3, table_size[1] * 0.3
        # )
        # container_height = 0.01  # Slightly above the table
        # # Set container pose
        #
        # container_pose = Pose(
        #     p=[0, 0 , 0],
        #     q=euler2quat(*create_orientation_from_degree(0, 0, 0))
        # )
        # self.container.set_pose(container_pose)

        # Create the objects
        self.objects["cube"] = self.load_from_config(
            self.cube_path, "cube", convex=True
        )
        self.objects["triangle"] = self.load_from_config(
            self.triangle_path, "triangle", convex=True
        )
        self.objects["cylinder"] = self.load_from_config(
            self.cylinder_path, "cylinder", convex=True
        )

        # self.objects["cube"] = self.load_from_config(self.cube_path, 'cube', convex=True,body_type='static')
        # self.objects["triangle"] = self.load_from_config(self.triangle_path, 'triangle', convex=True,body_type='static')
        # self.objects["cylinder"] = self.load_from_config(self.cylinder_path, 'cylinder', convex=True,body_type='static')
        self.a = self._create_target_marker([0, -0.15, 0.3], "a_marker", size=0.001)
        self.b = self._create_target_marker([0, 0, 0.3], "b_marker", size=0.001)
        self.c = self._create_target_marker([0, 0.1, 0.3], "c_marker", size=0.001)

        for obj in self.objects.values():
            print(obj.pose)
        # input()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode by placing objects and container"""
        # Create the container
        super()._initialize_episode(env_idx, options)
        #
        # # Set physical properties for objects
        # self.objects["cube"].set_pose(sapien.Pose(p=(0.3, 0, 0.3)))

        # self.set_pos(self.objects['cube'], (0.3, 0, 0.3))
        # self.set_pos(self.objects['cylinder'], (0.3, -0.3, 0.3))
        # self.set_pos(self.objects['triangle'], (0.3, 0.3, 0.3))

        object_init_x = -0.2
        # self.objects["cube"].set_pose(sapien.Pose(p=(object_init_x, 0, 0.0),q=euler2quat(*np.deg2rad([90,0 ,0]))))
        # self.objects["triangle"].set_pose(sapien.Pose(p=(object_init_x, 0.05, 0.0),q=euler2quat(*np.deg2rad([90,0 ,0])) ))
        # self.objects["cylinder"].set_pose(sapien.Pose(p=(object_init_x, -0.05, 0.0),q=euler2quat(*np.deg2rad([90,0 ,0]) )))
        self.objects["cube"].set_pose(
            sapien.Pose(
                p=(object_init_x, 0, 0.0), q=euler2quat(*np.deg2rad([90, 0, 0]))
            )
        )
        self.objects["triangle"].set_pose(
            sapien.Pose(
                p=(object_init_x, -0.3, 0.0), q=euler2quat(*np.deg2rad([90, 0, 0]))
            )
        )
        self.objects["cylinder"].set_pose(
            sapien.Pose(
                p=(object_init_x, 0.3, 0.0), q=euler2quat(*np.deg2rad([90, 0, 0]))
            )
        )
        # self.objects["triangle"] = self.load_from_config(self.triangle_path, 'triangle')

        # self.objects["cylinder"] = self.load_from_config(self.cylinder_path, 'cylinder')
        self.a.set_pose(sapien.Pose(p=[0, -0.05, 0.05], q=[1, 0, 0, 0]))
        self.b.set_pose(sapien.Pose(p=[0, 0, 0.05], q=[1, 0, 0, 0]))
        self.c.set_pose(sapien.Pose(p=[0, 0.05, 0.05], q=[1, 0, 0, 0]))
        # Store initial poses for reset
        for obj_name, obj in self.objects.items():
            if obj is not None:
                self.object_initial_poses[obj_name] = obj.pose

    def _get_success(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)

        # Check if all objects are in their target positions
        all_inserted = True
        insert_threshold = 0.1
        inserted_objects = []
        a_success = (
            self.calculate_object_distance(self.a, self.objects["cylinder"])
            < insert_threshold
        )
        b_success = (
            self.calculate_object_distance(self.b, self.objects["cube"])
            < insert_threshold
        )
        c_success = (
            self.calculate_object_distance(self.c, self.objects["triangle"])
            < insert_threshold
        )
        print(a_success, b_success, c_success)
        all_inserted = a_success and b_success and c_success
        all_static = True
        for obj in self.objects.values():
            if obj is not None:
                all_static = all_static and self.is_static(obj)

        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if all_inserted and all_static:
            success = torch.ones_like(success)
        return success
