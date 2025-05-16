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


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


@register_env("Tabletop-Move-Line-WithStick-v1", max_episode_steps=5000)
class MoveLineWithStickEnv(UniversalTabletopEnv):
    """
    **Task Description:**
    A task where the objective is to use a stick to move a small cube along a straight line path
    made of fixed cubes.

    **Randomizations:**
    - The small cube's position is randomized at the start of the line path
    - The stick's position is randomized on the table

    **Success Conditions:**
    - The small cube is moved to the end of the line path
    - The robot is static (velocity < 0.2)
    """

    description = "Use the stick to move the small cube along the straight line path to the target position "
    workflow = [
        "pick the stick and aligh it with the line entry",
        "push the cube till the cube achieve the marker",
    ]

    def __init__(
        self,
        *args,
        success_threshold=0.05,  # Distance threshold for successful completion
        **kwargs,
    ):
        # Set success threshold
        
        # Tags for object types
        from mani_skill.envs.tasks.coin_bench.all_types import Obj, Robot, Inter
        self.tags = {
            "obj": [Obj.GEOMETRY, Obj.ORIENT],
            "rob": [Robot.ACT_NAV],
            "iter": [Inter.PLAN, Inter.TOOL]
        }
        self.success_threshold = success_threshold

        # Set task description

        # Define line path parameters
        self.path_cube_size = 0.04  # Size of each cube in the path
        self.path_cube_spacing = 0.001  # Small gap between path cubes
        self.path_length_horizontal = 7  # Number of cubes in horizontal path

        # Initialize objects
        self.path_cubes = {}
        self.small_cube = None
        self.stick = None

        # Target position (end of line path)
        self.line_start_x = None
        self.line_end_x = None

        super().__init__(*args, **kwargs)
        self.query_query = (
            "What approach should be used to move the cube along the path?"
        )
        self.query_selection = {
            "A": "Use the stick to push the cube",
            "B": "Directly push the cube with continuous contact",
            "C": "Pick the cube up and place it to the target",
        }
        self.query_answer = "B"

    def _create_cuboid(self, name, half_size, mass, color, body_type="dynamic"):
        """Create a cuboid object (path cube, small cube, or stick)"""
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
        # cuboid = builder.build(name=name)
        cuboid = _build_by_type(builder, name, body_type=body_type)
        # Set mass and density
        # cuboid.set_mass(mass)

        return cuboid

    def _create_line_path(self):
        """Create a straight line path with static cuboids (walls)"""
        self.path_cubes = {}
        slot_width = 0.06  # Channel width
        wall_height = 0.08
        wall_thickness = 0.02
        path_length = self.path_length_horizontal * self.path_cube_size
        # Left wall
        left_wall = self._create_cuboid(
            name="left_wall",
            half_size=[path_length / 2, wall_thickness / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],
            body_type="static",
        )
        left_wall.set_pose(sapien.Pose([0, -slot_width / 2, wall_height / 2]))
        self.path_cubes["left_wall"] = left_wall
        # Right wall
        right_wall = self._create_cuboid(
            name="right_wall",
            half_size=[path_length / 2, wall_thickness / 2, wall_height / 2],
            mass=2.0,
            color=[0.5, 0.5, 0.5, 1.0],
            body_type="static",
        )
        right_wall.set_pose(sapien.Pose([0, slot_width / 2, wall_height / 2]))
        self.path_cubes["right_wall"] = right_wall
        # (Optional) Floor
        # floor = self._create_cuboid(
        #     name="line_floor",
        #     half_size=[path_length / 2, slot_width / 2, 0.005],
        #     mass=5.0,
        #     color=[0.7, 0.7, 0.7, 1.0],
        #     body_type='static'
        # )
        # floor.set_pose(sapien.Pose([0, 0, 0.005]))
        # self.path_cubes['line_floor'] = floor
        # Save start/end x for goal
        self.line_start_x = -path_length / 2 + 0.02
        self.line_end_x = path_length / 2 - 0.02

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self._create_line_path()
        self.small_cube = self._create_cuboid(
            name="small_cube",
            half_size=[0.02, 0.02, 0.02],
            mass=0.1,
            color=[0.2, 0.8, 0.2, 1.0],
        )
        self.stick = self._create_cuboid(
            name="stick",
            half_size=[0.01, 0.15, 0.02],
            mass=0.02,
            color=[0.8, 0.4, 0.0, 1.0],
        )
        self.target_area = self._create_goal_area(
            sapien.Pose(p=(0.3, 0, 0), q=(0.707, 0, 0.707, 0))
        )  # No target area marker needed for line path

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Place cube at start of line
        start_pos = [self.line_start_x, 0, 0.04]
        self.small_cube.set_pose(sapien.Pose(start_pos, [1, 0, 0, 0]))
        # Optionally randomize stick position
        self.stick.set_pose(sapien.Pose([0, -0.2, 0.04], [1, 0, 0, 0]))

    def _get_success(self, env_idx=None):
        """Success if cube is at end of line path and is static"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        cube_pos = self.small_cube.pose.p[0]
        static = self.is_static(self.small_cube)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        if self.calculate_object_distance(
            self.small_cube, self.target_area
        ) <= 0.05 and self.is_stable(self.small_cube):
            success = torch.ones_like(success)
        return success
