import base64
from math import isfinite
from sys import prefix
from loguru import logger
from typing import Any, Dict, List, Optional, Tuple, Union
import os
from warnings import catch_warnings
import cv2
from cv2 import add, line, threshold
from git import objects
from git.objects.tree import TreeCacheTup
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat, quat2euler
from trimesh.util import is_instance_named
import pysnooper
from mani_skill.envs import sapien_env
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import articulation
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.building.articulations.articulation_loader import (
    load_articulation,
    load_articulation_from_json,
    load_articulation_from_urdf,
)
from mani_skill.utils.vis3d import BBoxObject, BBoxVisualizer
# import logging
#
# logging.getLogger("mani_skill").setLevel(logging.ERROR)
#
from .all_types import Obj, Robot, Inter
    


def degree2rad(angle):
    return angle / 180 * np.pi


def create_orientation_from_degree(r, p, y):
    return degree2rad(r), degree2rad(p), degree2rad(y)


class UniversalTabletopEnv(BaseEnv):
    """
    **Base Class for Universal Tabletop Manipulation Tasks**

    This class provides a foundation for creating tabletop manipulation tasks with:
    - Franka Panda robot
    - Two cameras (wrist-mounted and fixed-position)
    - Support for loading GLB/USD assets with configurable properties

    The class handles:
    1. Setting up the environment with a table and robot
    2. Configuring cameras for RGBD perception
    3. Loading and configuring assets with proper collision meshes
    4. Providing a framework for task success conditions and rewards

    Subclasses should implement:
    - _initialize_episode: Task-specific initialization
    - evaluate: Task-specific success conditions
    - compute_dense_reward: Task-specific rewards
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    _sample_video_link = "https://github.com/Lr-2002/COIN_videos/blob/main/medias"
    description = "This is the unviersal task"
    workflow = []
    tags = {
        "obj": [],
        "rob": [],
        "iter": []
    }
    long_name_map = {
        # Object-Centric Reasoning - Physical Property Inference
        Obj.MASS: "Mass estimation",
        Obj.FRICTION: "Friction coefficient assessment",
        Obj.SCALE: "Scale analysis",
        Obj.MOVEABLE: "Movable object classification",
        
        # Object-Centric Reasoning - Spatial Reasoning
        Obj.OBSTACLE: "Obstacle handling",
        Obj.ORIENT: "Orientation analysis",
        Obj.SPATIALRELATE: "Spatial relationship analysis",

        # Object-Centric Reasoning - Mechanism Understanding
        Obj.LOCK: "Locking system comprehension",
        Obj.KINEMATIC: "Kinematic constraint inference",
        Obj.SEQ_NAV: "Sequential mechanism navigation",
        
        # Object-Centric Reasoning - Visual Reasoning
        Obj.GEOMETRY: "Geometric reasoning",
        Obj.MESH: "Visual comparison",
        
        # Robot-Centric Reasoning - Embodiment Awareness
        Robot.MORPH: "Morphological reasoning",
        Robot.PERSPECTIVE: "Perceptual perspective optimization",
        Robot.JOINT_AWARE: "Kinematic constraint awareness",
        
        # Robot-Centric Reasoning - Control Optimization
        Robot.DYN_TUNE: "Dynamic response tuning",
        Robot.ACT_NAV: "Action space navigation",
        Robot.SKIL_APAPT: "Skill adaptation",

        # Compositional Reasoning Capabilities
        Inter.TOOL: "Tool-mediated problem solving",
        Inter.FAIL_ADAPT: "Failure-driven adaptation", 
        Inter.PLAN: "Hierarchical planning",
        Inter.HISTORY: "Experience utilization",
    }
    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",  # Default to Panda with wrist camera
        robot_init_qpos_noise=0.0,
        use_external_camera=True,  # Whether to use an external camera
        external_camera_pose=None,  # Optional custom pose for external camera
        wrist_camera_fov=np.pi / 3,  # Field of view for wrist camera
        external_camera_fov=np.pi / 3,  # Field of view for external camera
        camera_width=448,  # Camera resolution width
        camera_height=448,  # Camera resolution height
        env_debug=False,
        marker_collision=False,
        **kwargs,
    ):
        # print(kwargs)
        # breakpoint()
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.use_external_camera = use_external_camera
        self.external_camera_pose = external_camera_pose
        self.wrist_camera_fov = wrist_camera_fov
        self.external_camera_fov = external_camera_fov
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.table_height = 0.0
        # Initialize the scene assets list
        self.called_obj = False
        self.scene_assets = []
        self.description = (
            "this is the universal task"
            if not hasattr(self, "description")
            else self.description
        )
        self.workflow = [] if not hasattr(self, "workflow") else self.workflow
        self.logger = logger
        self.actor_reg = []
        self.articulation_reg = []
        self.debug = env_debug
        self.debug_cnt = 0
        from pathlib import Path
        


        self.all_objects = []
        self.asset_path_prefix = Path(
            __file__
        ).parent.parent.parent.parent.parent.resolve()
        # self.log("------ self.asset_path_prefix", self.asset_path_prefix)
        # self.start_rr()
        self.object_dict = {}
        self.marker_collision = marker_collision
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.stable_dict = {}
        if self.debug:
            self.create_coordinate_frame()

        self.query_image_path_task = None
        self.query_query = None
        self.query_selection = None
        self.query_answer = None
    def extend_tags(self):
        """Convert short Enum tags into their long string names"""
        self.extended = {
            domain: [self.long_name_map.get(tag, str(tag)) for tag in tags]
            for domain, tags in self.tags.items()}
        return self.extended
    def encode_image(self, image_array):
        """Encode image array to base64 string for LLM input."""
        # If image_array is already bytes (file content), decode it directly
        if isinstance(image_array, bytes):
            # Just base64 encode the bytes directly
            return base64.b64encode(image_array).decode("utf-8")

        # Otherwise process as numpy array
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Encode image to JPEG
        success, buffer = cv2.imencode(".jpg", image_array)
        if not success:
            logger.error("Failed to encode image")
            return None
        # Convert to base64
        base64_string = base64.b64encode(buffer).decode("utf-8")
        return base64_string

    def build_image(self, image_path):
        """
        Load the image from the image path with base64 encoding

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Base64 encoded image string
        """
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                self.log(f"Warning: Image file not found: {image_path}")
                return None

            # Read the image file
            with open(image_path, "rb") as f:
                print(f"image type: {type(f)}")
                image_bytes = f.read()

            # Encode the image
            image_base64 = self.encode_image(image_bytes)
            return image_base64
        except Exception as e:
            self.log(f"Error in build_image: {e}")
            return None

    def build_instruction(
        self,
        query="which one part contain the target object?",
        selection="""{"A": The object is in A " , "B": "The object is in B " , "C": "There is no target object in the cabinet"}""",
    ):
        return {
            "query": query,
            "selection": selection,
        }

    @property
    def _default_sensor_configs(self):
        """Configure default sensors (cameras)"""
        configs = []

        # Configure the four cameras: left_front, right_front, base_front, and wrist

        # Left front camera
        left_front_pose = sapien_utils.look_at(
            eye=[-0.4, 0.6, 0.6], target=[0.0, 0, 0.1]
        )
        configs.append(
            CameraConfig(
                "left_camera",
                left_front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        # Right front camera
        right_front_pose = sapien_utils.look_at(
            eye=[-0.4, -0.6, 0.6], target=[0.0, 0, 0.1]
        )
        configs.append(
            CameraConfig(
                "right_camera",
                right_front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        # front camera
        front_pose = sapien_utils.look_at(eye=[-0.55, 0, 0.1], target=[0.0, 0, 0.1])
        configs.append(
            CameraConfig(
                "front_camera",
                front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        # Base front camera
        base_front_pose = sapien_utils.look_at(eye=[0.6, 0, 0.6], target=[-0.1, 0, 0.1])
        configs.append(
            CameraConfig(
                "base_front_camera",
                base_front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        vlm_pose = sapien_utils.look_at(eye=[1, 0, 1], target=[-0.1, 0, 0.0])
        configs.append(
            CameraConfig(
                "vlm_camera",
                vlm_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        base_pose = sapien_utils.look_at(eye=[-0.5, 0, 1.8], target=[0.4, 0, 0.0])
        configs.append(
            CameraConfig(
                "base_camera",
                base_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )
        base_up_front_pose = sapien_utils.look_at(
            eye=[0.5, 0, 0.6], target=[0.4, 0, 0.0]
        )
        configs.append(
            CameraConfig(
                "base_up_front_camera",
                base_up_front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )
        # front camera
        front_pose = sapien_utils.look_at(eye=[-0.55, 0, 0.1], target=[0.0, 0, 0.1])
        configs.append(
            CameraConfig(
                "front_camera",
                front_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        human_pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        configs.append(
            CameraConfig(
                "human_camera",
                human_pose,
                self.camera_width,
                self.camera_height,
                self.external_camera_fov,
                0.01,
                100,
            )
        )

        # Note: Wrist camera is configured in the PandaWristCam class
        # If using standard Panda, we need to add a wrist camera manually
        if self.robot_uids == "panda":
            configs.append(
                CameraConfig(
                    "wrist_camera",
                    Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),  # Will be mounted on hand
                    self.camera_width,
                    self.camera_height,
                    self.wrist_camera_fov,
                    0.01,
                    100,
                    entity_uid="panda_hand",  # Mount on the hand link
                )
            )

        return configs

    @property
    def _default_human_render_camera_configs(self):
        """Configure camera for human rendering (visualization)"""
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        """Load the robot agent"""
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        """Load the basic scene with table"""
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

    def _load_asset(
        self,
        asset_path: str,
        scale=1.0,
        mass: float = 1.0,
        friction: float = 1.0,
        restitution: float = 0.1,
        density: float = None,
        name: str = None,
        body_type: str = "dynamic",
        convex=True,
    ):
        """
        Load an asset (GLB/USD) with configurable physical properties

        Args:
            asset_path: Path to the GLB/USD file
            scale: Scaling factor for the asset
            mass: Mass of the asset in kg
            friction: Friction coefficient
            restitution: Restitution coefficient (bounciness)
            density: Density of the asset (if provided, overrides mass)
            name: Name for the asset (defaults to filename)

        Returns:
            The created asset actor
        """
        asset_path = self.update_prefix(asset_path)
        if not os.path.exists(asset_path):
            self.log(f"Asset file not found: {asset_path}")
            return None
        if friction > 0.1:
            dyn_friction = friction - 0.1
        else:
            dyn_friction = friction
        # Get file extension and base name
        file_ext = os.path.splitext(asset_path)[1].lower()
        base_name = os.path.basename(asset_path)

        if name is None:
            name = os.path.splitext(base_name)[0]

        self.log(f"Loading asset: {name} from {asset_path}")

        # Create actor builder
        builder = self.scene.create_actor_builder()
        # Check for collision mesh with "_collision" or "_col" suffix
        collision_path = None
        base_path, ext = os.path.splitext(asset_path)
        potential_collision_paths = [
            f"{base_path}_collision{ext}",
            f"{base_path}_col{ext}",
            f"{base_path}.collision{ext}",
            f"{base_path}{ext}",
        ]

        for path in potential_collision_paths:
            if os.path.exists(path):
                collision_path = path
                self.log(f"Found collision mesh: {collision_path}")
                break
        # Add collision geometry
        # breakpoint()
        mtl = sapien.physx.PhysxMaterial(
            static_friction=friction,
            dynamic_friction=dyn_friction,
            restitution=0.01,
        )
        if not isinstance(scale, list):
            scale = [scale] * 3
        self.log(f"-------- input scale is {scale}")
        if collision_path:
            # Use the exact collision mesh from the separate file
            try:
                if not convex:
                    builder.add_nonconvex_collision_from_file(
                        filename=collision_path, scale=scale, material=mtl
                    )
                    self.log(f"Added exact collision geometry from {collision_path}")
                else:
                    builder.add_convex_collision_from_file(
                        filename=collision_path, scale=scale, material=mtl
                    )
                    self.log(f"Added convex collision geometry from {collision_path}")
            except Exception as e:
                self.log(f"Error adding exact collision: {e}")
                self._add_primitive_collision(builder, asset_path, file_ext, scale)
        else:
            # Add primitive collision based on asset type
            self.log("loading primitive collision as there is not proviede")
            self._add_primitive_collision(builder, asset_path, file_ext, scale)

        # Add visual from file
        try:
            success = builder.add_visual_from_file(filename=asset_path, scale=scale)
            if not success:
                self.log(f"Failed to load visual from {asset_path}")
                return None
        except Exception as e:
            self.log(f"Error loading visual: {e}")
            return None

        # Build the actor
        # if density is not None:
        #     asset = builder.build(name=name, density=density)
        # else:
        #     asset = builder.build(name=name)
        #
        asset = _build_by_type(builder, name, body_type)
        if asset is None:
            self.log(f"Failed to build asset")
            return None

        # Set physical properties
        try:
            # Set mass if density wasn't used
            if density is None and mass > 0:
                asset.set_mass(mass)
                self.log(f"the mass have been set to {asset.mass}")

            # Set damping for stability
            asset.set_damping(linear=0.1, angular=0.5)

            # Set friction and restitution
            if hasattr(asset, "get_collision_shapes"):
                for collision_shape in asset.get_collision_shapes():
                    collision_shape.set_friction(friction)
                    collision_shape.set_restitution(restitution)
            else:
                # Try alternative API
                for link in asset.get_links():
                    for cs in link.get_collision_shapes():
                        cs.set_friction(friction)
                        cs.set_restitution(restitution)

            # input()
        except Exception as e:
            self.log(f"Warning: Could not set some physical properties: {e}")

        # Add to scene assets list
        self.scene_assets.append(asset)

        self.log(f"Successfully created asset: {name}")
        return asset

    def update_prefix(self, path):
        if path is None:
            raise FileNotFoundError("your input path is none ")
        if not os.path.exists(path):
            return os.path.join(self.asset_path_prefix, path)
        return path

    # @pysnooper.snoop()
    def load_from_config(
        self,
        object_config,
        name,
        body_type="dynamic",
        convex=False,
        scale_override=None,
        friction_override=None,
        position_override=[0, 0, 0],
    ):
        import json

        config = {}
        object_scale = 1  # Scale of the object
        object_mass = 0.5  # Mass of the object in kg
        object_friction = 1.0  # Friction coefficient of the object
        # object_scale = 0.05
        object_path = None
        object_restitution = 0.1
        object_config = self.update_prefix(object_config)
        if object_config and os.path.exists(object_config):
            # Distance threshold for successful placement

            # Load configuration from JSON file
            try:
                with open(object_config, "r") as f:
                    config = json.load(f)
            except Exception as e:
                raise e
            self.log(f"Loaded object configuration from {object_config}")
            self.log(f"Configuration contents: {config}")

        object_path = config.get("usd-path", object_path)
        object_urdf_path = config.get("urdf_path", None)
        if object_urdf_path:
            return self.load_articulation_from_json(
                object_config,
                name=name,
                scale_override=scale_override,
                fix_root_link=body_type == "static",
            )
        object_scale = config.get("scale", object_scale)
        object_scale = scale_override if scale_override is not None else object_scale
        object_mass = config.get("mass", object_mass)
        object_friction = config.get("friction", object_friction)
        object_friction = (
            friction_override if friction_override is not None else object_friction
        )
        object_restitution = config.get("restitution", object_restitution)
        object_orientation = config.get("orientation", [0, 0, 0])
        object_position = config.get("position", [0, 0, 0])
        self.log(f"----loaded orientation is {object_orientation}")
        obj = self.load_asset(
            object_path,
            object_scale,
            object_mass,
            object_friction,
            restitution=object_restitution,
            name=name,
            body_type=body_type,
            convex=convex,
        )
        if position_override != [0, 0, 0]:
            object_position = position_override
        obj.set_pose(
            sapien.Pose(
                p=object_position, q=euler2quat(*np.deg2rad(object_orientation))
            )
        )

        return obj

    def load_asset(
        self,
        asset_path: str,
        scale=1.0,
        mass: float = 1.0,
        friction: float = 1.0,
        restitution: float = 0.1,
        density: float = None,
        name: str = None,
        body_type: str = "dynamic",
        convex=False,
    ):
        try:
            return self._load_asset(
                asset_path,
                scale,
                mass,
                friction,
                restitution,
                density,
                name,
                body_type=body_type,
                convex=convex,
            )
        except Exception as e:
            self.log(f"Error in _load_asset: {e}")

            return self._create_default_object(
                name=name, body_type=body_type, friction=friction
            )

    def _create_default_object(
        self,
        friction=None,
        name="cube",
        position_override=[0, 0, 0],
        body_type="dynamic",
        size=0.02,
    ):
        """Create a default object (red cube) if no object path is provided."""
        builder = self.scene.create_actor_builder()
        if not (isinstance(friction, float) or isinstance(friction, int)):
            friction = 0.5
        dyn_friction = max(0, friction - 0.1)
        mtl = sapien.physx.PhysxMaterial(
            static_friction=friction,
            dynamic_friction=dyn_friction,
            restitution=0.01,
        )
        # Add collision component
        builder.add_box_collision(half_size=[size] * 3, material=mtl)

        # Add visual component (red cube)
        try:
            # Try with material parameter (newer SAPIEN versions)
            material = sapien.render.RenderMaterial()
            material.set_base_color([1, 0, 0, 1])  # Red color with alpha=1
            builder.add_box_visual(half_size=[size] * 3, material=material)
        except TypeError:
            # Fallback for older SAPIEN versions
            try:
                # Try with color parameter
                builder.add_box_visual(half_size=[size] * 3, color=[1, 0, 0, 1])
            except TypeError:
                # Fallback with no color
                builder.add_box_visual(half_size=[size] * 3)

        # Create the actor
        actor = (
            builder.build(name=name)
            if body_type == "dynamic"
            else builder.build_kinematic(name=name)
        )
        actor.set_pose(sapien.Pose(p=position_override, q=[1, 0, 0, 0]))
        # Set physical properties
        try:
            actor.set_damping(linear=0.5, angular=0.5)
        except Exception as e:
            self.log(f"Warning: Could not set some physical properties: {e}")

        # Set friction
        # breakpoint()
        try:
            for collision_shape in actor.get_collision_shapes():
                friction = friction if friction is not None else self.object_friction
                collision_shape.set_friction(friction)
        except Exception as e:
            self.log(f"Warning: Could not set friction: {e}")

        return actor

    def _add_primitive_collision(self, builder, asset_path, file_ext, scale):
        """Add primitive collision shapes based on asset type"""
        file_path = asset_path.lower()
        assert isinstance(scale, list), "the input scale is not a list"
        self.log(f"---- in add primitve {scale}")
        half_size = [0.03 * x for x in scale]  # Default size

        # Detect shape from filename
        if any(x in file_path for x in ["bottle", "can", "cylinder"]):
            # For cylindrical objects
            cylinder_radius = half_size * 0.8
            cylinder_half_length = half_size * 2.5

            # Add capsule collision for better grasping
            builder.add_capsule_collision(
                radius=cylinder_radius,
                half_length=cylinder_half_length,
                pose=sapien.Pose([0, 0, cylinder_half_length]),
            )
            self.log(f"Added capsule collision for cylindrical object")

        elif any(x in file_path for x in ["box", "cube", "rect"]):
            # For box-like objects
            box_half_size = [half_size * 0.9, half_size * 0.9, half_size * 0.9]
            builder.add_box_collision(half_size=box_half_size)
            self.log(f"Added box collision for box-like object")

        elif any(x in file_path for x in ["ball", "sphere"]):
            # For spherical objects
            sphere_radius = half_size * 0.9
            builder.add_sphere_collision(radius=sphere_radius)
            self.log(f"Added sphere collision for spherical object")

        else:
            # Try to use convex hull collision if available
            try:
                builder.add_convex_collision_from_file(filename=asset_path, scale=scale)
                self.log("Added convex hull collision from file")
            except (AttributeError, Exception) as e:
                self.log(f"Could not add convex hull collision: {e}")
                # Fallback to box collision
                builder.add_box_collision(half_size=[half_size, half_size, half_size])
                self.log(f"Fallback to box collision")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize episode - to be implemented by subclasses"""
        # Initialize the table scene
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

        # Initialize the robot
        self.agent.reset(None)

    def _get_success(self, env_idx=None):
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        return success

    def evaluate(self, env_idx=None):
        """Evaluate task success - to be implemented by subclasses"""
        suc = self._get_success(env_idx=env_idx)
        if isinstance(suc, dict):
            suc = suc["success"]
        return {"success": suc, "description": self.description}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward - to be implemented by subclasses"""
        suc = self._get_success()
        if isinstance(suc, dict):
            suc = suc["success"]

        return torch.ones(self.num_envs, device=self.device) * suc

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """Compute normalized dense reward (0-1 range)"""
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def get_actor_size(self, actor):
        # 假设 container 是你的对象
        actor_meshes = actor.get_collision_meshes()
        if actor_meshes:
            min_bound = np.array([float("inf"), float("inf"), float("inf")])
            max_bound = np.array([float("-inf"), float("-inf"), float("-inf")])

            for mesh in actor_meshes:
                vertices = mesh.vertices
                if len(vertices) > 0:
                    min_bound = np.minimum(min_bound, vertices.min(axis=0))
                    max_bound = np.maximum(max_bound, vertices.max(axis=0))

            bbox_size = max_bound - min_bound
            return bbox_size  # 返回 [x_size, y_size, z_size]
        else:
            raise ValueError("没有可用的碰撞网格，无法计算大小")

    def set_orientation(self, actor, orientation):
        """
        orientation: rad
        """

        actor.set_pose(
            sapien.Pose(
                p=actor.pose.p.cpu().numpy()[0].tolist(), q=euler2quat(orientation)
            )
        )

    def set_pos(self, actor, pos):
        actor.set_pose(sapien.Pose(p=pos, q=actor.pose.q.cpu().numpy()[0].tolist()))

    def get_articulation_limits_info(self, actor):
        infos = {}
        for joint in actor.get_active_joints():
            limits = joint.get_limits()
            name = joint.name
            infos[name] = limits
            # joint_pos = joint.get_drive_target()
            # joint_vel = joint.get_drive_velocity()
            # joint_name = joint.get_name()
            # infos[joint_name] = joint_pos, joint_vel
        return infos

    def set_articulation_joint(
        self,
        actor,
        name,
        target,
        pd_version="easy",
        hard_pd=(50, 4),
        easy_pd=(5, 1),
        friction=0.5,
    ):
        pd = easy_pd if pd_version == "easy" else hard_pd
        for j, joint in enumerate(actor.get_active_joints()):
            if joint.name == name:
                minn, maxx = joint.get_limits()[0]

                target = target * (maxx - minn) + minn
                qpos = torch.zeros_like(actor.qpos)  # 获取当前 qpos 的形状
                qpos[:, j] = target  # 设置所有环境的门为打开状态
                actor.set_qpos(qpos)
                joint.set_drive_properties(stiffness=pd[0], damping=pd[1])
                joint.set_drive_target(target)
                joint.set_friction(friction)
                self.target_joint_idx = j
                break
        self.log(f"--- set joint {name} {target}")

    def get_articulation_joint_info(self, actor, name):
        for joint in actor.get_active_joints():
            if joint.name == name:
                minn, maxx = joint.get_limits()[0]
                now_pose = joint.qpos
                pos = (now_pose - minn) / (maxx - minn)
                # print(name, pos)
                return pos

    def get_joint(self, actor, name):
        for joint in actor.get_active_joints():
            if joint.name == name:
                return joint

    def _load_cabinet(self, cabinet_id):
        """Load a cabinet model from PartNet Mobility dataset"""
        # If a cabinet configuration file is provided, use it
        # if self.cabinet_config_path:
        #     # Load from JSON configuration with scale override
        #     return load_articulation_from_json(
        #         scene=self.scene,
        #         json_path=self.cabinet_config_path,
        #         scale_override=self.cabinet_scale
        #     )
        #
        # Otherwise, use the high-level articulation loader

        return load_articulation_from_json(
            scene=self.scene,
            json_path="configs/drawer_cabinet.json",
            scale_override=0.5,
            json_type="urdf",
            position_override=[-0.15, -0.6, 0.35],
            orientation_override=[0, 0, np.pi * -0.5],
            prefix_function=self.update_prefix,
        )
        table_height = 0.3
        offset = 0.05  # Small offset to prevent intersection with the table

        # Set the cabinet position
        position = np.array([-0.15, -0.45, table_height + offset])
        orientation = np.array([0, 0, np.pi * -0.5])  # No rotation
        cabinet_scale = 0.5
        # Load the articulation
        cabinet = load_articulation(
            scene=self.scene,
            position=position,
            orientation=orientation,
            scale=cabinet_scale,
            data_source="partnet-mobility",
            class_name="cabinet",
            class_id=cabinet_id,
            fix_root_link=True,
            name=f"cabinet-{cabinet_id}",
        )

        return cabinet

    def load_articulation_from_json(
        self,
        json_path,
        position_override: np.ndarray = [0, 0, 0],
        orientation_override: np.ndarray = [1, 0, 0, 0],
        scale_override: float = None,
        fix_root_link=None,
        name=None,
    ):
        json_path = self.update_prefix(json_path)
        return load_articulation_from_json(
            self.scene,
            json_path,
            json_type="urdf",
            prefix_function=self.update_prefix,
            position_override=position_override,
            scale_override=scale_override,
            orientation_override=orientation_override,
            name=name,
            fix_override=fix_root_link,
        )

    def _get_obs_extra(self, info):
        obs = {}
        if not self.called_obj:
            self.get_all_objects()

        for obj in self.actor_reg:
            obs[obj.name] = {"pos": obj.pose.p, "rot": obj.pose.q}
        for obj in self.articulation_reg:
            obs[obj.name] = {
                "pos": obj.pose.p,
                "rot": obj.pose.q,
                "qpos": obj.get_qpos(),
                "qvel": obj.get_qvel(),
            }

        tcp_pose = self.agent.tcp.pose.raw_pose
        obs["tcp_pose"] = tcp_pose
        return obs

    def update_obj_register(self, obj):
        if isinstance(obj, list):
            for i in obj:
                self.update_obj_register(i)
            return
        assert isinstance(obj, Articulation) or isinstance(obj, Actor)
        if isinstance(obj, Articulation):
            self.articulation_reg.append(obj)
        elif isinstance(obj, Actor):
            self.actor_reg.append(obj)

        return

    def get_all_objects(self):
        self.called_obj = True
        # 获取场景对象
        scene = self.scene

        # 获取所有静态物体 (Actors)
        actors = scene.get_all_actors()
        # print("All Actors:")
        for actor in actors:
            # print(self.get_actor_size(actor))
            self.log(f" - {actor.name} ")

        # 获取所有关节物体 (Articulations)
        articulations = scene.get_all_articulations()
        # breakpoint()
        for articulation in articulations:
            # print("All Articulations:")
            # print(self.get_actor_size(articulation))

            self.log(f" - {articulation.name} ")
            # actors += [x for x in articulation.get_links()]

        # 合并所有物体到一个列表
        all_objects = actors + articulations
        # print("\nAll Spawned Objects:")
        # for obj in all_objects:
        #     print(f" - {obj.name} ")
        self.actor_reg = actors
        self.articulation_reg = articulations
        self.all_objects = all_objects

    def get_aabb(self, obj):
        actor_meshes = obj.get_collision_meshes()

        if actor_meshes:
            min_bound = np.array([float("inf"), float("inf"), float("inf")])
            max_bound = np.array([float("-inf"), float("-inf"), float("-inf")])

            for mesh in actor_meshes:
                vertices = mesh.vertices
                if len(vertices) > 0:
                    min_bound = np.minimum(min_bound, vertices.min(axis=0))
                    max_bound = np.maximum(max_bound, vertices.max(axis=0))

            # bbox_size = max_bound - min_bound
        else:
            raise ValueError("没有可用的碰撞网格，无法计算大小")

        # min_bound += obj_pose.cpu().numpy()[0]
        # max_bound += obj_pose.cpu().numpy()[0]
        position = obj.pose.p.cpu().numpy()[0]
        mesh_center = (max_bound + min_bound) / 2

        min_bound = position + np.array(min_bound) - mesh_center
        max_bound = position + np.array(max_bound) - mesh_center
        return min_bound, max_bound

    # def make_bbox(self, obj):
    #     return BBoxObject(obj.pose.p[0], obj.pose.q[0], self.get_aabb(obj))
    #
    def show_two_objects(self, obj1, obj2=None):
        if not self.debug:
            return
        # print("---aabb is ", self.get_aabb(obj1), obj1.pose)
        # if not hasattr(self, "mark1"):
        # if not hasattr(self, "mark2"):
        self.debug_cnt += 1
        self.mark1 = self.mark_aabb_corners(obj1, iter=self.debug_cnt)
        if obj2 == None:
            return
        # print(self.get_aabb(obj2), obj2.pose)
        self.mark2 = self.mark_aabb_corners(obj2, iter=self.debug_cnt)
        # if not hasattr(self, "vis1"):
        #     self.vis1, self.del1 = self.create_visual_cube(obj1)
        # else:
        #     self.vis1.set_pose(
        #         sapien.Pose(
        #             p=obj1.pose.p.cpu().numpy()[0] + self.del1,
        #             # p=obj1.pose.p.cpu().numpy()[0],
        #             q=obj1.pose.q.cpu().numpy()[0],
        #         )
        #     )
        # if not hasattr(self, "vis2"):
        #     self.vis2, self.del2 = self.create_visual_cube(obj2)
        # else:
        #     self.vis2.set_pose(
        #         sapien.Pose(
        #             p=obj1.pose.p.cpu().numpy()[0] + self.del2,
        #             # p=obj1.pose.p.cpu().numpy()[0],
        #             q=obj1.pose.q.cpu().numpy()[0],
        #         )
        #     )

        # bbox1 = self.make_bbox(obj1)
        # bbox2 = self.make_bbox(obj2)
        # self.bbox_visualizer.visualize(bbox1, bbox2)
        #

    def calculate_obj_roi(self, obj1, obj2):
        """
        Calculate the ratios of the ROI volume to the volumes of obj1 and obj2.

        Parameters:
            obj1: First object
            obj2: Second object

        Returns:
            tuple: (ratio_a, ratio_b) where
                ratio_a: ROI volume / obj1 volume
                ratio_b: ROI volume / obj2 volume
                Each ratio is 0 if there’s no overlap or if the respective object volume is 0
        """
        # Get AABBs for both objects
        aabb1 = self.get_aabb(obj1)
        aabb2 = self.get_aabb(obj2)

        # Calculate ROI volume
        roi_volume = self.calculate_aabb_roi(aabb1, aabb2)
        # Calculate volume of obj1
        min1, max1 = aabb1[0], aabb1[1]
        obj1_volume = (max1[0] - min1[0]) * (max1[1] - min1[1]) * (max1[2] - min1[2])

        # Calculate volume of obj2
        min2, max2 = aabb2[0], aabb2[1]
        obj2_volume = (max2[0] - min2[0]) * (max2[1] - min2[1]) * (max2[2] - min2[2])

        # Calculate ratios, handling edge cases
        ratio_a = roi_volume / obj1_volume if obj1_volume > 0 and roi_volume > 0 else 0
        ratio_b = roi_volume / obj2_volume if obj2_volume > 0 and roi_volume > 0 else 0

        return (ratio_a, ratio_b)

    def calculate_obj_aabb_roi(self, obj, aabb):
        aabb1 = self.get_aabb(obj)
        aabb2 = aabb
        roi_volume = self.calculate_aabb_roi(aabb1, aabb2)

        # Calculate volume of obj1
        min1, max1 = aabb1[0], aabb1[1]
        obj1_volume = (max1[0] - min1[0]) * (max1[1] - min1[1]) * (max1[2] - min1[2])

        # Calculate volume of obj2
        min2, max2 = aabb2[0], aabb2[1]
        obj2_volume = (max2[0] - min2[0]) * (max2[1] - min2[1]) * (max2[2] - min2[2])

        # Calculate ratios, handling edge cases
        ratio_a = roi_volume / obj1_volume if obj1_volume > 0 and roi_volume > 0 else 0
        ratio_b = roi_volume / obj2_volume if obj2_volume > 0 and roi_volume > 0 else 0

        return (ratio_a, ratio_b)

    def calculate_aabb_roi(self, aabb1, aabb2):
        """
        Calculate the volume of the overlapping region between two AABBs.

        Parameters:
            aabb1: [[x1 y1 z1] [x2 y2 z2]] - First AABB with min and max coordinates
            aabb2: [[x1 y1 z1] [x2 y2 z2]] - Second AABB with min and max coordinates

        Returns:
            float: Volume of the overlapping region,
            or 0 if there is no overlap
        """
        # Extract min and max coordinates for both AABBs
        min1, max1 = aabb1[0], aabb1[1]
        min2, max2 = aabb2[0], aabb2[1]

        # Calculate the intersection coordinates
        x_min = max(min1[0], min2[0])
        y_min = max(min1[1], min2[1])
        z_min = max(min1[2], min2[2])

        x_max = min(max1[0], max2[0])
        y_max = min(max1[1], max2[1])
        z_max = min(max1[2], max2[2])

        # Calculate dimensions of overlap
        x_dim = x_max - x_min
        y_dim = y_max - y_min
        z_dim = z_max - z_min

        # If any dimension is negative or zero, there is no overlap
        if x_dim <= 0 or y_dim <= 0 or z_dim <= 0:
            return 0

        # Calculate and return volume
        volume = x_dim * y_dim * z_dim
        return volume

    def create_visual_cube(
        self,
        obj,
    ):
        min_bound, max_bound = self.get_aabb(obj)
        half_size = (max_bound - min_bound) / 2
        position = obj.pose.p.cpu().numpy()[0]
        name = obj.name + "_vis"
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=half_size,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=[0, 1, 0, 1],
            ),
        )
        # strongly recommended to set initial poses for objects, even if you plan to modify them later
        # print("delta pos is ", (max_bound + min_bound) / 2)
        delta_pose = (max_bound + min_bound) / 2 + np.array([0, 0, 0.1])
        begin_pose = sapien.Pose(
            p=position + delta_pose,
            q=obj.pose.q.cpu().numpy()[0].tolist(),
        )
        # print("the name is ", name)
        vis = builder.build(name=name)
        vis.set_pose(begin_pose)
        return vis, delta_pose

    def mark_aabb_corners(self, obj, marker_size=0.003, color=None, iter=0):
        """
        Mark the eight corners of an object's AABB with small sphere markers.

        Args:
            obj: The object to visualize the AABB corners for
            marker_size: Size of the marker spheres (radius)
            color: RGBA color values as a list [r, g, b, a], default is [1.0, 0.0, 0.0, 1.0] (red)

        Returns:
            List of marker actors created
        """
        if color is None:
            color = [0.0, 1.0, 0.0, 1.0]  # Default to red

        # Get the AABB for the object
        min_bound, max_bound = self.get_aabb(obj)
        # Calculate all 8 corners of the AABB
        corners = [
            # Bottom face
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            # Top face
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
        ]

        markers = []
        # Create a marker for each corner
        for i, corner in enumerate(corners):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=marker_size,
                material=sapien.render.RenderMaterial(base_color=color),
            )

            # Position the marker at the corner in world space
            # corner_pos = np.array(corner)
            corner_pos = np.array(corner)
            marker_pose = sapien.Pose(
                p=corner_pos,
                q=[1, 0, 0, 0],  # Identity quaternion (no rotation)
            )

            marker = builder.build_kinematic(name=f"{obj.name}_corner_{i}_{iter}")
            marker.set_pose(marker_pose)
            markers.append(marker)

        return markers

    def create_coordinate_frame(self, origin=None, axis_length=0.1, axis_radius=0.01):
        """
        Creates a coordinate frame visualization at the specified origin with RGB colors for the 3 axes.

        Args:
            origin: The position of the origin for the coordinate frame. If None, uses [0, 0, 0].
            axis_length: Length of each axis.
            axis_radius: Radius of each axis cylinder.

        Returns:
            List of actor objects representing the coordinate frame axes
        """
        if origin is None:
            origin = [0, 0, 0]

        # Create actors for each axis (X, Y, Z) with RGB colors
        axes = []

        # X-axis (Red)
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=sapien.render.RenderMaterial(
                base_color=[1, 0, 0, 1],  # Red
            ),
        )
        x_axis = builder.build_kinematic(name="x_axis")
        # Position the cylinder along the X-axis
        x_pose = sapien.Pose(
            p=[origin[0] + axis_length / 2, origin[1], origin[2]],
            q=euler2quat(0, 0, 0),  # Rotate to align with X-axis
        )
        x_axis.set_pose(x_pose)
        axes.append(x_axis)

        # Y-axis (Green)
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=sapien.render.RenderMaterial(
                base_color=[0, 1, 0, 1],  # Green
            ),
        )
        y_axis = builder.build_kinematic(name="y_axis")
        # Position the cylinder along the Y-axis
        y_pose = sapien.Pose(
            p=[origin[0], origin[1] + axis_length / 2, origin[2]],
            q=euler2quat(0, 0, np.pi / 2),  # Rotate to align with Y-axis
        )
        y_axis.set_pose(y_pose)
        axes.append(y_axis)

        # Z-axis (Blue)
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=axis_radius,
            half_length=axis_length / 2,
            material=sapien.render.RenderMaterial(
                base_color=[0, 0, 1, 1],  # Blue
            ),
        )
        z_axis = builder.build_kinematic(name="z_axis")
        # Position the cylinder along the Z-axis
        z_pose = sapien.Pose(
            p=[origin[0], origin[1], origin[2] + axis_length / 2],
            q=euler2quat(0, np.pi / 2, 0),  # No rotation needed for Z-axis
        )
        z_axis.set_pose(z_pose)
        axes.append(z_axis)

        # Add a small sphere at the origin
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=axis_radius * 1.5,
            material=sapien.render.RenderMaterial(
                base_color=[1, 1, 1, 1],  # White
            ),
        )
        origin_sphere = builder.build_kinematic(name="origin")
        origin_sphere.set_pose(sapien.Pose(p=origin))
        axes.append(origin_sphere)

        return axes

    def _create_target_marker(
        self, position=[0, 0, 0], name="target_marker", add_collision=False, size=0.01
    ):
        """Create a visual marker for the target placement location"""
        # Create actor builder
        builder = self.scene.create_actor_builder()

        # Add visual shape (green transparent cylinder)
        builder.add_cylinder_visual(
            radius=size,
            half_length=size,
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.5]),
        )
        if add_collision:
            builder.add_box_collision(sapien.Pose(), half_size=[0.03, 0.03, 0.02])
        # Build the actor (kinematic - no physics)
        target = builder.build_kinematic(name=name)
        target.set_pose(sapien.Pose(p=position, q=[1, 0, 0, 0]))

        return target

    def _create_goal_area(
        self,
        pose=sapien.Pose(p=[0.0, 0, 0], q=(euler2quat(*np.deg2rad([0, 90, 0])))),
        position=None,
        radius=0.05,
        thickness=0.001,
    ):
        if position:
            pose.p = position

        self.log(f"{self.marker_collision}")
        # breakpoint()
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=radius,
            thickness=thickness,
            name="goal_region",
            add_collision=self.marker_collision,
            body_type="kinematic",
            initial_pose=pose,
        )

        return self.goal_region

    def calculate_object_distance(self, obj1, obj2, axis=[0, 1, 2]):
        pos1 = obj1.pose.p
        pos2 = obj2.pose.p
        delta = pos1 - pos2
        delta = delta[0][axis]
        return np.linalg.norm(delta)

    def is_static(self, obj, threshold=0.05):
        if isinstance(obj, Actor):
            vel = obj.get_linear_velocity()
            gsp = self.agent.is_grasping(obj)
            return np.linalg.norm(vel) <= threshold and not gsp
        else:
            vel = obj.get_root_linear_velocity()
            q_vel = obj.get_qvel()
            a = np.linalg.norm(vel) <= threshold
            b = np.linalg.norm(q_vel) <= threshold
            return a and b

    def get_obj_angle(self, obj):
        obj_quat = obj.pose.q.cpu().numpy()[0]
        obj_euler_rad = quat2euler(obj_quat)
        obj_euler_deg = np.rad2deg(obj_euler_rad)
        return obj_euler_deg

    def compare_obj_angle(self, obj1, obj2, threshold=5):
        ag1 = self.get_obj_angle(obj1)
        ag2 = self.get_obj_angle(obj2)
        return np.linalg.norm(ag1 - ag2) <= threshold

    def compare_angle(
        self, obj, target_orientation, specific_axis=None, threshold=5, debug=False
    ):
        """
        Compare the orientation of an object with a target orientation.

        Args:
            obj: Actor object whose orientation to check
            target_orientation: Target orientation in degrees [roll, pitch, yaw]
            specific_axis: Optional, specify which axis to compare ('roll', 'pitch', 'yaw' or index 0,1,2)
                           If None, compares all axes
            threshold: Threshold in degrees for considering angles equal (default: 0.01)

        Returns:
            bool: True if orientations match within threshold, False otherwise
        """
        # Get object's quaternion and convert to Euler angles in degrees
        obj_quat = obj.pose.q.cpu().numpy()[0]
        obj_euler_rad = quat2euler(obj_quat)
        obj_euler_deg = np.rad2deg(obj_euler_rad)

        # Convert target orientation to numpy array if it's not already
        target_orientation = np.array(target_orientation)

        # If specific axis is specified, only compare that axis
        if specific_axis is not None:
            # Convert string axis names to indices
            if specific_axis == "roll" or specific_axis == 0:
                axis_idx = 0
            elif specific_axis == "pitch" or specific_axis == 1:
                axis_idx = 1
            elif specific_axis == "yaw" or specific_axis == 2:
                axis_idx = 2
            else:
                raise ValueError(
                    f"Invalid axis: {specific_axis}. Must be 'roll', 'pitch', 'yaw' or 0, 1, 2"
                )

            # Compare only the specified axis
            angle_diff = abs(obj_euler_deg[axis_idx] - target_orientation[axis_idx])
            # Handle angle wrapping (e.g., 359° vs 1°)
            angle_diff = min(angle_diff, 360 - angle_diff)
            return angle_diff <= threshold

        # Compare all axes
        angle_diffs = np.abs(obj_euler_deg - target_orientation)
        # Handle angle wrapping for each axis
        angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)
        if debug:
            self.log(f"{angle_diffs}")
        return np.all(angle_diffs <= threshold)

    # def is_agent_empty(self):
    #
    def get_object(self, object_name):
        for obj in self.scene.get_all_actor():
            # print(obj.name)
            if object_name in obj.entity.name:
                return obj
        # print([obj.name for obj in self.all_objects])
        return None
        # self.scene.get_actor(object_name)

    def get_all_object_name(self):
        name_list = []
        name_id_dict = {}
        for obj in self.all_objects:
            if isinstance(obj, sapien.pysapien.Entity):
                name_list.append(obj.name)
                name_id_dict[obj.name] = obj.per_scene_id
            else:
                # breakpoint()
                for link in obj.get_links():
                    # breakpoint()
                    name = link.name
                    if "scene-0_" in link.name:
                        name = link.name.replace(
                            link.name.split("scene-0_")[1] + "_", "", 1
                        )
                    elif "scene-0-" in link.name:
                        name = link.name.replace(
                            link.name.split("scene-0-")[1] + "_", "", 1
                        )

                    name_list.append(name)
                    name_id_dict[name] = link.entity.per_scene_id
        # self.log(f"name_list is {name_list}")
        self.name_id_dict = name_id_dict
        self.name_list = name_list
        # breakpoint()
        return name_list

    def get_all_name_id_dict(self):
        self.get_all_object_name()
        return self.name_id_dict

    def is_grasping(self, object_name):
        obj = self.object_dict[object_name]
        return self.agent.is_grasping(obj)

    def is_stable(self, obj, threshold=10):
        if not hasattr(self, "stable_dict"):
            self.stable_dict = {}
            self.log("--- rebuild stable_dict ---")
        if obj not in self.stable_dict.keys():
            self.stable_dict[obj] = 0
        if not self.is_static(obj):
            self.stable_dict[obj] = 0
        else:
            self.stable_dict[obj] += 1
        return self.stable_dict[obj] > threshold

    def log(self, msg):
        self.logger.info(msg)

    def data_for_VQA(self):
        """
        return:
            query_image: base64 image
                there are bounding box with annotation on the image
            query_instruction:
                {
                    "query": "which one part contain the target object?  "
                    "selection": {"A": The object is in A " , "B": "The object is in B " , "C": "There is no target object in the cabinet"}
                }
        """

        return {
            "query_image": self.build_image(self.query_image_path_task),
            "query_instruction": self.build_instruction(
                self.query_query if hasattr(self, 'query_query') else "", 
                str(self.query_selection) if hasattr(self, 'query_selection') else ""
            ),  #
        }

    def get_all_ids(self):
        breakpoint()
