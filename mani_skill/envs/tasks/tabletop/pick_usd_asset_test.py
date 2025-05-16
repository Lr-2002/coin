from typing import Any, Dict, Union, Optional
import os

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickUSDAssetTest-v1", max_episode_steps=50)
class PickUSDAssetTestEnv(BaseEnv):
    """
    **Task Description:**
    A test task where the objective is to grasp a USD asset from the table.

    **Randomizations:**
    - the asset's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]
    - the asset's z-axis rotation is randomized to a random angle

    **Success Conditions:**
    - the asset is grasped and lifted at least 0.05m above the table
    - the robot is static (q velocity < 0.2)
    """

    _sample_video_link = None
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    lift_height_thresh = 0.05  # Minimum height to lift the asset above the table

    def __init__(
        self, 
        *args, 
        robot_uids="panda", 
        robot_init_qpos_noise=0.02, 
        usd_path=None,
        asset_scale=1.0,
        asset_init_pos=None,
        asset_init_ori=None,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.usd_path = usd_path
        self.asset_scale = asset_scale
        self.asset_size = 0.02  # Default size estimate, will be updated after loading
        self.asset_init_pos = asset_init_pos
        self.asset_init_ori = asset_init_ori
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Load USD asset if provided, otherwise create a fallback cube
        if self.usd_path and os.path.exists(self.usd_path):
            self.asset = self._load_usd_asset(self.usd_path)
        else:
            # Fallback to a simple cube if no USD file is provided
            self.asset_size = 0.02
            self.asset = self._create_fallback_asset()

    def _load_usd_asset(self, usd_path):
        """Load a USD asset into the scene"""
        print(f"Loading asset from {usd_path}")
        
        # Get file extension
        file_ext = os.path.splitext(usd_path)[1].lower()
        
        # Create actor builder
        builder = self.scene.create_actor_builder()
        
        # Try to load the asset
        success = False
        half_size = 0.02  # Default size
        
        try:
            # For GLB files, try to estimate size from filename or use specific handling
            if file_ext == '.glb':
                print(f"Loading GLB file: {usd_path}")
                
                # Try to estimate size from filename
                file_name = os.path.basename(usd_path).lower()
                
                # Check for common objects and set appropriate sizes
                if 'bottle' in file_name or '7_up' in file_name:
                    # Typical bottle size
                    half_size = 0.03  # Radius
                    print(f"Detected bottle-like object, using half_size={half_size}")
                elif 'can' in file_name:
                    # Typical can size
                    half_size = 0.025  # Radius
                    print(f"Detected can-like object, using half_size={half_size}")
                elif 'box' in file_name or 'cube' in file_name:
                    # Typical box size
                    half_size = 0.04  # Half-width
                    print(f"Detected box-like object, using half_size={half_size}")
                else:
                    # Default size for unknown objects
                    half_size = 0.035
                    print(f"Using default size for GLB object: half_size={half_size}")
                
                # Try to load the GLB file directly
                try:
                    success = builder.add_visual_from_file(
                        filename=usd_path,
                        scale=[self.asset_scale] * 3
                    )
                    print(f"Added GLB visual directly: {success}")
                except Exception as e:
                    print(f"Error adding GLB visual directly: {e}")
                    success = False
            else:
                # For USD and other formats
                try:
                    # Add visual from file
                    success = builder.add_visual_from_file(usd_path)
                    print(f"Added visual from file: {success}")
                except Exception as e:
                    print(f"Error adding visual from file: {e}")
                    success = False
            
            if not success:
                print(f"Failed to load visual from file")
                return self._create_fallback_asset()
                
            # We can't get the size directly from the builder in this SAPIEN version
            # Just use a reasonable default size for now
            print(f"Using size: {half_size * 2}")
            
        except Exception as e:
            print(f"Error loading asset: {e}")
            return self._create_fallback_asset()
        
        # Check if there's a collision mesh with "_collision" or "_col" suffix
        collision_path = None
        base_path, ext = os.path.splitext(usd_path)
        potential_collision_paths = [
            f"{base_path}_collision{ext}",
            f"{base_path}_col{ext}",
            f"{base_path}.collision{ext}"
        ]
        
        print(f"Looking for collision meshes:")
        for path in potential_collision_paths:
            print(f"  Checking: {path}")
            if os.path.exists(path):
                collision_path = path
                print(f"  Found collision mesh: {collision_path}")
                break
        
        if not collision_path:
            print(f"  No collision mesh found, using fallback")
        
        # Add collision geometry
        if collision_path:
            # Use the exact collision mesh from the separate file
            try:
                builder.add_nonconvex_collision_from_file(
                    filename=collision_path,
                    scale=[self.asset_scale] * 3
                )
                print(f"Added exact collision geometry from {collision_path}")
            except Exception as e:
                print(f"Error adding exact collision: {e}")
                # Fallback to primitive collision shapes
                self._add_primitive_collision(builder, file_ext, half_size)
        else:
            # Fallback to primitive collision shapes
            self._add_primitive_collision(builder, file_ext, half_size)
        
        # Build the actor
        asset = builder.build(name="loaded_asset")
        
        if asset is None:
            print(f"Failed to build asset")
            return self._create_fallback_asset()
        
        # Set the asset size
        self.asset_size = half_size
        
        # Apply scale if needed
        if self.asset_scale != 1.0:
            try:
                # Scale the visual meshes
                for visual in asset.get_visual_bodies():
                    visual.set_scale([self.asset_scale, self.asset_scale, self.asset_scale])
            except Exception as e:
                print(f"Error scaling asset: {e}")
            
            # Update asset size
            self.asset_size *= self.asset_scale
        
        # Set physical properties for better interaction
        try:
            # Different SAPIEN versions have different API
            if hasattr(asset, 'get_collision_shapes'):
                for collision_shape in asset.get_collision_shapes():
                    # Make the object graspable by setting appropriate physical properties
                    collision_shape.set_friction(1.0)  # Higher friction helps with grasping
            else:
                # Try alternative API
                for link in asset.get_links():
                    for cs in link.get_collision_shapes():
                        cs.set_friction(1.0)
        except Exception as e:
            print(f"Warning: Could not set collision properties: {e}")
        
        # Make the object dynamic so it can be picked up
        try:
            asset.set_damping(linear=0.5, angular=0.5)  # Damping helps stabilize the object
        except Exception as e:
            print(f"Warning: Could not set damping: {e}")
        
        print(f"Successfully created asset")
        return asset

    def _add_primitive_collision(self, builder, file_ext, half_size):
        """Add primitive collision shapes based on file type"""
        print(f"Adding primitive collision shape for file type: {file_ext}")
        print(f"Using half_size: {half_size}")
        
        # For GLB files, try to add collision geometry that matches the visual better
        if file_ext == '.glb':
            try:
                # Try to detect the model type from the file path
                file_path = self.usd_path.lower()
                
                # Check if it's a bottle or cylindrical object
                if 'bottle' in file_path or '7_up' in file_path or 'cylinder' in file_path or 'can' in file_path:
                    # For bottle-like objects, a cylinder or capsule collision shape works better
                    cylinder_radius = half_size * 0.8
                    cylinder_half_length = half_size * 2.5
                    
                    # Add capsule collision for better grasping (smoother than cylinder)
                    builder.add_capsule_collision(
                        radius=cylinder_radius,
                        half_length=cylinder_half_length,
                        pose=sapien.Pose([0, 0, cylinder_half_length])
                    )
                    print(f"Added capsule collision for bottle-like object: radius={cylinder_radius}, half_length={cylinder_half_length}")
                
                # Check if it's a box-like object
                elif 'box' in file_path or 'cube' in file_path or 'rect' in file_path:
                    # For box-like objects, use a box collision
                    box_half_size = [half_size * 0.9, half_size * 0.9, half_size * 0.9]
                    builder.add_box_collision(half_size=box_half_size)
                    print(f"Added box collision for box-like object: half_size={box_half_size}")
                
                # Check if it's a spherical object
                elif 'ball' in file_path or 'sphere' in file_path:
                    # For spherical objects, use a sphere collision
                    sphere_radius = half_size * 0.9
                    builder.add_sphere_collision(radius=sphere_radius)
                    print(f"Added sphere collision for spherical object: radius={sphere_radius}")
                
                # Default case - try to use a compound collision shape
                else:
                    # Create a compound collision shape for better interaction
                    # Main body - slightly smaller box
                    builder.add_box_collision(
                        half_size=[half_size * 0.8, half_size * 0.8, half_size * 0.8]
                    )
                    
                    # Try to add a convex hull collision if available in this SAPIEN version
                    try:
                        # Some SAPIEN versions support this
                        builder.add_convex_collision_from_file(
                            filename=self.usd_path,
                            scale=[self.asset_scale] * 3
                        )
                        print("Added convex hull collision from file")
                    except (AttributeError, Exception) as e:
                        print(f"Could not add convex hull collision: {e}")
                        
                    print(f"Added compound collision shape for general GLB object")
            except Exception as e:
                print(f"Error adding specialized collision: {e}")
                # Fallback to box collision
                builder.add_box_collision(half_size=[half_size, half_size, half_size])
                print(f"Fallback to box collision with half_size={half_size}")
        else:
            # Default box collision
            builder.add_box_collision(half_size=[half_size, half_size, half_size])
            print(f"Added default box collision with half_size={half_size}")

    def _create_fallback_asset(self):
        """Create a fallback asset when USD loading fails"""
        print("Creating fallback asset")
        self.asset_size = 0.02
        
        # Create actor builder
        builder = self.scene.create_actor_builder()
        
        # Add collision shape
        builder.add_box_collision(half_size=[self.asset_size, self.asset_size, self.asset_size])
        
        # Add visual shape
        try:
            builder.add_box_visual(half_size=[self.asset_size, self.asset_size, self.asset_size], color=[1, 0, 0, 1])
        except TypeError:
            # For older versions of SAPIEN that don't support color parameter
            builder.add_box_visual(half_size=[self.asset_size, self.asset_size, self.asset_size])
        
        # Build the actor
        asset = builder.build(name="fallback_asset")
        
        # Set initial pose
        asset.set_pose(sapien.Pose(p=[0, 0, self.asset_size]))
        
        return asset

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # We need to initialize the table scene first
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
        
        # Place the asset on the table
        table_height = self.table_scene.table_height
        
        # Use custom position if provided, otherwise randomize
        if self.asset_init_pos is not None:
            asset_xy = torch.tensor([self.asset_init_pos[0], self.asset_init_pos[1]]).expand(self.num_envs, 2)
            asset_z = self.asset_init_pos[2] if len(self.asset_init_pos) > 2 else table_height + self.asset_size * 2
        else:
            # Randomize position on table
            xy_range = 0.1
            asset_xy = torch.rand(self.num_envs, 2) * xy_range * 2 - xy_range
            asset_z = table_height + self.asset_size * 2  # Place above table
        
        # Use custom orientation if provided, otherwise randomize
        if self.asset_init_ori is not None:
            asset_ori = torch.tensor(self.asset_init_ori).expand(self.num_envs, 3)
        else:
            # Randomize orientation (only around z-axis)
            asset_ori = torch.zeros(self.num_envs, 3)
            asset_ori[:, 2] = torch.rand(self.num_envs) * 2 * np.pi
        
        # Set the asset pose
        asset_pose = sapien.Pose(
            p=[asset_xy[env_idx, 0].item(), asset_xy[env_idx, 1].item(), asset_z],
            q=euler2quat(
                asset_ori[env_idx, 0].item(),
                asset_ori[env_idx, 1].item(),
                asset_ori[env_idx, 2].item(),
            ),
        )
        self.asset.set_pose(asset_pose)
        
        # Debug collision information
        print(f"\nAsset collision information:")
        print(f"  Asset position: {asset_pose.p}")
        print(f"  Asset size: {self.asset_size}")
        try:
            if hasattr(self.asset, 'has_collision_shapes'):
                if callable(self.asset.has_collision_shapes):
                    print(f"  Has collision shapes: {self.asset.has_collision_shapes()}")
                else:
                    print(f"  Has collision shapes: {self.asset.has_collision_shapes}")
            else:
                print(f"  No has_collision_shapes attribute")
        except Exception as e:
            print(f"  Error checking collision shapes: {e}")
            
        try:
            # Try to get collision information using different API versions
            collision_info = []
            
            # Try different methods to get collision info
            if hasattr(self.asset, 'get_collision_shapes'):
                shapes = self.asset.get_collision_shapes()
                collision_info.append(f"  Direct shapes: {len(shapes)}")
            
            if hasattr(self.asset, 'get_links'):
                links = self.asset.get_links()
                total_shapes = 0
                for link in links:
                    if hasattr(link, 'get_collision_shapes'):
                        shapes = link.get_collision_shapes()
                        total_shapes += len(shapes)
                collision_info.append(f"  Link shapes: {total_shapes}")
                
            if hasattr(self.asset, 'get_visual_bodies'):
                bodies = self.asset.get_visual_bodies()
                collision_info.append(f"  Visual bodies: {len(bodies)}")
                
            if collision_info:
                for info in collision_info:
                    print(info)
            else:
                print("  No collision information available through API")
                
        except Exception as e:
            print(f"  Error getting detailed collision info: {e}")
        
        # Initialize the robot - pass None instead of env_idx to avoid DOF mismatch error
        self.agent.reset(None)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.asset.pose.raw_pose,
                tcp_to_obj_pos=self.asset.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.asset)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if the asset is lifted above the table by the required threshold
        is_lifted = self.asset.pose.p[:, 2] > (self.asset_size + self.lift_height_thresh)
        
        return {
            "success": is_grasped & is_lifted & is_robot_static,
            "is_lifted": is_lifted,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.asset.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped
        
        # Reward for lifting the asset
        lift_height = self.asset.pose.p[:, 2] - self.asset_size
        lift_reward = torch.clamp(lift_height / self.lift_height_thresh, 0, 1)
        reward += lift_reward * is_grasped

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_lifted"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
