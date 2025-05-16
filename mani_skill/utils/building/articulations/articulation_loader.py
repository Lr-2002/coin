"""
Articulation loader utilities for ManiSkill environments.
This module provides high-level functions for loading articulations from various sources.
"""

import json
import os
import numpy as np
import sapien
from transforms3d.euler import euler2quat
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import sapien_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.building.articulations import get_articulation_builder


def orientation_check(orientation):
    # Handle orientation format
    if orientation is None:
        # Default to no rotation
        quat = euler2quat(0, 0, 0)
    elif len(orientation) == 3:
        # Euler angles [roll, pitch, yaw]
        quat = euler2quat(orientation[0], orientation[1], orientation[2])
    elif len(orientation) == 4:
        # Quaternion [w, x, y, z]
        quat = orientation
    else:
        raise ValueError(f"Invalid orientation format: {orientation}")
    return quat


def load_articulation(
    scene: ManiSkillScene,
    position: np.ndarray,
    orientation: np.ndarray = None,
    scale: float = 1.0,
    data_source: str = "partnet-mobility",
    class_name: str = None,
    class_id: str = None,
    fix_root_link: bool = True,
    urdf_config: dict = None,
    name: str = None,
) -> Any:
    """
    High-level function to load an articulation with specified parameters.

    Args:
        scene: ManiSkill scene to load the articulation into
        position: 3D position [x, y, z] for the articulation
        orientation: Orientation as quaternion [w, x, y, z] or Euler angles [roll, pitch, yaw]
        scale: Scale factor for the articulation
        data_source: Source of the articulation data (default: "partnet-mobility")
        class_name: Class name of the articulation (e.g., "cabinet", "faucet")
        class_id: ID of the articulation within the data source
        fix_root_link: Whether to fix the root link of the articulation
        urdf_config: Additional URDF configuration
        name: Name to assign to the articulation

    Returns:
        The loaded articulation object
    """
    if urdf_config is None:
        urdf_config = dict()

    # Process URDF configuration to ensure it's in the correct format
    processed_urdf_config = sapien_utils.parse_urdf_config(urdf_config)

    quat = orientation_check(orientation)
    # Construct the full ID based on data source and class ID
    if data_source == "partnet-mobility":
        if class_id is None:
            raise ValueError(
                "class_id must be provided for partnet-mobility data source"
            )
        full_id = f"{data_source}:{class_id}"
    else:
        raise ValueError(f"Unsupported data source: {data_source}")

    # Get the articulation builder
    builder = get_articulation_builder(
        scene=scene,
        id=full_id,
        fix_root_link=fix_root_link,
        urdf_config=processed_urdf_config,
        scale=scale,
    )

    # Set the initial pose
    builder.initial_pose = sapien.Pose(p=position, q=quat)

    # Build the articulation
    articulation_name = (
        name
        if name
        else f"{class_name}-{class_id}"
        if class_name
        else f"articulation-{class_id}"
    )
    articulation = builder.build(name=articulation_name)

    return articulation


def load_articulation_from_json(
    scene: ManiSkillScene,
    json_path: str,
    position_override: np.ndarray = [0, 0, 0],
    orientation_override: np.ndarray = [1, 0, 0, 0],
    scale_override: float = None,
    json_type="partnet",
    prefix_function=None,
    fix_override=None,
    name=None,
) -> Any:
    if json_type == "partnet":
        return load_articulation_from_partnet_json(
            scene, json_path, position_override, orientation_override, scale_override
        )
    elif json_type == "urdf":
        return load_articulation_from_urdf_json(
            scene,
            json_path,
            position_override,
            orientation_override,
            scale_override,
            prefix_function,
            name=name,
            fix_override=fix_override,
        )


def load_articulation_from_partnet_json(
    scene: ManiSkillScene,
    json_path: str,
    position_override: np.ndarray = None,
    orientation_override: np.ndarray = None,
    scale_override: float = None,
) -> Any:
    """
    Load an articulation from a JSON configuration file.

    Args:
        scene: ManiSkill scene to load the articulation into
        json_path: Path to the JSON configuration file
        position_override: Optional override for the position
        orientation_override: Optional override for the orientation
        scale_override: Optional override for the scale

    Returns:
        The loaded articulation object

    JSON format example:
    {
        "data_source": "partnet-mobility",
        "class_name": "cabinet",
        "class_id": "1234",
        "position": [0.3, 0.0, 0.35],
        "orientation": [0, 0, 0],  # Euler angles [roll, pitch, yaw]
        "scale": 0.5,
        "fix_root_link": true,
        "name": "my_cabinet",
        "urdf_config": {
            "material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            }
        }
    }
    """
    # Load the JSON configuration
    config = load_json(json_path)

    # Extract parameters from the JSON
    data_source = config.get("data_source", "partnet-mobility")
    class_name = config.get("class_name")
    class_id = config.get("class_id")
    position = np.array(config.get("position", [0, 0, 0]))
    orientation = np.array(config.get("orientation", [0, 0, 0]))
    scale = config.get("scale", 1.0)
    fix_root_link = config.get("fix_root_link", True)
    urdf_config = config.get("urdf_config", {})
    name = config.get("name")

    # Apply overrides if provided
    if position_override is not None:
        position = position_override
    if orientation_override is not None:
        orientation = orientation_override
    if scale_override is not None:
        scale = scale_override

    # Load the articulation
    articulation = load_articulation(
        scene=scene,
        position=position,
        orientation=orientation,
        scale=scale,
        data_source=data_source,
        class_name=class_name,
        class_id=class_id,
        fix_root_link=fix_root_link,
        urdf_config=urdf_config,
        name=name,
    )

    return articulation


def load_articulations_from_directory(
    scene: ManiSkillScene,
    directory_path: str,
    filter_pattern: str = "*.json",
    scale_override: float = None,
) -> Dict[str, Any]:
    """
    Load multiple articulations from JSON configuration files in a directory.

    Args:
        scene: ManiSkill scene to load the articulations into
        directory_path: Path to the directory containing JSON configuration files
        filter_pattern: Pattern to filter JSON files (default: "*.json")
        scale_override: Optional override for the scale of all articulations

    Returns:
        Dictionary mapping articulation names to articulation objects
    """
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(
            f"Directory {directory_path} does not exist or is not a directory"
        )

    # Find all JSON files in the directory
    json_files = list(directory.glob(filter_pattern))
    if not json_files:
        print(
            f"Warning: No JSON files found in {directory_path} with pattern {filter_pattern}"
        )
        return {}

    # Load articulations from each JSON file
    articulations = {}
    for json_file in json_files:
        try:
            articulation = load_articulation_from_json(
                scene=scene,
                json_path=str(json_file),
                scale_override=scale_override,
            )

            # Use the filename (without extension) as the key if no name is specified in the JSON
            name = (
                articulation.name if hasattr(articulation, "name") else json_file.stem
            )
            articulations[name] = articulation
            print(f"Loaded articulation {name} from {json_file}")
        except Exception as e:
            print(f"Error loading articulation from {json_file}: {e}")

    return articulations


def load_articulation_from_urdf(
    scene,
    urdf_path,
    position=[0, 0, 0],
    orientation=None,
    scale=None,
    fix_root_link=True,
    urdf_config=None,
    name=None,
) -> Any:
    """
    Load an articulation directly from a URDF file.

    Args:
        scene: ManiSkill scene to load the articulation into
        urdf_path: Path to the URDF file
        position: 3D position [x, y, z] for the articulation, defaults to [0.7, 0, 0.5]
        orientation: Orientation as quaternion [w, x, y, z] or Euler angles [roll, pitch, yaw]
        scale: Scale factor for the articulation
        fix_root_link: Whether to fix the root link of the articulation
        urdf_config: Additional URDF configuration
        name: Name to assign to the articulation

    Returns:
        The loaded articulation object
    """
    # Set default values
    quat = orientation_check(orientation)
    # Process URDF configuration
    if urdf_config is None:
        urdf_config = dict()
    processed_urdf_config = sapien_utils.parse_urdf_config(urdf_config)

    # Create the loader
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link

    # Set scale if provided
    if scale is not None:
        loader.scale = scale

    # Apply URDF configuration
    sapien_utils.apply_urdf_config(loader, processed_urdf_config)
    loader.load_multiple_collisions_from_file = True
    # Parse the URDF file
    articulation_builders = loader.parse(str(urdf_path))["articulation_builders"]
    if not articulation_builders:
        raise ValueError(f"No articulations found in URDF file: {urdf_path}")

    builder = articulation_builders[0]
    builder.initial_pose = sapien.Pose(p=position, q=quat)

    # Build the articulation
    articulation_name = name if name else f"articulation_{Path(urdf_path).stem}"
    articulation = builder.build(name=articulation_name)

    return articulation


def load_articulation_from_urdf_json(
    scene: ManiSkillScene,
    json_path: str,
    position_override: np.ndarray = None,
    orientation_override: np.ndarray = None,
    scale_override: float = None,
    prefix_function=None,
    fix_override=None,
    name=None,
) -> Any:
    """
    Load an articulation from a JSON configuration file that specifies a URDF file.

    Args:
        scene: ManiSkill scene to load the articulation into
        json_path: Path to the JSON configuration file
        position_override: Optional override for the position
        orientation_override: Optional override for the orientation
        scale_override: Optional override for the scale

    Returns:
        The loaded articulation object

    JSON format example:
    {
        "urdf_path": "/path/to/model.urdf",
        "position": [0.0, 0.0, 0.0],
        "orientation": [0, 0, 0],
        "scale": 1.0,
        "fix_root_link": true,
        "urdf_config": {
            "material": {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0
            }
        },
        "name": "my_articulation"
    }
    """
    # Load the JSON configuration
    json_path = prefix_function(json_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    # Extract parameters from the JSON
    urdf_path = config.get("urdf_path")
    urdf_path = prefix_function(urdf_path)
    if not urdf_path:
        raise ValueError("urdf_path must be specified in the JSON configuration")

    position = config.get("position")
    orientation = config.get("orientation")
    scale = config.get("scale")
    fix_root_link = config.get("fix_root_link", True)
    urdf_config = config.get("urdf_config", {})
    name = config.get("name") if not name else name

    # Apply overrides if provided
    if position_override is not None:
        position = position_override
    if orientation_override is not None:
        orientation = orientation_override
    if scale_override is not None:
        scale = scale_override
    if fix_override is not None:
        fix_root_link = fix_override
    # Load the articulation
    articulation = load_articulation_from_urdf(
        scene=scene,
        urdf_path=urdf_path,
        position=position,
        orientation=orientation,
        scale=scale,
        fix_root_link=fix_root_link,
        urdf_config=urdf_config,
        name=name,
    )

    return articulation
