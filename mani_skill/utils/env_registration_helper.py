"""
Environment Registration Helper

This module provides utilities to help register new environments with consistent naming
and metadata according to the ManiSkill naming convention.
"""

import inspect
import os
from typing import Any, Dict, List, Optional, Type, Union

from mani_skill.utils.env_naming import (
    SceneType, ActionType, ObjectType, TaskClass,
    EnvNameGenerator, EnvNameParser
)
from mani_skill.utils.registration import register_env


def register_env_with_metadata(
    env_class: Type,
    scene_type: Union[SceneType, str],
    action_type: Union[ActionType, str],
    object_type: Union[ObjectType, str],
    task_class: Union[TaskClass, str],
    modifier: Optional[str] = None,
    version: int = 1,
    max_episode_steps: int = 200,
    description: Optional[str] = None,
    **kwargs
) -> str:
    """
    Register an environment with consistent naming and metadata.
    
    Args:
        env_class: The environment class to register
        scene_type: Scene type (e.g., SceneType.TABLETOP)
        action_type: Action type (e.g., ActionType.OPEN)
        object_type: Object type (e.g., ObjectType.DOOR)
        task_class: Task class (e.g., TaskClass.PRIMITIVE)
        modifier: Optional modifier (e.g., "WithSwitch")
        version: Version number (default: 1)
        max_episode_steps: Maximum number of steps per episode
        description: Optional description of the environment
        **kwargs: Additional keyword arguments for register_env
        
    Returns:
        The registered environment name
    """
    # Generate the environment name
    env_name = EnvNameGenerator.generate_env_name(
        scene_type, action_type, object_type, modifier, version
    )
    
    # Extract docstring from the environment class if no description is provided
    if description is None and env_class.__doc__:
        description = inspect.getdoc(env_class)
    
    # Create metadata
    metadata = {
        "scene_type": scene_type.value if isinstance(scene_type, SceneType) else scene_type,
        "action_type": action_type.value if isinstance(action_type, ActionType) else action_type,
        "object_type": object_type.value if isinstance(object_type, ObjectType) else object_type,
        "task_class": task_class.value if isinstance(task_class, TaskClass) else task_class,
        "description": description
    }
    
    # Register the environment
    register_env(
        env_name,
        env_class=env_class,
        max_episode_steps=max_episode_steps,
        metadata=metadata,
        **kwargs
    )
    
    return env_name


def create_env_template(
    env_name: str,
    output_dir: str,
    base_class: str = "UniversalTabletopEnv",
    overwrite: bool = False
) -> str:
    """
    Create a template file for a new environment based on its name.
    
    Args:
        env_name: The environment name (e.g., "Tabletop-Open-Door-v1")
        output_dir: Directory to save the template file
        base_class: Base class for the environment (default: "UniversalTabletopEnv")
        overwrite: Whether to overwrite existing files
        
    Returns:
        Path to the created template file
    """
    # Parse the environment name
    parsed = EnvNameParser.parse_env_name(env_name)
    scene = parsed.get("scene", "")
    action = parsed.get("action", "")
    obj = parsed.get("object", "")
    modifier = parsed.get("modifier", "")
    version = parsed.get("version", "1")
    
    # Determine the task class
    task_class = EnvNameParser.get_task_class(env_name)
    
    # Create class name
    class_name = f"{action}{obj}Env"
    if modifier:
        # Convert modifiers like "WithSwitch" to "WithSwitchEnv"
        class_name = f"{action}{obj}{modifier}Env"
    
    # Create file name
    file_name = f"{action.lower()}_{obj.lower()}"
    if modifier:
        file_name += f"_{modifier.lower()}"
    file_name += ".py"
    
    # Determine the directory based on task class
    if task_class == TaskClass.PRIMITIVE:
        task_dir = "primitive_actions"
    elif task_class == TaskClass.INTERACTIVE:
        task_dir = "interactive_reasoning"
    else:
        task_dir = "complex_tasks"
    
    # Create full output path
    output_path = os.path.join(output_dir, task_dir, file_name)
    
    # Check if file already exists
    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create template content with escaped curly braces
    template = f'''"""
{env_name} Environment

This environment implements a {scene.lower()} task where the robot needs to {action.lower()} a {obj.lower()}.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.tasks.coin_bench.universal_tabletop_env import {base_class}
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import articulations
from mani_skill.utils.building.articulations.articulation_loader import load_articulation, load_articulation_from_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import table
from mani_skill.utils.structs import Articulation, Link, Pose
import os


@register_env("{env_name}", max_episode_steps=200)
class {class_name}({base_class}):
    """
    **Task Description:**
    A {scene.lower()} environment where the robot needs to {action.lower()} a {obj.lower()}.
    
    **Randomizations:**
    - The {obj.lower()}'s position on the table
    
    **Success Conditions:**
    - The {obj.lower()} has been successfully {action.lower()}ed
    """
    
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Panda
    
    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # Add any task-specific parameters here
        
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            **kwargs
        )
        
    def _load_scene(self, options=None):
        """Load the task-specific objects into the scene"""
        # Call the parent method to set up the base scene (table, etc.)
        super()._load_scene(options)
        
        # TODO: Load task-specific objects
        # Example:
        # self.object = self._load_object()

    def _load_object(self):
        """Load the main object for this task"""
        # TODO: Implement object loading
        # Example:
        # return load_articulation_from_json(
        #     scene=self.scene,
        #     json_path="configs/object.json",
        #     scale_override=0.3,
        #     position_override=[0.3, 0.0, 0.4]
        # )
        pass
     
    def _initialize_episode(self, env_idx, options=None):
        """Initialize the episode by setting up objects and robot"""
        # Call the parent method to initialize the base scene
        super()._initialize_episode(env_idx, options)
        
        # TODO: Initialize task-specific objects
        # Example:
        # self.object.set_pose(Pose(p=[0.3, 0.0, self.table_height + 0.05]))
    
    def compute_dense_reward(self, obs, action, info):
        """Compute the reward for the task"""
        # Initialize rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # TODO: Implement task-specific rewards
        # Example:
        # distance = torch.norm(self.object.pose.p - self.target_position)
        # rewards = 1.0 - torch.clamp(distance / 0.5, 0, 1)
        
        return rewards
    
    def evaluate(self, env_idx=None):
        """Evaluate if the task is successful"""
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        
        success = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
        
        # TODO: Implement task-specific success criteria
        # Example:
        # distance = torch.norm(self.object.pose.p - self.target_position)
        # success = distance < 0.05
        
        return {{
            "success": success,
        }}
'''
    
    # Write template to file
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"Created environment template at: {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    # Example of registering an environment with metadata
    from mani_skill.envs.tasks.coin_bench.primitive_actions.open_door import OpenDoorEnv
    
    env_name = register_env_with_metadata(
        env_class=OpenDoorEnv,
        scene_type=SceneType.TABLETOP,
        action_type=ActionType.OPEN,
        object_type=ObjectType.DOOR,
        task_class=TaskClass.PRIMITIVE,
        version=1,
        max_episode_steps=200,
        description="A tabletop environment with a door that needs to be opened."
    )
    print(f"Registered environment: {env_name}")
    
    # Example of creating a template for a new environment
    template_path = create_env_template(
        env_name="Tabletop-Pick-Bottle-v1",
        output_dir="mani_skill/envs/tasks/coin_bench",
        overwrite=True
    )
    print(f"Created template at: {template_path}")
