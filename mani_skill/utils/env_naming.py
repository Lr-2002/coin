"""
Environment Naming Utility for ManiSkill Benchmark

This module provides a standardized naming system for environments in the ManiSkill benchmark.
It helps maintain consistency in naming conventions and provides utilities for parsing and
generating environment names.

The naming convention follows the pattern:
    [Scene]-[Action]-[Object]-[Modifier]-v[Version]

Examples:
    - Tabletop-Open-Door-v1
    - Tabletop-Pick-Objects-InBox-v1
    - Tabletop-Move-Cube-WithPivot-v1
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


class SceneType(Enum):
    """Types of scenes available in the benchmark."""
    TABLETOP = "Tabletop"
    KITCHEN = "Kitchen"
    OFFICE = "Office"
    WORKSHOP = "Workshop"


class ActionType(Enum):
    """Types of actions that can be performed in the environments."""
    # Primitive actions
    PICK = "Pick"
    PLACE = "Place"
    PUT = "Put"
    OPEN = "Open"
    CLOSE = "Close"
    MOVE = "Move"
    LIFT = "Lift"
    ROTATE = "Rotate"
    PUSH = "Push"
    PULL = "Pull"
    INSERT = "Insert"
    STACK = "Stack"
    
    # Interactive actions
    FIND = "Find"
    SEEK = "Seek"


class ObjectType(Enum):
    """Types of objects that can be manipulated in the environments."""
    # Single objects
    DOOR = "Door"
    CABINET = "Cabinet"
    MICROWAVE = "Microwave"
    APPLE = "Apple"
    CUBE = "Cube"
    CUBES = "Cubes"
    BALL = "Ball"
    BALLS = "Balls"
    PEN = "Pen"
    BOOK = "Book"
    CYLINDER = "Cylinder"
    FORK = "Fork"
    TRIGGER = "Trigger"
    SEAL = "Seal"
    OBJECT = "Object"
    OBJECTS = "Objects"
    
    # Multiple objects (for composite tasks)
    LONG_OBJECTS = "LongObjects"


class TaskClass(Enum):
    """Classification of tasks based on complexity."""
    PRIMITIVE = "Primitive"
    INTERACTIVE = "Interactive"
    REASONING = "Reasoning"
    SEQUENTIAL = "Sequential"


class EnvNameParser:
    """Parser for environment names following the ManiSkill naming convention."""
    
    @staticmethod
    def parse_env_name(env_name: str) -> Dict[str, str]:
        """
        Parse an environment name into its components.
        
        Args:
            env_name: The environment name to parse (e.g., "Tabletop-Open-Door-v1")
            
        Returns:
            Dictionary containing the parsed components:
            {
                "scene": Scene type (e.g., "Tabletop"),
                "action": Action type (e.g., "Open"),
                "object": Object type (e.g., "Door"),
                "modifier": Optional modifier (e.g., "WithSwitch"),
                "version": Version number (e.g., "1")
            }
        """
        parts = env_name.split("-")
        result = {
            "scene": parts[0],
            "version": parts[-1]
        }
        
        # Extract version number
        if parts[-1].startswith("v"):
            result["version"] = parts[-1][1:]
            parts = parts[:-1]
        
        # Extract action and object
        if len(parts) >= 3:
            result["action"] = parts[1]
            result["object"] = parts[2]
            
            # Extract modifiers (if any)
            if len(parts) >= 4:
                result["modifier"] = "-".join(parts[3:])
            else:
                result["modifier"] = ""  # Use empty string instead of None
        
        return result
    
    @staticmethod
    def get_task_class(env_name: str) -> TaskClass:
        """
        Determine the task class (Primitive, Interactive, etc.) based on the environment name.
        
        Args:
            env_name: The environment name to classify
            
        Returns:
            TaskClass enum value
        """
        # This is a simplified heuristic - in a real implementation, you might
        # want to use a more sophisticated approach or a lookup table
        components = EnvNameParser.parse_env_name(env_name)
        
        # Check for common interactive patterns
        if any(modifier in env_name for modifier in ["WithSwitch", "WithPivot", "WithObstacle", 
                                                    "InBox", "IntoContainer", "FromCabinet",
                                                    "ToHolder", "OnPlate", "IntoMicrowave"]):
            return TaskClass.INTERACTIVE
        
        # Check for reasoning tasks
        if "Find" in env_name or "Seek" in env_name:
            return TaskClass.INTERACTIVE
        
        # Check for sequential tasks
        if "Twice" in env_name or components.get("action") == "Stack":
            return TaskClass.INTERACTIVE
        
        # Default to primitive
        return TaskClass.PRIMITIVE


class EnvNameGenerator:
    """Generator for environment names following the ManiSkill naming convention."""
    
    @staticmethod
    def generate_env_name(
        scene: Union[SceneType, str],
        action: Union[ActionType, str],
        obj: Union[ObjectType, str],
        modifier: Optional[str] = "",
        version: int = 1
    ) -> str:
        """
        Generate an environment name following the ManiSkill naming convention.
        
        Args:
            scene: Scene type (e.g., SceneType.TABLETOP or "Tabletop")
            action: Action type (e.g., ActionType.OPEN or "Open")
            obj: Object type (e.g., ObjectType.DOOR or "Door")
            modifier: Optional modifier (e.g., "WithSwitch")
            version: Version number (default: 1)
            
        Returns:
            Generated environment name (e.g., "Tabletop-Open-Door-WithSwitch-v1")
        """
        # Convert enum values to strings if needed
        scene_str = scene.value if isinstance(scene, SceneType) else scene
        action_str = action.value if isinstance(action, ActionType) else action
        obj_str = obj.value if isinstance(obj, ObjectType) else obj
        
        # Build the name
        name_parts = [scene_str, action_str, obj_str]
        
        if modifier:
            name_parts.append(modifier)
        
        name_parts.append(f"v{version}")
        
        return "-".join(name_parts)


def get_all_env_names() -> List[str]:
    """
    Get a list of all registered environment names in the benchmark.
    
    Returns:
        List of environment names
    """
    from mani_skill.utils.registration import REGISTERED_ENVS
    return list(REGISTERED_ENVS.keys())


def get_env_by_category(
    scene_type: Optional[Union[SceneType, str]] = None,
    action_type: Optional[Union[ActionType, str]] = None,
    object_type: Optional[Union[ObjectType, str]] = None,
    task_class: Optional[Union[TaskClass, str]] = None
) -> List[str]:
    """
    Get environments matching specified criteria.
    
    Args:
        scene_type: Filter by scene type
        action_type: Filter by action type
        object_type: Filter by object type
        task_class: Filter by task class
        
    Returns:
        List of environment names matching the criteria
    """
    all_envs = get_all_env_names()
    filtered_envs = []
    
    # Convert enum values to strings if needed
    scene_str = scene_type.value if isinstance(scene_type, SceneType) else scene_type
    action_str = action_type.value if isinstance(action_type, ActionType) else action_type
    object_str = object_type.value if isinstance(object_type, ObjectType) else object_type
    task_class_str = task_class.value if isinstance(task_class, TaskClass) else task_class
    
    for env_name in all_envs:
        components = EnvNameParser.parse_env_name(env_name)
        env_task_class = EnvNameParser.get_task_class(env_name)
        
        # Apply filters
        if scene_str and components.get("scene") != scene_str:
            continue
        if action_str and components.get("action") != action_str:
            continue
        if object_str and components.get("object") != object_str:
            continue
        if task_class_str and env_task_class.value != task_class_str:
            continue
        
        filtered_envs.append(env_name)
    
    return filtered_envs


# Example usage
if __name__ == "__main__":
    # Parse an environment name
    env_name = "Tabletop-Open-Door-v1"
    parsed = EnvNameParser.parse_env_name(env_name)
    print(f"Parsed {env_name}:", parsed)
    
    # Generate an environment name
    generated = EnvNameGenerator.generate_env_name(
        SceneType.TABLETOP,
        ActionType.PICK,
        ObjectType.APPLE,
        version=2
    )
    print(f"Generated: {generated}")
    
    # Get task class
    task_class = EnvNameParser.get_task_class("Tabletop-Pick-Objects-InBox-v1")
    print(f"Task class: {task_class}")
