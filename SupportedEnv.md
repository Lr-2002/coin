# Coin Supported Environments

This document provides a comprehensive list of all supported environments in the ManiSkill framework, organized by category and task type.

## Scanner 
```bash
conda activate sapien && cd /home/lr-2002/project/reasoning_manipulation/ManiSkill && python env_scanner.py


```


## Tabletop Manipulation Environments

| Environment ID | Description | Key Features |
|----------------|-------------|-------------|
| `Tabletop-Open-Cabinet-v1` | Open cabinet drawers and doors on a tabletop | - PartNet Mobility cabinets<br>- Multiple drawer configurations<br>- Randomized cabinet models<br>- Cabinet position randomization |
| `Tabletop-Open-Cabinet-With-Switch-v1` | Control cabinet using a switch mechanism | - Switch state detection<br>- Dynamic cabinet response<br>- Causal relationship learning<br>- Switch state affects cabinet drawers |
| `Tabletop-PickPlace-Apple-v1` | Pick up an apple and place it at a target location | - Object position randomization<br>- Target position randomization<br>- Placement success detection<br>- Configurable object properties |

1. **Configuration Files**: JSON-based configuration for object properties, rewards, and task parameters
2. **Asset Libraries**: Support for custom 3D models, including URDF, USD, and OBJ formats
3. **Reward Shaping**: Customizable reward functions for specific training objectives
4. **Observation Spaces**: Configurable observation modes including state, RGB, depth, and combined modalities
5. **Difficulty Levels**: Progressive difficulty settings for curriculum learning

## Adding New Environments

To add a new environment to Coin:

1. Create a new environment class inheriting from `UniversalTabletopEnv`
2. Register the environment using the `@register_env` decorator
3. Implement required methods: `_initialize_episode`, `compute_dense_reward`, `evaluate`, etc.
4. Add configuration files and assets as needed
5. Create test scripts to validate environment functionality

For detailed implementation examples, refer to the existing environment implementations in the `mani_skill/envs/tasks/coin_bench` directory.

## Coin Bench Environments (Auto-Generated)

| Environment ID | Class Name | Source File | Description |
|----------------|------------|-------------|-------------|
| `Tabletop-PickPlace-Apple-DynamicMass-v1` | PickPlaceDynamicMassEnv | pick_place_dynamic_mass.py | No description available |
| `Tabletop-Open-Cabinet-v1` | CabinetOnTableEnv | cabinet_on_table.py | **Task Description:** |
| `Tabletop-Open-Cabinet-With-Switch-v1` | CabinetSwitchOnTableEnv | cabinet_with_switch_on_table.py | **Task Description:** |
| `Tabletop-Lift-BiggerObject-v1` | PickPlaceDynamicFrictionEnv | lift_bigger_object.py | No description available |
| `Tabletop-PickPlace-Apple-v1` | PickPlaceEnv | pick_place.py | **Task Description:** |
| `Tabletop-PickPlace-Apple-DynamicFriction-v1` | PickPlaceDynamicFrictionEnv | pick_place_dynamic_friction.py | No description available |

## Coin Bench Environments (Auto-Generated)

| Environment ID | Class Name | Source File | Description |
|----------------|------------|-------------|-------------|
| `Tabletop-Stack-LongObjects-v1` | StackLongObjectsEnv | stack_long_objects.py | **Task Description:** |
| `Tabletop-PickPlace-BallIntoContainer-v1` | PickPlaceBallIntoContainerEnv | pick_place_ball_into_container.py | **Task Description:** |
| `Tabletop-PickPlace-Apple-DynamicMass-v1` | PickPlaceDynamicMassEnv | pick_place_dynamic_mass.py | No description available |
| `Tabletop-Open-Cabinet-v1` | CabinetOnTableEnv | cabinet_on_table.py | **Task Description:** |
| `Tabletop-Open-Cabinet-With-Switch-v1` | CabinetSwitchOnTableEnv | cabinet_with_switch_on_table.py | **Task Description:** |
| `Tabletop-Lift-BiggerObject-v1` | PickPlaceDynamicFrictionEnv | lift_bigger_object.py | No description available |
| `Tabletop-PickPlace-Apple-v1` | PickPlaceEnv | pick_place.py | **Task Description:** |
| `Tabletop-PickPlace-Apple-DynamicFriction-v1` | PickPlaceDynamicFrictionEnv | pick_place_dynamic_friction.py | No description available |
