from .assembling_kits import AssemblingKitsEnv
from .lift_peg_upright import LiftPegUprightEnv
from .peg_insertion_side import PegInsertionSideEnv
from .pick_clutter_ycb import PickClutterYCBEnv
from .pick_cube import PickCubeEnv
from .pick_cube_test import PickCubeTestEnv
from .pick_single_ycb import PickSingleYCBEnv
from .pick_usd_asset_test import PickUSDAssetTestEnv
from .plug_charger import PlugChargerEnv
from .pull_cube import PullCubeEnv
from .push_cube import PushCubeEnv
from .stack_cube import StackCubeEnv
from .turn_faucet import TurnFaucetEnv
from .two_robot_pick_cube import TwoRobotPickCube
from .two_robot_stack_cube import TwoRobotStackCube
from .poke_cube import PokeCubeEnv
from .place_sphere import PlaceSphereEnv
from .roll_ball import RollBallEnv
from .push_t import PushTEnv
from .pull_cube_tool import PullCubeToolEnv
# from .pick_place_task import PickPlaceTaskEnv

# Register environments
import gymnasium as gym
from gymnasium.envs.registration import register

# Note: The environment is already registered in pick_place_task.py using the @register_env decorator
# But we'll keep this here for clarity
# register(
#     id="PickPlaceTask-v1",
#     entry_point="mani_skill.envs.tasks.tabletop.pick_place_task:PickPlaceTaskEnv",
#     max_episode_steps=50,
# )

