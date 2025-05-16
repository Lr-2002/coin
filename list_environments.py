import os
import time
import pdb
import gymnasium as gym
from gymnasium import envs
import mani_skill
import cv2
from mani_skill.utils.wrappers import VLARecorderWrapper
import pickle as pkl
from typing import Dict
from gymnasium.envs.registration import EnvSpec
from os import error, path
import numpy as np
import json
from env_tests.utils.image_utils import extract_camera_image, save_debug_image

REGISTERED_ENVS: Dict[str, EnvSpec] = {}

INTERACTIVE_TASK_IMAGE_PATH = (
    "mani_skill/envs/tasks/coin_bench/interactive_reasoning/interactive_task_image"
)
os.makedirs(INTERACTIVE_TASK_IMAGE_PATH, exist_ok=True)


def main(test_vla_record=True, extract_images=False):
    # Get all registered environments
    all_envs = [env_spec.id for env_spec in envs.registry.values()]
    # Filter for ManiSkill environments
    mani_skill_envs = [
        env_id
        for env_id in all_envs
        if any(prefix in env_id for prefix in ["Tabletop", "Coin", "ManiSkill"])
    ]
    print("mani_skill_envs lens is ", len(mani_skill_envs))
    print("Available ManiSkill Environments:")
    ms_dict = {}
    primitive_dict = {}
    interactive_envs = {}
    with open("./primitive_instruction_objects.pkl", "rb") as f:
        primitive_envs = pkl.load(f).keys()
        print(f"primitive_envs: {primitive_envs}")
        interactive_envs = list(set(mani_skill_envs) - set(primitive_envs))
        print(f"interactive_envs: {interactive_envs}")
        print(f"primitive_envs lens is {len(primitive_envs)}")
        print(f"interactive_envs lens is {len(interactive_envs)}")
        input("Press Enter to continue...")

    error_list = []
    link_check = []
    workflow_dict = {}

    for name, env_set in zip(
        ["primitive", "interactive"], [primitive_envs, interactive_envs]
    ):
        env_set_object_dict = {}
        for env_id in sorted(env_set):
            env = gym.make(
                env_id,
                obs_mode="rgb+depth+segmentation",  # Use rgbd to get camera observations
                control_mode="pd_ee_delta_pose",
                robot_uids="panda_wristcam",
                render_mode="human",
            )

            if name == "interactive":
                workflow_dict[env_id] = env.workflow
            if test_vla_record:
                env = VLARecorderWrapper(
                    env,
                    output_dir="./debug/voxposer_evaluation_0422/",
                    model_class="Voxposer",
                    model_path="None",
                    save_trajectory=False,
                )

            # env = gym.make(env_id)
            obs, info = env.reset()
            object_list = env.unwrapped.get_all_object_name()
            seg_id_list = env.unwrapped.segmentation_id_map
            object_name_list = [x.name for x in seg_id_list.values()]
            
            # Get extended tags if available
            env_extended_tags = None
            if name == "interactive":
                try:
                    # Now extend the tags
                    if hasattr(env.unwrapped, 'extend_tags'):
                        extended_tags = env.unwrapped.extend_tags()
                        if extended_tags:
                            # Initialize if needed
                            if env_id not in ms_dict:
                                ms_dict[env_id] = {}
                            env_extended_tags = extended_tags
                            print(f"Added extended tags for {env_id}: {extended_tags}")
                except Exception as e:
                    print(f"Error handling tags for {env_id}: {e}")
            # breakpoint()
            instruction = env.unwrapped.description
            # breakpoint()
            object_list = [
                x
                for x in object_name_list
                if ("panda" not in x)
                and ("table" not in x)
                and ("ground" not in x)
                and ("camera" not in x)
                and ("_base" not in x)
            ] + ["gripper"]
            # query
            query_query = env.unwrapped.query_query
            query_selection = env.unwrapped.query_selection
            query_answer = env.unwrapped.query_answer
            # [
            env_set_object_dict[env_id] = {"ins": instruction, "objects": object_list}
            ms_dict[env_id] = {
                "ins": instruction,
                "query": {"query": query_query, "selection": query_selection},
                'tags': env_extended_tags
            }
            if max(["link" in x for x in object_list]):
                link_check.append(env_id)
            print(f"  - {env_id}")

            right_camera_tasks = [
                "Tabletop-Seek-Objects-WithObstacle-v1",
                "Tabletop-Lift-Book-v1",
            ]
            front_camera_tasks = ["Tabletop-Find-Book-From-Shelf-v1"]
            human_camera_tasks = [
                "Tabletop-Open-Cabinet-WithSwitch-v1",
                "Tabletop-Pick-Object-FromCabinet-v1",
                "Tabletop-Slide-Cube-WithPath-v1",
                "Tabletop-Slide-Cube-Into-Container-v1",
            ]
            base_camera_tasks = [
                "Tabletop-Pick-Cube-WithStick-v1",
                "Tabletop-Merge-Ball-WithBox-v1",
            ]

            if extract_images:
                for i in range(10):
                    # action = env.action_space.sample() * 0
                    action = np.array([-0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
                    obs, reward, terminated, truncated, info = env.step(action)

                    if i == 9:
                        if env_id in right_camera_tasks:
                            camera = "right_camera"
                        elif env_id in front_camera_tasks:
                            camera = "front_camera"
                        elif env_id in human_camera_tasks:
                            camera = "human_camera"
                        elif env_id in base_camera_tasks:
                            camera = "base_camera"
                        else:
                            camera = "left_camera"
                        # image = extract_camera_image(obs, camera)
                        # save_debug_image(image, os.path.join(INTERACTIVE_TASK_IMAGE_PATH, f"{env_id}.png"))
            env.reset()
            env.close()
        print("---------- link check ", link_check)
        # breakpoint()
        if name == "interactive":
            with open("./env_ins_objects.pkl", "wb") as f:
                pkl.dump(ms_dict, f)
            with open("./env_ins_objects.json", "w") as f:
                json.dump(ms_dict, f, indent=2)

            # Save workflow dictionary with skills for each environment
            # with open("./env_workflows.pkl", "wb") as f:
            #     pkl.dump(workflow_dict, f)
            with open("./env_workflows.json", "w") as f:
                json.dump(workflow_dict, f, indent=2)
                
            # Extended tags are already collected in the main loop and stored in ms_dict
            # Extract extended tags from ms_dict for a separate file
            extended_tags_dict = {}
            for env_id, env_data in ms_dict.items():
                if 'tags' in env_data and env_data['tags']:
                    extended_tags_dict[env_id] = env_data['tags']
            
            with open("./env_extended_tags.json", "w") as f:
                json.dump(extended_tags_dict, f, indent=2)

            print(ms_dict)
            print(f"env_num:{len(ms_dict)}")
            print(error_list)
        with open(f"{name}_instruction_objects.pkl", "wb") as f:
            pkl.dump(env_set_object_dict, f)


if __name__ == "__main__":
    main(test_vla_record=False)
