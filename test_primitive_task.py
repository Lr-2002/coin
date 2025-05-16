import gymnasium as gym
import numpy as np
import cv2
import time
from mani_skill.utils.display_multi_camera import display_camera_views
import argparse
import logging

logging.getLogger("mani_skill").setLevel(logging.ERROR)


def main(env_id):
    # Create the environment
    print(f"Creating environment: {env_id}")

    env = gym.make(
        env_id,
        obs_mode="rgbd",  # Use rgbd to get camera observations
        control_mode="pd_ee_delta_pose",
        robot_uids="panda_wristcam",
        render_mode="human",
    )
    # breakpoint()
    # Reset the environment
    print("Resetting environment...")
    obs, info = env.reset(seed=0)
    env.unwrapped.get_all_object_name()
    # Display camera views
    print("Displaying camera views...")
    display_camera_views(obs)
    cv2.waitKey(1)

    # Run a few steps with random actions
    # print("Running a few steps with random actions...")
    for i in range(10000):
        action = env.action_space.sample() * 0
        obs, reward, terminated, truncated, info = env.step(action)
        # print(obs.keys())
        # print(obs["extra"])
        # print(obs["agent"])
        # # print(info)
        # Display camera views
        display_camera_views(obs)
        cv2.waitKey(1)

        # Render the environment
        env.render_human()

        # Sleep to control the loop rate
        time.sleep(0.02)
        # time.sleep(5)

        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            break
            # obs, info = env.reset()


    # Close the environment
    env.close()
    print("Environment closed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run random steps in a specified environment."
    )
    parser.add_argument(
        "-e",
        "--env_id",
        type=str,
        default="Tabletop-PickPlace-Apple-v1",
        help="The environment ID to use.",
    )
    args = parser.parse_args()

    main(args.env_id)
