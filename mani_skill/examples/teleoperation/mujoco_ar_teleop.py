import argparse
from hmac import new
import gymnasium as gym
import numpy as np
import sapien.core as sapien
import time
import cv2
import os
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation
from mani_skill.utils.display_multi_camera import display_camera_views
from mani_skill.utils.plotting import RewardPlotter
import datetime

# Import MujocoARConnector for AR teleoperation
from mujoco_ar import MujocoARConnector


# Simple moving average filter for smoothing input
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []

    def apply_online(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="Tabletop-Open-Cabinet-v1",
        help="environment id",
    )
    parser.add_argument(
        "-o", "--obs-mode", type=str, default="rgbd", help="observation mode"
    )
    parser.add_argument(
        "-r",
        "--robot-uid",
        type=str,
        default="panda_wristcam",
        help="robot uid, must be panda_wristcam for this script",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="datasets",
        help="directory to record demonstrations",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save video"
    )
    return parser.parse_args()


class ARInputProcessor:
    def __init__(self, connector):
        self.connector = connector
        self.filter = MovingAverageFilter(window_size=5)
        self.last_pose = None
        self.initialized = False
        self.position_scale = 20  # Scale factor for position
        self.rotation_scale = 15  # Scale factor for rotation

    def _msg2pose(self, data, base=None):
        """Convert AR data to pose and calculate relative pose if base is provided"""
        if data["rotation"] is None:
            return None, None, False, False

        # Convert rotation matrix to rotation vector
        rotation = Rotation.from_matrix(data["rotation"]).as_rotvec()
        position = np.array(data["position"]).flatten()

        # Combine position and rotation into a single pose vector
        pose = np.concatenate([position, rotation])

        # Apply smoothing filter
        pose = self.filter.apply_online(pose)

        # Extract button states
        button = data["button"]
        toggle = data["toggle"]

        # Calculate relative pose if base is provided
        if base is not None:
            rel_pose = pose - base
        else:
            rel_pose = pose

        return rel_pose, pose, button, toggle

    def process_input(self):
        """Process the latest AR data and return action information"""
        # Get the latest data from the AR connector
        data = self.connector.get_latest_data()
        # Initialize if not already done
        if not self.initialized and data["rotation"] is not None:
            _, self.last_pose, _, _ = self._msg2pose(data)
            self.initialized = True
            return np.zeros(6), False, False  # Return zero delta on first frame

        # If not initialized or no data, return zeros
        if not self.initialized or data["rotation"] is None:
            return np.zeros(6), False, False

        # Process the AR data
        delta, current_pose, button_pressed, toggle_state = self._msg2pose(
            data, self.last_pose
        )

        # Update the last pose for next iteration
        self.last_pose = current_pose

        # Apply scaling factors but only keep the maximum component for position and rotation
        pos_delta = delta[:3]
        rot_delta = delta[3:]

        # Find the index of the maximum absolute value for position and rotation
        max_pos_idx = np.argmax(np.abs(pos_delta))
        max_rot_idx = np.argmax(np.abs(rot_delta))

        # Create new delta arrays with zeros except for the max component
        new_pos_delta = np.zeros(3)
        new_rot_delta = np.zeros(3)

        # Set only the maximum component with scaling
        new_pos_delta[max_pos_idx] = pos_delta[max_pos_idx] * self.position_scale
        new_rot_delta[max_rot_idx] = rot_delta[max_rot_idx] * self.rotation_scale
        new_rot_delta[1] = -new_rot_delta[1]
        # Combine position and rotation deltas
        delta = np.concatenate([new_pos_delta, new_rot_delta])
        print("moving direction: ", delta)
        time.sleep(0.02)
        return delta, button_pressed, toggle_state


def main():
    args = parse_args()

    # Ensure we're using the panda_wristcam robot
    if args.robot_uid != "panda_wristcam":
        print(
            "Warning: This script is designed for panda_wristcam. Setting robot_uid to panda_wristcam."
        )
        args.robot_uid = "panda_wristcam"

    # Initialize the MujocoARConnector
    print("Initializing MujocoARConnector...")
    connector = MujocoARConnector()
    connector.start()
    print("MujocoARConnector started. Please connect your iOS device.")

    # Initialize the AR input processor
    ar_processor = ARInputProcessor(connector)
    reward_plotter = RewardPlotter()

    # Create directory for saving videos if it doesn't exist
    video_dir = "mani_skill/envs/tasks/coin_bench/medias/videos"
    os.makedirs(video_dir, exist_ok=True)

    # Initialize video writer variables
    video_writer = None
    video_path = ""

    try:
        while True:
            # Create the environment
            output_dir = f"{args.record_dir}/{args.env_id}/"
            env = gym.make(
                args.env_id,
                obs_mode=args.obs_mode,
                control_mode="pd_ee_delta_pose",  # Use delta position control
                robot_uids=args.robot_uid,
                render_mode="human",
            )

            # Generate a unique trajectory name using timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_name = f"trajectory_{timestamp}"

            # Skip recording for now to avoid file locking issues
            env = RecordEpisode(
                env,
                output_dir=output_dir,
                trajectory_name=trajectory_name,
                save_video=args.save_video,
                info_on_video=False,
                source_type="ar_teleoperation",
                source_desc="teleoperation via MujocoARConnector",
            )

            # Reset the environment
            seed = 0
            obs, info = env.reset()

            # Display initial camera views
            initial_img, title = display_camera_views(obs)
            cv2.waitKey(1)
            cv2.moveWindow(title, 500, 400)

            # Setup video writer
            task_name = args.env_id.replace("-", "_")
            video_path = os.path.join(video_dir, f"{task_name}_{timestamp}.mp4")

            # Get frame dimensions from the initial image
            if initial_img is not None:
                h, w = initial_img.shape[:2]
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # Use uppercase 'MP4V'
                video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                print(f"Recording video to: {video_path}")
            else:
                video_writer = None
                print(
                    "Warning: Could not initialize video recording (no camera views available)"
                )

            # Print action space information
            print("\nAction Space Information:")
            print(f"Action Space: {env.action_space}")
            print(f"Action Space Shape: {env.action_space.shape}")
            print(f"Action Space Low: {env.action_space.low}")
            print(f"Action Space High: {env.action_space.high}")
            print(f"Action Space Sample: {env.action_space.sample()}\n")

            # Initialize gripper toggle state
            gripper_toggle = False  # Initialize toggle state to False (gripper open)
            prev_button_pressed = False

            print("Starting teleoperation. Press Ctrl+C to exit.")
            print("Controls: AR device movement controls the end-effector position")
            print(
                "         Toggle switch controls the gripper (toggle ON = gripper CLOSED, toggle OFF = gripper OPEN)"
            )
            print("         Button press resets the environment")
            steps = 0

            done = False
            while not done:
                # Process AR input to get delta pose and button states
                done = True
                delta_pose, button_pressed, toggle_state = ar_processor.process_input()
                steps += 1
                # print(steps)
                if delta_pose is None or np.all(delta_pose == 0):
                    # print("Waiting for AR data...")
                    time.sleep(5)
                    continue

                # Extract position and rotation components
                delta_position = delta_pose[:3]
                delta_rotation = delta_pose[3:]

                # Create action vector (depends on the environment's action space)
                action = np.zeros(env.action_space.shape[0])

                # Set delta position (first 3 elements)
                action[:3] = delta_position

                # Set delta rotation as quaternion (next 4 elements, if available)
                if len(action) >= 7:  # Make sure the action space includes rotation
                    action[3:6] = delta_rotation

                # Use toggle state for gripper control
                if len(action) >= 7:  # Make sure the action space includes gripper
                    # Set gripper action based on toggle state (1 = close, 0 = open)
                    action[6] = 1.0 if toggle_state else 0.0

                    # Update gripper toggle state for display
                    if gripper_toggle != toggle_state:
                        gripper_toggle = toggle_state
                        print(f"Gripper {'CLOSED' if gripper_toggle else 'OPEN'}")
                # print('robot action is ', )
                # Use button press to reset the environment
                if button_pressed and not prev_button_pressed:
                    print("Resetting environment...")
                    env.close()
                    if video_writer is not None:
                        video_writer.release()
                        print(f"Video saved to: {video_path}")
                    break
                # Update previous button state
                prev_button_pressed = button_pressed

                # Take a step in the environment
                if steps <= 5:
                    action = action * 0
                obs, reward, terminated, truncated, info = env.step(action)

                # print(info)
                # print("reward: ", reward)
                reward_plotter.update(reward)
                # Display camera views from the observation with 1024x1024 target size
                # and save the frame to the video
                frame, _ = display_camera_views(obs)

                # Write frame to video if available
                if frame is not None and video_writer is not None:
                    video_writer.write(frame)

                cv2.waitKey(1)  # Process any pending window ervents

                # Render the environment
                env.render_human()

                # Print success information
                if info.get("success", False):
                    print("Task completed successfully!")

                # Check if episode is done
                done = terminated or truncated
                if done:
                    print("Episode finished. Resetting environment.")
                    seed += 1
                    # obs, info = env.reset()
                    # print(info)
                    # done = False
                    # # Display camera views after reset
                    # img, _ = display_camera_views(obs)
                    # # Write final frame to video if available
                    # if img is not None and video_writer is not None:
                    #     video_writer.write(img)
                    # cv2.waitKey(1)
                    #
                    # # Release the video writer
                    if video_writer is not None:
                        video_writer.release()
                        print(f"Video saved to: {video_path}")

                    env.close()
                    break

                # Sleep to control the loop rate
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nTeleoperation stopped by user.")
    finally:
        # Stop the AR connector
        connector.stop()
        print("MujocoARConnector stopped.")

        # Close the environment and release video writer if it exists
        if "video_writer" in locals() and video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()
