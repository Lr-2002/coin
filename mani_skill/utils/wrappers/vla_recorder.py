#!/usr/bin/env python3
"""
VLA Recorder Wrapper for ManiSkill

This wrapper extends functionality to record:
1. Model class (pi0, gr00t, cogact)
2. Dataset version (git commit ID)
3. Env version (git commit ID)
4. Model path (real path)
5. Env name
6. Time (second level)
7. Success status
8. Video path (mp4)
9. Action path (pkl)
10. For HVLA: record all chatting progress

Structure follows the diagram provided in the task description.
"""

import os
import time
import json
import pickle
import logging
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import importlib.resources

import gymnasium as gym
import h5py

# Import from ManiSkill
from mani_skill import get_commit_info
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.io_utils import dump_json

logger = logging.getLogger(__name__)


def get_git_commit_id(repo_path: str) -> str:
    """Get the Git commit ID for a repository path."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        logger.warning(f"Failed to get Git commit ID for {repo_path}")
        return "unknown"


class VLARecorderWrapper(gym.Wrapper):
    """
    Wrapper to record VLA and hierarchical VLA testing data.

    This wrapper extends the RecordEpisode wrapper with additional metadata
    recording capabilities specific to VLA testing.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        model_class: str,
        model_path: str,
        dataset_repo_path: Optional[str] = None,
        env_repo_path: Optional[str] = None,
        is_hierarchical: bool = False,
        record_chat: bool = False,
        save_trajectory: bool = True,
        save_video: bool = True,
        video_fps: int = 30,
        external_camera: Optional[str] = None,
        cameras: Optional[str] = ["human_camera", "hand_camera", "base_front_camera"],
        **record_kwargs,
    ):
        """
        Initialize the VLA recorder wrapper.

        Args:
            env: The environment to wrap
            output_dir: Directory to save recordings
            model_class: Model class (pi0, gr00t, cogact)
            model_path: Path to the model
            dataset_repo_path: Path to dataset repository (for commit ID)
            env_repo_path: Path to environment repository (for commit ID)
            is_hierarchical: Whether this is a hierarchical VLA
            record_chat: Whether to record chat progress (for HVLA)
            save_trajectory: Whether to save trajectory data
            save_video: Whether to save video
            video_fps: FPS for video recording
            external_camera: Name of the external camera to use
            cameras: List of cameras to use
            **record_kwargs: Additional arguments for RecordEpisode
        """
        # Generate timestamp for consistent file naming across all recordings
        # Apply RecordEpisode wrapper first
        # Generate timestamp for consistent file naming across all recordings
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store parameters
        # self.output_dir = Path(output_dir) / self.timestamp
        self.original_output_dir = output_dir
        self.action_shape = None
        env = RecordEpisode(
            env=env,
            output_dir=output_dir,
            save_trajectory=False,
            # trajectory_name=f"traj_{self.timestamp}",
            save_video=save_video,
            save_on_reset=False,
            video_fps=video_fps,
            **record_kwargs,
        )

        # Initialize gym.Wrapper
        super().__init__(env)

        self.model_class = model_class
        self.model_path = model_path
        self.dataset_repo_path = dataset_repo_path
        self.env_repo_path = env_repo_path
        self.is_hierarchical = is_hierarchical
        self._enable_chat_recording = record_chat
        self.external_camera = external_camera
        self.cameras = cameras

        # Create metadata directory if it doesn't exist
        # self.metadata_dir = self.output_dir / "metadata"
        # os.makedirs(self.metadata_dir, exist_ok=True)

        # Initialize chat history if needed
        self.chat_history = [] if record_chat else None

        # Collect metadata
        self.metadata = self._collect_metadata()

        # Initialize recording section in metadata
        self.metadata["recording"] = {}
        # video_path = self.output_dir / f"traj_{self.timestamp}.mp4"
        # self.metadata["recording"]["video_paths"] = [str(video_path)]

        # Initialize action paths
        # pkl_path = self.output_dir / f"traj_{self.timestamp}.pkl"
        # self.metadata["recording"]["action_paths"] = [str(pkl_path)]

        # Save initial metadata
        # self._save_metadata()

        # Initialize additional recording structures
        self.success = False
        self.num_step = 0
        self.start_time = None
        self.episode_duration = None

        self.action_history = []

        # System prompt
        self.prompt_file = "prompt.txt"

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect metadata about the current run."""
        # Get environment ID
        env_id = (
            self.env.unwrapped.spec.id
            if hasattr(self.env.unwrapped, "spec")
            and self.env.unwrapped.spec is not None
            else "unknown"
        )

        # Get Git commit IDs
        dataset_commit_id = (
            get_git_commit_id(self.dataset_repo_path)
            if self.dataset_repo_path
            else "unknown"
        )
        env_commit_id = (
            get_git_commit_id(self.env_repo_path) if self.env_repo_path else "unknown"
        )

        # Get ManiSkill commit info
        maniskill_commit_info = get_commit_info()

        # Collect metadata
        metadata = {
            "model": {
                "class": self.model_class,
                "path": self.model_path,
            },
            "dataset": {
                "commit_id": dataset_commit_id,
            },
            "environment": {
                "name": env_id,
                "maniskill_commit": maniskill_commit_info,
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_hierarchical": self.is_hierarchical,
        }

        return metadata

    def _save_metadata(self):
        """Save metadata to JSON file."""
        metadata_path = os.path.join(self.metadata_dir, "metadata.json")
        dump_json(metadata_path, self.metadata, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    def _update_and_save_metadata(self):
        """Update and save metadata."""
        # Update metadata
        # self.metadata["camera_for_vla"] = (
        #     ["hand_camera", f"{self.external_camera}"]
        #     if self.metadata["model"]["class"] == "cogact"
        #     else ["base_front_camera", "hand_camera", f"{self.external_camera}"]
        # )
        self.metadata["camera_for_vla"] = self.cameras

        self.metadata["success"] = self.success
        self.metadata["step"] = self.num_step
        self.metadata["episode_duration"] = self.episode_duration

        # Save action history
        if self.action_history == []:
            action_history = np.zeros(3)
        else:
            # breakpoint()
            action_history = np.stack(self.action_history, axis=0)
        # pkl_path = os.path.join(self.output_dir, f"traj_{self.timestamp}.pkl")

        with open(self.pkl_path, "wb") as f:
            pickle.dump(action_history, f)
        logger.info(f"Saved action history to {self.pkl_path}")

        # # Ensure action path is in metadata
        # if "recording" not in self.metadata:
        #     self.metadata["recording"] = {}
        # self.metadata["recording"]["action_paths"] = [str(self.pkl_path)]

        # h5_path = self.output_dir / f"traj_{self.timestamp}.h5"

        # # Check for existing files in priority order
        # if pkl_path.exists():
        #     action_paths.append(str(pkl_path))
        # elif h5_path.exists():
        #     action_paths.append(str(h5_path))
        # elif hasattr(self.env, "_trajectory_id"):
        #     # Fallback to the old naming format if necessary
        #     traj_id = self.env._trajectory_id
        #     old_h5_path = self.output_dir / f"traj_{traj_id}.h5"

        #     if old_h5_path.exists():
        #         try:
        #             # Convert h5 to pkl format with our timestamp, storing only actions
        #             with h5py.File(old_h5_path, 'r') as h5f:
        #                 # Extract only action data from h5 file
        #                 actions = {}
        #                 for key in h5f.keys():
        #                     if 'action' in h5f[key]:
        #                         actions[key] = h5f[key]['action'][()]
        #                 data = {'actions': actions}

        #             # Save as pickle with our naming format
        #             with open(pkl_path, 'wb') as f:
        #                 pickle.dump(data, f)

        #             action_paths.append(str(pkl_path))
        #         except Exception as e:
        #             logger.error(f"Error converting h5 to pkl: {e}")
        #             action_paths.append(str(old_h5_path))

        # # Add chat history for hierarchical VLA if available
        # if self.is_hierarchical and self.chat_history:
        #     chat_path = self.output_dir / f"traj_{self.timestamp}.html"

        # # Create an HTML file with the chat history
        # try:
        #     html_content = "<html><head><title>Chat History</title></head><body>\n"
        #     for msg in self.chat_history:
        #         role = msg.get("role", "unknown")
        #         content = msg.get("content", "")
        #         msg_timestamp = msg.get("timestamp", "")

        #         html_content += f"<div class='{role}'>\n"
        #         html_content += (
        #             f"<p><strong>{role.upper()} ({msg_timestamp}):</strong></p>\n"
        #         )
        #         html_content += f"<p>{content}</p>\n"
        #         html_content += "</div>\n<hr>\n"

        #     html_content += "</body></html>"

        #     with open(chat_path, "w") as f:
        #         f.write(html_content)

        #     self.metadata["recording"]["chat_path"] = str(chat_path)
        # except Exception as e:
        #     logger.error(f"Error creating HTML chat history: {e}")

        #     # Fallback to JSON format with correct naming
        #     json_chat_path = self.output_dir / f"traj_{self.timestamp}_chat.json"
        #     with open(json_chat_path, "w") as f:
        #         json.dump(self.chat_history, f, indent=2)
        #     self.metadata["recording"]["chat_path"] = str(json_chat_path)

        # Save updated metadata
        self._save_metadata()

    def reset(self, **kwargs):
        """Reset the environment and start timing."""
        # Record start time
        self.start_time = time.time()
        self.env_id = self.env.unwrapped.spec.id

        # New output dir for new episode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_hierarchical:
            self.output_dir = os.path.join(
                self.original_output_dir,
                f"{self.timestamp}_{self.env_id}_{self.model_class}_{self.llm_model}",
            )
        else:
            self.output_dir = os.path.join(
                self.original_output_dir,
                f"{self.timestamp}_{self.env_id}_{self.model_class}",
            )

        # Metadata
        self.metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.metadata = self._collect_metadata()
        self.metadata["recording"] = {}

        # Update video and action paths in metadata
        self.video_path = os.path.join(self.output_dir, f"traj_{self.timestamp}.mp4")
        self.pkl_path = os.path.join(self.output_dir, f"traj_{self.timestamp}.pkl")
        self.metadata["recording"]["video_paths"] = [str(self.video_path)]
        self.metadata["recording"]["action_paths"] = [str(self.pkl_path)]

        # Reset success status
        self.success = False

        self.action_history = []
        self.num_step = 0

        # Reset chat history if needed
        if self._enable_chat_recording:
            self.chat_history = []

        # Reset environment
        return self.env.reset()

    def step(self, action):
        """Step the environment and record data."""
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.action_shape:
            self.action_shape = np.array(action).shape
        # Update success status
        if info.get("success", False):
            self.success = True

        # Update episode duration
        self.episode_duration = time.time() - self.start_time

        # Record action
        record_action = np.array(action).reshape(self.action_shape)
        self.action_history.append(record_action)
        self.num_step += 1

        return obs, reward, terminated, truncated, info

    def record(self):
        """Close the environment and finalize metadata."""
        # Update episode duration if not already set
        if self.start_time is not None and self.episode_duration is None:
            self.episode_duration = time.time() - self.start_time

        # Update and save metadata
        self._update_and_save_metadata()
        if self._enable_chat_recording:
            self.save_chat_history()

        # Ensure consistent video naming before closing
        if hasattr(self.env, "flush_video"):
            # Record video with our naming convention directly
            try:
                # Manual call to flush_video with our naming convention
                self.env.flush_video(
                    name=f"traj_{self.timestamp}", output_dir=self.output_dir
                )
            except Exception as e:
                logger.error(f"Error when flushing video with custom name: {e}")

    def close(self):
        super().close()
        self.record()


class HVLARecorderWrapper(VLARecorderWrapper):
    """
    Specialized wrapper for Hierarchical VLA (HVLA) recording.

    This extends the VLARecorderWrapper with additional features specific to
    hierarchical VLA agents, including plan recording.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: str,
        model_class: str,
        model_path: str,
        llm_model: str,
        cameras: Optional[str] = None,
        dataset_repo_path: Optional[str] = None,
        env_repo_path: Optional[str] = None,
        external_camera: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the HVLA recorder wrapper.

        Args:
            env: The environment to wrap
            output_dir: Directory to save recordings
            model_class: Model class (pi0, gr00t, cogact)
            model_path: Path to the model
            llm_model: LLM model name used for high-level planning
            dataset_repo_path: Path to dataset repository (for commit ID)
            env_repo_path: Path to environment repository (for commit ID)
            external_camera: Name of the external camera to use
            cameras: List of cameras to use
            **kwargs: Additional arguments for VLARecorderWrapper
        """ 
        # Initialize parent class
        super().__init__(
            env=env,
            output_dir=output_dir,
            model_class=model_class,
            model_path=model_path,
            dataset_repo_path=dataset_repo_path,
            env_repo_path=env_repo_path,
            is_hierarchical=True,
            record_chat=True,  # Always record chat for HVLA
            external_camera=external_camera,
            cameras=cameras,
            **kwargs,
        )

        # Store LLM model
        self.llm_model = llm_model
        self.metadata["llm_model"] = llm_model

    def record_chat_func(self, step, input_instruction, input_image, response):
        """
        Record a chat message (for hierarchical VLA).

        Args:
            role: 'system', 'user', or 'assistant'
            content: Message content
        """
        chat_entry = {
            "step": step,
            "input_instruction": input_instruction,
            "response": response,
        }
        if input_image:
            chat_entry["input_image"] = input_image

        self.chat_history.append(chat_entry)

    def save_chat_history(self):
        prompt_file = importlib.resources.files("mani_skill.prompts") / self.prompt_file
        try:
            with open(prompt_file, "r") as f:
                system_prompt = f.read()
        except Exception as e:
            logger.error(f"Error reading prompt file: {e}")
            # Fallback to default prompt
            system_prompt = None
        chat_json = {
            "system_prompt": system_prompt,
            # add other metadata as needed
            "data": self.chat_history,
        }
        chat_history_path = os.path.join(
            self.output_dir, f"traj_{self.timestamp}_chat.json"
        )
        with open(chat_history_path, "w") as f:
            json.dump(chat_json, f, indent=2)

    def _update_and_save_metadata(self):
        """Update and save metadata with HVLA-specific information."""
        # First, call the parent method
        super()._update_and_save_metadata()

        # Save updated metadata
        self._save_metadata()

    def save_VQA_answer(self, vqa):
        """Save the VQA answer to the metadata."""
        if vqa['idx'] == 'initial':
            vqa_path = os.path.join(self.output_dir, f"traj_{self.timestamp}_vqa_{vqa[idx]}.json")
        else:
            vqa_path = os.path.join(self.output_dir, f"traj_{self.timestamp}_vqa.json")
        with open(vqa_path, "w") as f:
            json.dump(vqa, f, indent=2)
        self.metadata["vqa_answer"] = vqa
        self._save_metadata()
