import os
import subprocess

root_dir = (
    "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset"
)

# Walk through all subdirectories
results = {}

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".json") and "cpu" not in filename:
            json_path = os.path.join(dirpath, filename)
            replay_json = json_path.replace(
                ".json", ".rgbd.pd_ee_delta_pose.physx_cpu.json"
            )
            if os.path.exists(replay_json):
                try:
                    with open(replay_json, "r") as f:
                        replay_data = __import__("json").load(f)
                    # Check if the first episode is successful
                    if (
                        "episodes" in replay_data
                        and replay_data["episodes"]
                        and replay_data["episodes"][0].get("success", False)
                    ):
                        continue  # Already successful, skip
                except Exception as e:
                    print(f"Warning: Failed to read or parse {replay_json}: {e}")
                    # If error, proceed to try replay

            h5_path = json_path.replace(".json", ".h5")

            # Construct the command
            cmd = [
                "python",
                "-m",
                "mani_skill.trajectory.replay_trajectory",
                "--traj-path",
                h5_path,
                "-c",
                "pd_ee_delta_pose",
                "-o",
                "rgbd",
                # "--max_retry",
                # "3",
                "--save-traj",
                # "--vis",
                # "--verbose",
            ]

            print(f"Running command: {' '.join(cmd)}")
            try:
                completed = subprocess.run(
                    cmd, check=True, capture_output=True, text=True
                )
                output = completed.stdout + completed.stderr
                if "Replayed 1 episodes, 1/1=100.00% demos saved" in output:
                    results[h5_path] = "success"
                else:
                    results[h5_path] = f"fail (no success message)"
            except subprocess.CalledProcessError as e:
                results[h5_path] = f"fail (exception): {e}"

# Print and save the summary
import json as _json

summary_path = os.path.join(root_dir, "replay_results_summary.json")
with open(summary_path, "w") as f:
    _json.dump(results, f, indent=2)
print("Replay summary saved to:", summary_path)
