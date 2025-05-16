#!/bin/bash

# 指定目录
DIR="/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Pick-Book-FromShelf-v1/"

# 统计 .json 文件数量
json_count=$(find "$DIR" -maxdepth 1 -type f -name "*.json" | wc -l)
echo "Number of .json files: $json_count"

# 查找不含 pd_ee_delta 的 .json 文件，生成 .h5 文件名并执行命令
echo "Processing .h5 files (converted from .json, excluding those containing pd_ee_delta):"
find "$DIR" -maxdepth 1 -type f -name "*.json" ! -name "*pd_ee_delta*" -exec basename {} .json \; | while read -r base_name; do
  h5_file="${base_name}.h5"
  echo "Running command for $h5_file"
  python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$DIR/$h5_file" \
    -c pd_ee_delta_pose -o rgbd --save-traj --allow-failure --save-video --use-first-env-state
done
