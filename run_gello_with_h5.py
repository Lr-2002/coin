import h5py
import cv2
import gymnasium as gym 
import time 
import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from mani_skill.utils.display_multi_camera import display_camera_views


# h5_file = "trajectory_20250414_161246.h5"
h5_file = "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Find-Book-From-Shelf-v1/trajectory_20250414_160747.h5"

with h5py.File(h5_file, "r") as f:
    tcp_pose = f["traj_0/obs/extra/tcp_pose"][:]
    # breakpoint()
    gripper = f["traj_0/actions"][:, -1]
    origin_actions = f["traj_0/actions"][:]
    angles = np.array([[*quat2euler(pose[3:])] for pose in tcp_pose])
    # breakpoint()
    actions = np.concatenate([tcp_pose[:, :3], angles, np.zeros((angles.shape[0] ,1))], axis=1)
    delta_actions = np.diff(actions, axis=0)
    # np.save("actions_from_tcp_pose.npy", actions)
# env = gym.make(
#     'Tabletop-Find-Book-From-Shelf-v1',
#     obs_mode='rgbd',
#     control_mode="pd_joint_pos",  # Use delta position control
#     robot_uids='panda_wristcam',
#     render_mode="human",
#     )
# env.reset()
# for action in origin_actions : 
#     print(action)
#     obs, _, _, _, _ = env.step(action)
#     display_camera_views(obs)
#     cv2.waitKey(1)  # Process any pending window ervents
#
#     time.sleep(0.02)

import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation
# 加载URDF
# /robots/panda/panda_v3.urdf
urdf_path = "mani_skill/agents/robots/panda/panda_v3.urdf"
model = pin.BuildFromURDF(urdf_path)
data = model.createData()
ee_frame_id = model.getFrameId("panda_link7")  # 替换为实际frame名称

# 假设输入数据
joint_data = origin_actions  # 模拟数据，替换为你的实际数据
joint_angles = joint_data[:, 0:7]  # 关节角度 (297, 7)

# 计算位姿
poses = []
for q in joint_angles:
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    poses.append(data.oMf[ee_frame_id])

# 计算增量位姿
delta_poses = []
for i in range(len(poses) - 1):
    pose_t = poses[i]
    pose_t1 = poses[i + 1]
    delta_p = pose_t1.translation - pose_t.translation
    delta_R = pose_t.rotation.T @ pose_t1.rotation
    delta_quat = Rotation.from_matrix(delta_R).as_quat()
    delta_poses.append({
        "delta_p": delta_p,
        "delta_R": delta_R,
        "delta_quat": delta_quat
    })

# 提取结果
delta_p_array = np.array([dp["delta_p"] for dp in delta_poses])
delta_quat_array = np.array([dp["delta_quat"] for dp in delta_poses])

print("位置增量形状:", delta_p_array.shape)  # (296, 3)
print("四元数增量形状:", delta_quat_array.shape)  # (296, 4)


#
#
#
#
# env = gym.make(
#     'Tabletop-Find-Book-From-Shelf-v1',
#     obs_mode='rgbd',
#     control_mode="pd_ee_delta_pose",  # Use delta position control
#     robot_uids='panda_wristcam',
#     render_mode="human",
#     )
# env.reset()
# for action in delta_actions : 
#     print(action)
#     obs, _, _, _, _ = env.step(action)
#     display_camera_views(obs)
#     cv2.waitKey(1)  # Process any pending window ervents
#
#     time.sleep(0.02)
# print(action.shape)
# breakpoint()
