import os
import shutil
from test_hdf5 import convert_h5_images_to_mp4
import json


def has_true_value(data):
    """递归检查JSON数据中是否包含值为True的字段"""
    if isinstance(data, dict):
        for value in data.values():
            if has_true_value(value):
                return True
    elif isinstance(data, list):
        for item in data:
            if has_true_value(item):
                return True
    elif isinstance(data, bool) and data is True:
        return True
    return False


def find_json_with_true(current_dir):
    """遍历当前目录，查找文件名含'delta_pos'且JSON内容有True的字段"""
    # current_dir = os.getcwd()  # 获取当前目录
    for filename in os.listdir(current_dir):
        # 检查文件是否为JSON文件且文件名包含'delta_pos'
        if filename.endswith(".json") and "delta_pos" in filename:
            try:
                filename = os.path.join(current_dir, filename)
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 检查JSON内容是否包含True
                    if has_true_value(data):
                        print(f"文件 '{filename}' 中包含值为 True 的字段")
                        shutil.copyfile(
                            filename, "./double_dataset/" + filename.split("/")[-1]
                        )
                        shutil.copyfile(
                            filename.replace("json", "h5"),
                            "./double_dataset/"
                            + filename.split("/")[-1].replace("json", "h5"),
                        )  # convert_h5_images_to_mp4(
                        #     filename.replace(".json", ".h5"),
                        #     output_dir="./conveted_videos/",
                        # )
            except json.JSONDecodeError:
                print(f"文件 '{filename}' 不是有效的JSON文件")
            except Exception as e:
                print(f"处理文件 '{filename}' 时出错: {e}")


if __name__ == "__main__":
    path = "/home/lr-2002/project/reasoning_manipulation/gello_software/teleoperation_dataset/Tabletop-Open-Drawer-v1/"
    find_json_with_true(path)
