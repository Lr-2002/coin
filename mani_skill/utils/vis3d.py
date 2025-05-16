import rerun as rr
import numpy as np
from dataclasses import dataclass

@dataclass
class BBoxObject:
    """定义一个3D bounding box对象"""
    position: list  # 中心位置 [x, y, z]
    rotation: list  # 四元数 [w, x, y, z]
    bounds: list    # [xmin, ymin, zmin, xmax, ymax, zmax]

class BBoxVisualizer:
    def __init__(self, save_path=None):
        """
        初始化可视化类
        
        参数:
        - save_path: str，可选，保存路径
        """
        self.save_path = save_path
        self.cnt = 0
        # 初始化Rerun
        rr.init("dynamic_bboxes", spawn=True)

    def _compute_center_and_size(self, bounds, position):
        """从min/max坐标和初始位置计算中心和半尺寸"""
        xmin, ymin, zmin, xmax, ymax, zmax = bounds
        # 计算局部中心（相对于bounds定义）
        local_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
        # 加上全局位置偏移
        center = [position[0] + local_center[0], 
                  position[1] + local_center[1], 
                  position[2] + local_center[2]]
        half_size = [(xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2]
        return center, half_size

    def visualize(self, bbox1: BBoxObject, bbox2: BBoxObject):
        """
        显示两个输入的bbox，更新可视化内容
        
        参数:
        - bbox1: BBoxObject，第一个bbox
        - bbox2: BBoxObject，第二个bbox
        """
        rr.set_time_sequence('frames', self.cnt)
        # 计算bbox1的中心和半尺寸
        center1, half_size1 = self._compute_center_and_size(bbox1.bounds, bbox1.position)
        rr.log(
            "bbox1",
            rr.Boxes3D(
                centers=center1,
                half_sizes=half_size1,
                rotations=bbox1.rotation,  # 四元数 [w, x, y, z]
                colors=[255, 0, 0],  # 红色
                labels=["Box1"]
            )
        )

        # 计算bbox2的中心和半尺寸
        center2, half_size2 = self._compute_center_and_size(bbox2.bounds, bbox2.position)
        rr.log(
            "bbox2",
            rr.Boxes3D(
                centers=center2,
                half_sizes=half_size2,
                rotations=bbox2.rotation,  # 四元数 [w, x, y, z]
                colors=[0, 0, 255],  # 蓝色
                labels=["Box2"]
            )
        )

        # 可选保存（每次调用都会覆盖之前的文件）
        if self.save_path:
            rr.save(self.save_path)
            print(f"数据已保存到 {self.save_path}")

# 示例调用
if __name__ == "__main__":
    # 创建可视化实例
    vis = BBoxVisualizer(save_path="dynamic_bboxes.rrd")

    # 定义初始的两个bbox对象
    bbox1 = BBoxObject(
        position=[0, 0, 0],           # 中心位置
        rotation=[1, 0, 0, 0],        # 无旋转（四元数）
        bounds=[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]  # 1x1x1立方体
    )
    bbox2 = BBoxObject(
        position=[2, 2, 0],
        rotation=[0.707, 0, 0, 0.707],  # 绕Z轴旋转45度
        bounds=[0.5, 0.5, -0.3, 1.5, 1.5, 0.3]  # 1x1x0.6的长方体
    )

    # 第一次调用，显示初始状态
    vis.visualize(bbox1, bbox2)

    # 更新bbox1的位置
    bbox1.position = [1, 0, 0]
    vis.visualize(bbox1, bbox2)  # 更新显示
