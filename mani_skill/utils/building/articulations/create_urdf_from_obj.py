import os
import xml.etree.ElementTree as ET
from pathlib import Path

import os
import re
import pybullet as p


def calculate_convex(input_file, output_file):
    p.vhacd(
        input_file,
        output_file,
        "vhacd_log.txt",
        resolution=1000000,
        depth=20,
        concavity=0.0025,
        planeDownsampling=4,
        alpha=0.05,
        beta=0.05,
        maxNumVerticesPerCH=64,
        minVolumePerCH=0.0001,
        pca=0,
        mode=0,
        convexhullApproximation=1,
    )


def decompose(folder, file_name):
    input_file = os.path.join(folder, file_name + ".obj")
    output_file = os.path.join(folder, file_name + "_collision.obj")
    calculate_convex(input_file, output_file)


def generate_urdf_from_folder(folder_path, output_path=None):
    """
    Generate URDF files by finding .obj and _collision.obj files in a folder
    and updating the base URDF template.

    Args:
        folder_path (str): Path to the folder containing .obj files
        output_path (str, optional): Path to save the generated URDF files
    """

    # Convert to Path object for easier handling
    folder_path = Path(folder_path)
    if output_path is None:
        output_path = folder_path

    # Base URDF template
    base_urdf = """<?xml version="1.0" ?>
<robot name="generated_robot">
    <link name="base"/>
    <link name="link_0">
        <visual name="door_frame-6">
            <origin xyz="0 0 0 "/>
            <geometry>
                <mesh filename="MESH_FILE"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0 "/>
            <geometry>
                <mesh filename="COLLISION_FILE"/>
            </geometry>
        </collision>
    </link>    
    <joint name="joint_0" type="fixed">
        <origin xyz="0 0 0"/>
        <axis xyz="0 -1 0"/>
        <child link="base"/>
        <parent link="link_0"/>
    </joint>
</robot>"""

    # Find all .obj files that don't have _collision in their name
    obj_files = [f for f in folder_path.glob("*.obj") if "_collision" not in f.stem]

    for obj_file in obj_files:
        # Check if corresponding collision file exists
        collision_file = obj_file.with_stem(f"{obj_file.stem}_collision")

        if not collision_file.exists():
            print(f"Warning: No collision file found for {obj_file.name}")
            continue

        # Parse the base URDF
        tree = ET.ElementTree(ET.fromstring(base_urdf))
        root = tree.getroot()

        # Update mesh filenames
        for mesh in root.findall(".//mesh"):
            if mesh.get("filename") == "MESH_FILE":
                mesh.set("filename", str(obj_file.name))
            elif mesh.get("filename") == "COLLISION_FILE":
                mesh.set("filename", str(collision_file.name))

        # Generate output filename
        output_filename = output_path / f"{obj_file.stem}.urdf"

        # Write the modified URDF
        tree.write(output_filename)
        print(f"Generated URDF: {output_filename}")


def main():
    # Example usage
    # folder_path = input("Enter the folder path containing .obj files: ")
    file_name = "hanoi_mid"
    folder_path = f"partnet-mobility-dataset/{file_name}"
    decompose(folder_path, file_name)
    try:
        generate_urdf_from_folder(folder_path)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
