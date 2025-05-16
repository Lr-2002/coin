import os
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import os
import re
import pybullet as p
import bpy
import sys
import os


def convert_glb_to_obj(input_path, output_path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Try importing
    imported = False
    try:
        bpy.ops.import_scene.gltf(filepath=input_path)
        print("Successfully imported using gltf")
        imported = True
    except Exception as e:
        print(f"Error importing GLTF: {str(e)}")
        return False

    if not imported:
        print("Failed to import file")
        return False

    # Print available export operators
    print("Available export operators:")
    for op in dir(bpy.ops.export_scene):
        print(f" - {op}")

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Try exporting
    try:
        # Standard OBJ export
        bpy.ops.export_scene.obj(
            filepath=output_path,
            use_selection=False,
            use_materials=True,
            use_triangles=True,
        )
        print(f"Successfully exported to {output_path}")
        return True
    except AttributeError:
        print("Standard OBJ export operator not available")
        # Try alternative OBJ export (older operator name)
        try:
            bpy.ops.wm.obj_export(
                filepath=output_path,
                export_selected_objects=False,
                export_materials=True,
                export_triangulated_mesh=True,
            )
            print(f"Successfully exported using wm.obj_export to {output_path}")
            return True
        except Exception as e:
            print(f"Alternative export failed: {str(e)}")
            return False
    except Exception as e:
        print(f"Error exporting OBJ: {str(e)}")
        return False


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


def create_json_config(file_name, output_path="config.json"):
    # Construct the URDF path using the file_name variable
    urdf_path = f"partnet-mobility-dataset/{file_name}/{file_name}.urdf"

    # Define the configuration dictionary
    config = {
        "urdf_path": urdf_path,
        "scale": 0.01,
        "mass": 0.2,
        "fix_root_link": False,
        "urdf_config": {
            "material": {
                "static_friction": 2.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
            }
        },
        "name": file_name,
    }

    # Write to JSON file
    try:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Successfully wrote JSON config to {os.path.realpath(output_path)}")
    except Exception as e:
        print(f"Error writing JSON file: {str(e)}")
        return False

    return True


def process_one(file_name="hanoi_biggest"):
    # Example usage
    # folder_path = input("Enter the folder path containing .obj files: ")
    glb_path = f"assets_glb/{file_name}.glb"
    glb_path = os.path.abspath(glb_path)
    folder_path = f"partnet-mobility-dataset/{file_name}"
    obj_path = os.path.abspath(folder_path) + f"/{file_name}.obj"
    os.makedirs(folder_path, exist_ok=True)
    convert_glb_to_obj(input_path=glb_path, output_path=obj_path)
    decompose(folder_path, file_name)
    try:
        generate_urdf_from_folder(folder_path)
    except Exception as e:
        print(f"Error: {str(e)}")

    create_json_config(file_name, f"configs/{file_name}.json")


if __name__ == "__main__":
    # for name in ["base", "small", "biggest", "mid"]:
    # process_one(f"hanoi_{name}")
    process_one("plate_new")
