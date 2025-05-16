#!/usr/bin/env python3

import argparse
import os
import numpy as np
import sapien
from sapien.utils import Viewer


def create_scene():
    """Create a SAPIEN scene with a renderer"""
    # Create a scene
    scene = sapien.Scene()

    # Add lighting
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])
    scene.add_point_light([1, 2, 2], [1, 1, 1])

    return scene


def load_usd_file(scene, usd_path, scale=1.0, show_collision=False):
    """Load a USD file into the scene and optionally show collision geometries"""
    print(f"Loading USD file: {usd_path}")

    # Create a loader
    loader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = True

    try:
        # Try to load the USD file
        usd_object = loader.load(usd_path)

        if usd_object is None:
            print(f"Failed to load USD file: {usd_path}")
            create_fallback_object(scene)
            return None

        print(f"Successfully loaded USD file: {usd_path}")
        print(f"Object name: {usd_object.name}")

        # Try to get links if it's an articulation
        try:
            links = usd_object.get_links()
            print(f"Links: {[link.name for link in links]}")

            # Show collision geometries if requested
            if show_collision:
                for link in links:
                    for collision_shape in link.get_collision_shapes():
                        # Create visual for each collision shape with translucent red material
                        material = scene.create_physical_material(1, 1, 0)
                        scene.add_visual_from_shape(
                            collision_shape, sapien.Pose(), material, link
                        )
                        # Set the visual to be translucent red
                        for visual in link.get_visuals():
                            if visual.name.startswith("visual_"):
                                visual.set_color([1, 0, 0, 0.3])
        except:
            # It's a rigid body, not an articulation
            print("Object is a rigid body, not an articulation")

            # Show collision geometries if requested
            if show_collision:
                for collision_shape in usd_object.get_collision_shapes():
                    # Create visual for each collision shape with translucent red material
                    material = scene.create_physical_material(1, 1, 0)
                    scene.add_visual_from_shape(
                        collision_shape, sapien.Pose(), material, usd_object
                    )
                    # Set the visual to be translucent red
                    for visual in usd_object.get_visuals():
                        if visual.name.startswith("visual_"):
                            visual.set_color([1, 0, 0, 0.3])

        # Set initial pose
        usd_object.set_pose(sapien.Pose())

        return usd_object

    except Exception as e:
        print(f"Error loading USD file: {e}")
        create_fallback_object(scene)
        return None


def create_fallback_object(scene):
    """Create a fallback object if USD loading fails"""
    print("Creating fallback object")
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[0.1, 0.1, 0.1])

    # Create a material for the visual
    material = scene.create_physical_material(1, 1, 0)
    builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=material)

    fallback.set_pose(sapien.Pose(p=[0, 0, 0.1]))
    fallback = builder.build(name="fallback_cube")

    # Set the color to red
    for visual in fallback.get_visuals():
        visual.set_color([1, 0, 0, 1])

    return fallback


def create_ground_plane(scene):
    """Create a ground plane"""
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[5, 5, 0.05])

    # Create a material for the visual
    material = scene.create_physical_material(1, 1, 0)
    builder.add_box_visual(half_size=[5, 5, 0.05], material=material)

    ground = builder.build_static(name="ground")
    ground.set_pose(sapien.Pose(p=[0, 0, -0.05]))

    # Set the color to gray
    for visual in ground.get_visuals():
        visual.set_color([0.5, 0.5, 0.5, 1])

    return ground


def create_coordinate_axes(scene, length=0.5):
    """Create coordinate axes for reference"""
    builder = scene.create_actor_builder()

    # Create materials
    material = scene.create_physical_material(1, 1, 0)

    # X-axis (red)
    builder.add_box_visual(half_size=[length / 2, 0.005, 0.005], material=material)
    # Y-axis (green)
    builder.add_box_visual(half_size=[0.005, length / 2, 0.005], material=material)
    # Z-axis (blue)
    builder.add_box_visual(half_size=[0.005, 0.005, length / 2], material=material)

    axes = builder.build_static(name="axes")
    axes.set_pose(sapien.Pose(p=[0, 0, 0]))

    # Set colors for each axis
    visuals = axes.get_visuals()
    visuals[0].set_color([1, 0, 0, 1])  # X-axis: red
    visuals[1].set_color([0, 1, 0, 1])  # Y-axis: green
    visuals[2].set_color([0, 0, 1, 1])  # Z-axis: blue

    return axes


def main():
    parser = argparse.ArgumentParser(
        description="Load and visualize USD files in SAPIEN"
    )
    parser.add_argument(
        "--usd_path", type=str, required=True, help="Path to the USD file"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Scale factor for the USD model"
    )
    parser.add_argument(
        "--show_collision", action="store_true", help="Show collision geometries"
    )
    args = parser.parse_args()

    if not os.path.exists(args.usd_path):
        print(f"USD file not found: {args.usd_path}")
        return

    # Create scene
    scene = create_scene()

    # Create ground plane
    create_ground_plane(scene)

    # Create coordinate axes
    create_coordinate_axes(scene)

    # Load USD file
    load_usd_file(scene, args.usd_path, args.scale, args.show_collision)

    # Create viewer
    viewer = Viewer(scene)
    viewer.set_camera_xyz(2, 2, 2)
    viewer.set_camera_rpy(0, -0.5, -0.8)

    # Print instructions
    print("\nControls:")
    print("  Mouse Left: Rotate camera")
    print("  Mouse Right: Pan camera")
    print("  Mouse Scroll: Zoom camera")
    print("  R: Reset camera")
    print("  Esc: Exit")

    # Run the viewer
    while not viewer.closed:
        scene.step()
        viewer.render()


if __name__ == "__main__":
    main()
