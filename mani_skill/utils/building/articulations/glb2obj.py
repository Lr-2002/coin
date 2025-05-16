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


def main():
    if "--" not in sys.argv:
        print("Usage: blender -b -P script.py -- <input_path> <output_path>")
        return

    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 2:
        print("Please provide both input and output paths")
        return

    input_path = args[0]
    output_path = args[1]

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    success = convert_glb_to_obj(input_path, output_path)
    print("Conversion " + ("succeeded" if success else "failed"))


if __name__ == "__main__":
    main()
