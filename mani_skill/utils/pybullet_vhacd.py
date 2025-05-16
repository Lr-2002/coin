import os
import re
import pybullet as p
def mergy_obj(obj_list, output_path):

    import bpy
    import os

    bpy.ops.wm.read_factory_settings(use_empty=True)

    for obj_file in obj_list:
        if os.path.exists(obj_file):
            bpy.ops.wm.obj_import(filepath=obj_file)
        else:
            print(f"{obj_file} not exist")

    bpy.ops.object.select_all(action='SELECT')

    bpy.ops.object.join()

    bpy.ops.wm.obj_export(filepath=output_path)
def change_mtl_path(old, new, mtl_path):
    text = None
    with open(mtl_path, 'r') as file:
        text = file.read()
        text = text.replace(old, new)
    file.close()
    with open(mtl_path, 'w') as file:
        file.write(text)
    file.close()
def calculate_convex(input_file, output_file):
    p.vhacd(
        input_file,
        output_file,
        'vhacd_log.txt',
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
        convexhullApproximation=1
    )
link_pattern = r'<link name=(.*?)</link>'
joint_pattern = r'<joint(.*?)</joint>'
robot_pattern = r'<robot name=(.*?)>'
xyz_pattern = r'<origin xyz=(.*?)/>'
mesh_pattern = r'<mesh filename=(.*?)/>'
visual_pattern = r'<visual name=(.*?)>'

if __name__ == '__main__':
    urdfs_folder = './partnet_mobility_part'
    for urdf_folder in os.listdir(urdfs_folder):
        if os.path.exists(os.path.join(os.path.join(urdfs_folder, urdf_folder), f"textured_objs/link_0_convex.obj")):
            print(f"skip {urdf_folder}")
            continue
        urdf_save_path = os.path.join(os.path.join(urdfs_folder, urdf_folder), f'{urdf_folder}.urdf')
        if not os.path.exists(os.path.join(os.path.join(urdfs_folder, urdf_folder), 'mobility.urdf')):
            continue
        with open(os.path.join(os.path.join(urdfs_folder, urdf_folder), 'mobility.urdf')) as urdf_file:
            urdf_text = urdf_file.read()

            robot_matches = re.findall(robot_pattern, urdf_text, re.DOTALL)
            robot_name = robot_matches[0]
            
            joint_list = []
            joint_matches = re.findall(joint_pattern, urdf_text, re.DOTALL)
            for match in joint_matches:
                joint_list.append(f'\t<joint{match}\t<dynamics damping=\"0.0002\" friction=\"0.01\"/>\n\t</joint>\n')

            link_list = []
            link_list.append("\t<link name=\"base\"/>")
            link_matches = re.findall(link_pattern, urdf_text, re.DOTALL)
            for id, match in enumerate(link_matches):
                visual_matches = re.findall(visual_pattern, match, re.DOTALL)
                visual_name = visual_matches[0]
                xyz_matches = re.findall(xyz_pattern, match, re.DOTALL)
                xyz = xyz_matches[0]
                link_list.append(f"\t<link name=\"link_{id}\">\n\t\t<visual name={visual_name}>\n\t\t\t<origin xyz={xyz}/>\n\t\t\t<geometry>\n\t\t\t\t\
<mesh filename=\"textured_objs/link_{id}.obj\"/>\n\t\t\t</geometry>\n\t\t</visual>\n\t\t<collision>\n\t\t\t<origin xyz={xyz}/>\n\t\t\t\
<geometry>\n\t\t\t\t<mesh filename=\"textured_objs/link_{id}_convex.obj\"/>\n\t\t\t</geometry>\n\t\t</collision>\n\t</link>")
                mesh_matches = re.findall(mesh_pattern, match, re.DOTALL)
                mesh_list = []
                for mesh in mesh_matches:
                    mesh_list.append(os.path.join(os.path.join(urdfs_folder, urdf_folder), mesh[1:-1]))
                mesh_output_path = os.path.join(os.path.join(urdfs_folder, urdf_folder), f"textured_objs/link_{id}.obj")
                mesh_mtl_path = os.path.join(os.path.join(urdfs_folder, urdf_folder), f"textured_objs/link_{id}.mtl")
                calculate_convex(mesh_output_path, mesh_output_path[:-4] + '_convex.obj')
                mergy_obj(mesh_list, mesh_output_path)
                change_mtl_path('partnet_mobility_part/' + urdf_folder, '..', mesh_mtl_path)
            with open(urdf_save_path, 'w') as write_file:
                write_file.write("<?xml version=\"1.0\" ?>\n")
                write_file.write(f"<robot name={robot_name}>\n")
                for link in link_list:
                    write_file.write(link + '\n')
                for joint in joint_list:
                    write_file.write(joint)
                write_file.write('</robot>')
            write_file.close()


