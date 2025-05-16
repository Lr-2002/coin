import os
import glob
from re import sub

# Define the base directory where the urdf files are located
base_dir = "./"

# Check if the base directory exists
if not os.path.exists(base_dir):
    print(f"Error: The directory {base_dir} does not exist.")
    exit(1)

# Find all directories in the base directory (e.g., hanoi_biggest, hanoi_small, etc.)
subdirs = [
    d
    for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and not d.isdigit()
]
print(subdirs)

for subdir in subdirs:
    # Construct the path to the subdirectory (e.g., hanoi_biggest)
    subdir_path = os.path.join(base_dir, subdir)

    # Find the .urdf file in the subdirectory
    urdf_files = glob.glob(os.path.join(subdir_path, "*.urdf"))
    if not urdf_files:
        print(f"No .urdf file found in {subdir_path}. Skipping...")
        continue

    # Assuming there's only one .urdf file per directory
    urdf_file = urdf_files[0]
    print(f"Processing {urdf_file}...")

    # Extract the directory name to use as the replacement (e.g., hanoi_biggest)
    xxx = subdir  # The directory name is used as 'xxx'

    # Define the replacement names
    link_replacement = xxx  # e.g., hanoi_biggest
    base_replacement = f"{xxx}_base"  # e.g., hanoi_biggest_base

    # Read the content of the .urdf file
    with open(urdf_file, "r") as file:
        content = file.read()

    # Perform the replacements
    updated_content = content.replace("link_0", link_replacement)
    updated_content = updated_content.replace("base", base_replacement)

    # Write the updated content back to the file
    with open(urdf_file, "w") as file:
        file.write(updated_content)

    print(f"Successfully updated {urdf_file}:")
    print(f"  - Replaced 'link_0' with '{link_replacement}'")
    print(f"  - Replaced 'base' with '{base_replacement}'")
