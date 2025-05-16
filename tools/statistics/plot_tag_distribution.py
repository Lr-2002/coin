import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# Define path to tag CSV files
tags_dir = "github_page/static/tags"

# Map file names to reasoning groups
group_mapping = {
    "tag_obj_distribution.csv": "Object-Centric Reasoning",
    "tag_rob_distribution.csv": "Robot-Centric Reasoning",
    "tag_iter_distribution.csv": "Interactive Reasoning"
}

# Read all CSV files and combine data
all_data = []

for csv_file in glob.glob(os.path.join(tags_dir, "*.csv")):
    filename = os.path.basename(csv_file)
    if filename in group_mapping:
        group = group_mapping[filename]
        df = pd.read_csv(csv_file)
        
        # Add group column
        df["Group"] = group
        
        # Select only the needed columns and rename
        df = df[["Tag", "Percentage", "Group"]]
        
        # Add to combined data
        all_data.append(df)

# Combine all dataframes
if all_data:
    df = pd.concat(all_data, ignore_index=True)
    # Sort by group and percentage
    df = df.sort_values(by=["Group", "Percentage"], ascending=[True, False])
else:
    print("No CSV files found in", tags_dir)
    exit(1)

# Make sure the Percentage column exists and is numeric
if 'Percentage' not in df.columns:
    # Check if there's a column that might contain percentage data
    if 'Count' in df.columns:
        # Calculate percentage from counts if available
        total_counts = df.groupby('Group')['Count'].transform('sum')
        df['Percentage'] = (df['Count'] / total_counts) * 100
    else:
        print("Error: Neither 'Percentage' nor 'Count' column found in CSV files")
        exit(1)

# Ensure Percentage is numeric
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')
df = df.dropna(subset=['Percentage'])

# Set colors for each reasoning group
colors = {
    "Object-Centric Reasoning": "#8FB339",
    "Robot-Centric Reasoning": "#3C8031",
    "Interactive Reasoning": "#BBD8A3",
}
bar_colors = [colors.get(g, "#333333") for g in df["Group"]]

# Create output directory if it doesn't exist
output_dir = "github_page/static/plots"
os.makedirs(output_dir, exist_ok=True)

# Create the plot
plt.figure(figsize=(12, 8))
bars = plt.barh(df["Tag"], df["Percentage"], color=bar_colors)

# Add legend (top right corner)
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
labels = list(colors.keys())
plt.legend(handles, labels, title="Reasoning Group", loc="upper right")

# Additional settings
plt.xlabel("Percentage (%)")
plt.title("Reasoning Tag Distribution by Group")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save with transparent background
plt.savefig(
    "github_page/static/plots/reasoning_tags_transparent.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True,
)

# Also save a version with white background for easier viewing
plt.savefig(
    "github_page/static/plots/reasoning_tags.png",
    dpi=300,
    bbox_inches="tight",
)

print(f"Plots saved to {output_dir}/reasoning_tags_transparent.png and {output_dir}/reasoning_tags.png")

# Display plot if running interactively
try:
    plt.show()
except:
    pass
