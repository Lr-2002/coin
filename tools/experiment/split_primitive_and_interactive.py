import pickle as pkl
import pandas as pd
import os

# Load primitive task names from pickle file
with open('primitive_instruction_objects.pkl', 'rb') as f:
    pri_dict = pkl.load(f)
    primitive_tasks = set(pri_dict.keys())

# Load the merged success rates CSV
merged_df = pd.read_csv('evaluation_results/merged_success_rates.csv')

# Create output directory if it doesn't exist
output_dir = 'evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Split into primitive and interactive tasks
primitive_df = merged_df[merged_df['Task'].isin(primitive_tasks)]
interactive_df = merged_df[~merged_df['Task'].isin(primitive_tasks)]

# Save the split dataframes to CSV files
primitive_df.to_csv(os.path.join(output_dir, 'primitive_success_rates.csv'), index=False)
interactive_df.to_csv(os.path.join(output_dir, 'interactive_success_rates.csv'), index=False)

print(f"Split complete!")
print(f"Primitive tasks: {len(primitive_df)} tasks saved to {os.path.join(output_dir, 'primitive_success_rates.csv')}")
print(f"Interactive tasks: {len(interactive_df)} tasks saved to {os.path.join(output_dir, 'interactive_success_rates.csv')}")
