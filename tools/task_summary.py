#!/usr/bin/env python
import pickle as pkl
import json
import csv
import os

def load_pkl_path(pkl_path):
    """Load pickle file and return list of keys"""
    with open(pkl_path, 'rb') as f:
        pkl_list = pkl.load(f)
        pkl_list = [x for x in pkl_list.keys()]
    return pkl_list, pkl.load(open(pkl_path, 'rb'))

def load_json_file(json_path):
    """Load JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def generate_task_summary():
    # Define file paths
    interactive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/interactive_instruction_objects.pkl"
    primitive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/primitive_instruction_objects.pkl"
    workflows_path = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_workflows.json"
    tags_path = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_extended_tags.json"
    vqa_path = "/home/wangxianhao/data/project/reasoning/ManiSkill/env_ins_objects.json"
    
    # Output CSV path
    output_csv = "/home/wangxianhao/data/project/reasoning/ManiSkill/task_summary.csv"
    
    # Load data
    interactive_envs, interactive_data = load_pkl_path(interactive_path)
    primitive_envs, primitive_data = load_pkl_path(primitive_path)
    workflows = load_json_file(workflows_path)
    tags = load_json_file(tags_path)
    vqa_data = load_json_file(vqa_path)
    
    # Create CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['env_id', 'task_type', 'instruction', 'workflow', 'obj_tags', 
                     'rob_tags', 'iter_tags', 'query', 'query_selections', 'answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process interactive tasks
        for env_id in interactive_envs:
            row = {'env_id': env_id, 'task_type': 'interactive'}
            
            # Get instruction and objects
            if env_id in interactive_data:
                row['instruction'] = interactive_data[env_id].get('ins', '')
            
            # Get workflow
            if env_id in workflows:
                row['workflow'] = json.dumps(workflows[env_id])
            else:
                row['workflow'] = ''
            
            # Get tags
            if env_id in tags:
                row['obj_tags'] = json.dumps(tags[env_id].get('obj', []))
                row['rob_tags'] = json.dumps(tags[env_id].get('rob', []))
                row['iter_tags'] = json.dumps(tags[env_id].get('iter', []))
            else:
                row['obj_tags'] = ''
                row['rob_tags'] = ''
                row['iter_tags'] = ''
            
            # Get VQA data
            if env_id in vqa_data:
                query_data = vqa_data[env_id].get('query', {})
                if query_data:
                    row['query'] = query_data.get('query', '')
                    row['query_selections'] = json.dumps(query_data.get('selection', {}))
                    row['answer'] = vqa_data[env_id].get('answer', '')
                else:
                    row['query'] = ''
                    row['query_selections'] = ''
                    row['answer'] = ''
            else:
                row['query'] = ''
                row['query_selections'] = ''
                row['answer'] = ''
            
            writer.writerow(row)
        
        # Process primitive tasks
        for env_id in primitive_envs:
            row = {'env_id': env_id, 'task_type': 'primitive'}
            
            # Get instruction and objects
            if env_id in primitive_data:
                row['instruction'] = primitive_data[env_id].get('ins', '')
            
            # Leave other fields empty for primitive tasks
            row['workflow'] = ''
            row['obj_tags'] = ''
            row['rob_tags'] = ''
            row['iter_tags'] = ''
            row['query'] = ''
            row['query_selections'] = ''
            row['answer'] = ''
            
            # # Check if primitive task has VQA data (though unlikely)
            # if env_id in vqa_data:
            #     query_data = vqa_data[env_id].get('query', {})
            #     if query_data:
            #         row['query'] = query_data.get('query', '')
            #         row['query_selections'] = json.dumps(query_data.get('selection', {}))
            #         row['answer'] = vqa_data[env_id].get('answer', '')
            
            writer.writerow(row)
    
    print(f"CSV file generated: {output_csv}")

def get_env_demo_image():
    """Extract the 15th frame from videos, crop the second view from the right, and save as images"""
    import cv2
    import os
    import pickle as pkl
    
    # Define paths
    video_dir = "/home/wangxianhao/data/project/reasoning/ManiSkill/coin_videos/medias"
    output_dir = "/home/wangxianhao/data/project/reasoning/ManiSkill/task_demo_image"
    interactive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/interactive_instruction_objects.pkl"
    primitive_path = "/media/raid/workspace/wangxianhao/project/reasoning/ManiSkill/primitive_instruction_objects.pkl"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load environment IDs from pickle files
    all_env_ids = set()
    try:
        with open(interactive_path, 'rb') as f:
            interactive_data = pkl.load(f)
            for env_id in interactive_data.keys():
                all_env_ids.add(env_id)
    except Exception as e:
        print(f"Error loading interactive environments: {str(e)}")
    
    try:
        with open(primitive_path, 'rb') as f:
            primitive_data = pkl.load(f)
            for env_id in primitive_data.keys():
                all_env_ids.add(env_id)
    except Exception as e:
        print(f"Error loading primitive environments: {str(e)}")
    
    print(f"Found {len(all_env_ids)} total environments in pickle files")
    
    # Get all mp4 files in the directory
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"Found {len(videos)} videos to process")
    
    # Track which videos we've processed
    processed_env_ids = set()
    
    for video_file in videos:
        video_path = os.path.join(video_dir, video_file)
        
        # Extract environment ID from filename (remove extension)
        env_id = os.path.splitext(video_file)[0].replace('_', '-')
        output_path = os.path.join(output_dir, f"{env_id}.png")
        processed_env_ids.add(env_id)
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue
            
            # Calculate FPS and get total frame count
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Processing {video_file}: {total_frames} frames, FPS: {fps}")
            
            # Set frame position to 15th frame (0-based index, so 14)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 14)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read 15th frame from {video_file}")
                cap.release()
                continue
            
            # Crop the frame to get the second view from the right
            height, width = frame.shape[:2]
            shelf_related_envs = {
                "Tabletop-Pick-Book-FromShelf-v1",
                "Tabletop-Find-Book-Black-v1",
                "Tabletop-Find-Book-FromShelf-v1"
            }
            # Assuming there are 4 equal-width views side by side
            view_width = width // 5

            if env_id in shelf_related_envs:
                cropped_frame = frame[:, 0:view_width]
            # Extract the second view from right (index 2 from left, or 1 from right)
            else:
                cropped_frame = frame[:, view_width*3+3:view_width*4]
            
            # Save the cropped frame as an image
            cv2.imwrite(output_path, cropped_frame)
            print(f"Saved cropped frame from {video_file} to {output_path}")
            
            # Release the video capture object
            cap.release()
            
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
    
    # Check for environments without videos
    missing_videos = all_env_ids - processed_env_ids
    if missing_videos:
        print("\nThe following environments do not have corresponding videos:")
        for env_id in sorted(missing_videos):
            print(f"  - {env_id}")
    else:
        print("\nAll environments have corresponding videos.")
    
    print("\nFinished extracting frames from all videos")
    
if __name__ == "__main__":
    # generate_task_summary()
    get_env_demo_image()