import os
import cv2

# Directory containing videos
VIDEO_DIR = os.path.join(os.path.dirname(__file__), '../coin_videos/medias')

# Supported video extensions
VIDEO_EXTS = ['.mp4', '.avi', '.mov']

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def main():
    results = {}
    for fname in os.listdir(VIDEO_DIR):
        if any(fname.endswith(ext) for ext in VIDEO_EXTS):
            task_name = os.path.splitext(fname)[0]
            video_path = os.path.join(VIDEO_DIR, fname)
            length = get_video_length(video_path)
            results[task_name] = length
    # Save results
    with open('video_lengths.txt', 'w') as f:
        for k, v in results.items():
            f.write(f'{k}: {v}\n')
    print('Saved video lengths to video_lengths.txt')

if __name__ == '__main__':
    main()