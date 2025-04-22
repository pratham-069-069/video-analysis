# src/process_video.py

import cv2 # OpenCV for video processing
import numpy as np
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Assuming this script might eventually be run from the project root
# Or adjust the path if run differently. Let's load from parent for consistency.
load_dotenv('../.env')

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- S3 Setup ---
# (Could be refactored into a shared utility later)
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    print(f"S3 client initialized for region {AWS_REGION}.")
except Exception as e:
    print(f"Error initializing S3 client: {e}")
    exit()

# --- Configuration ---
# Directory for temporary video downloads
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
TEMP_VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos_temp_processing")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# Scene detection parameters (tune these later)
FRAME_DIFFERENCE_THRESHOLD = 0.4 # Percentage of pixels that need to change significantly
RESIZE_WIDTH = 320 # Resize frame for faster comparison (optional)


def download_s3_video(bucket, s3_key, local_dir):
    """Downloads a video from S3 to a local temporary directory."""
    local_filename = os.path.basename(s3_key)
    local_filepath = os.path.join(local_dir, local_filename)

    print(f"Attempting to download s3://{bucket}/{s3_key} to {local_filepath}...")
    try:
        s3_client.download_file(bucket, s3_key, local_filepath)
        print(f"Successfully downloaded video to {local_filepath}")
        return local_filepath
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(f"Error: S3 object s3://{bucket}/{s3_key} not found.")
        elif e.response['Error']['Code'] == "403":
            print(f"Error: Access Denied downloading s3://{bucket}/{s3_key}. Check permissions.")
        else:
            print(f"Error downloading video from S3: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during S3 download: {e}")
        return None

def detect_scenes(video_path, threshold=FRAME_DIFFERENCE_THRESHOLD, resize_width=RESIZE_WIDTH):
    """
    Opens a video file and detects scene changes using frame differencing.
    Returns a list of timestamps (in seconds) where scene changes are detected.
    """
    scene_change_timestamps = []
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return scene_change_timestamps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return scene_change_timestamps

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not determine video FPS. Timestamps might be inaccurate.")
        fps = 30 # Assume a default FPS

    prev_frame_gray = None
    frame_count = 0

    print(f"Processing video: {os.path.basename(video_path)} at {fps:.2f} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video frames.")
            break # End of video

        # --- Frame Processing for Scene Detection ---
        # 1. Resize for speed (optional)
        if resize_width and frame.shape[1] > resize_width:
            aspect_ratio = frame.shape[0] / frame.shape[1]
            new_height = int(resize_width * aspect_ratio)
            current_frame_resized = cv2.resize(frame, (resize_width, new_height))
        else:
            current_frame_resized = frame

        # 2. Convert to Grayscale
        current_frame_gray = cv2.cvtColor(current_frame_resized, cv2.COLOR_BGR2GRAY)

        # 3. Blur slightly to reduce noise
        current_frame_gray = cv2.GaussianBlur(current_frame_gray, (9, 9), 0)

        # 4. Compare with previous frame (if exists)
        if prev_frame_gray is not None:
            frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
            # Create a binary mask where differences are above a certain level
            _, diff_thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            # Calculate the percentage of changed pixels
            changed_pixels_ratio = np.sum(diff_thresh > 0) / diff_thresh.size

            if changed_pixels_ratio > threshold:
                timestamp_sec = frame_count / fps
                scene_change_timestamps.append(timestamp_sec)
                # print(f"  Scene change detected at frame {frame_count} ({timestamp_sec:.2f}s) - Diff Ratio: {changed_pixels_ratio:.3f}") # Optional: more verbose logging

        # Update previous frame
        prev_frame_gray = current_frame_gray
        frame_count += 1

        # Optional: Display processing progress
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")


    cap.release() # Release the video capture object
    print(f"Detected {len(scene_change_timestamps)} potential scene changes.")
    return scene_change_timestamps

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Video Scene Detection Script ---")

    # Example: Process one specific video from S3
    # You can get this key from the collect_data.py output or your S3 bucket
    example_s3_key = "raw_videos/bcGxg3c1HE8.f616.mp4"
     

    # 1. Download the video file temporarily
    local_video_path = download_s3_video(S3_BUCKET_NAME, example_s3_key, TEMP_VIDEO_DIR)

    if local_video_path:
        # 2. Perform scene detection
        detected_times = detect_scenes(local_video_path)

        if detected_times:
            print("\nScene change timestamps (seconds):")
            # Format output for better readability
            formatted_times = [f"{t:.2f}" for t in detected_times]
            print(", ".join(formatted_times))
        else:
            print("No scene changes detected or processing failed.")

        # 3. Clean up: Delete the temporary local video file
        try:
            os.remove(local_video_path)
            print(f"\nRemoved temporary file: {local_video_path}")
        except OSError as e:
            print(f"\nWarning: Could not remove temporary file {local_video_path}: {e}")
    else:
        print(f"Could not process video {example_s3_key} due to download failure.")

    print("\nScript finished.")