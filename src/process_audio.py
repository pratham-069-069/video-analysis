# src/process_audio.py

import whisper # OpenAI Whisper
import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import time
import warnings

# Suppress specific warnings from Whisper if needed
warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)


# --- Load Environment Variables ---
load_dotenv('../.env')

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- S3 Setup ---
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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio_temp_processing") # Temp dir for videos
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Whisper Model Selection:
# Options: "tiny.en", "tiny", "base.en", "base", "small.en", "small",
#          "medium.en", "medium", "large-v1", "large-v2", "large-v3"
# Smaller models are faster, less accurate, use less RAM/VRAM.
# ".en" models are optimized for English.
# Start with a smaller model for faster testing.
WHISPER_MODEL = "tiny.en"
# WHISPER_MODEL = "base.en" # Good balance
# WHISPER_MODEL = "small.en" # More accurate, slower

# --- Helper Functions ---

def download_s3_video(bucket, s3_key, local_dir):
    """Downloads a video from S3 to a local temporary directory."""
    # (Identical to the function in process_video.py - consider refactoring later)
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

def transcribe_audio(video_path, model_name=WHISPER_MODEL):
    """
    Transcribes the audio from a video file using Whisper.
    Returns the structured transcription result.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    print(f"Loading Whisper model '{model_name}'...")
    # Loading the model can take time, especially the first time
    # or for larger models. It might also require internet access
    # the first time to download the model weights.
    try:
        model = whisper.load_model(model_name)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Ensure the model name is correct and you have internet access if downloading.")
        return None

    print(f"Starting transcription for: {os.path.basename(video_path)}...")
    start_time = time.time()
    try:
        # Whisper can often process video files directly using ffmpeg
        # Set fp16=False if you don't have a compatible GPU or encounter issues
        result = model.transcribe(video_path, fp16=False, verbose=False) # Set verbose=True for more progress output
        end_time = time.time()
        print(f"Transcription finished in {end_time - start_time:.2f} seconds.")
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        print("Ensure ffmpeg is installed correctly and accessible in your PATH.")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Video Audio Transcription Script ---")

    # Example: Process one specific video from S3
    # Use one of the NEW video IDs you collected (and ensure it's in S3)
    # Make sure the filename matches what's in S3 (e.g., VIDEO_ID.mp4 if renamed)
    example_s3_key = "raw_videos/bcGxg3c1HE8.mp4" 

    if "YOUR_NEW_VIDEO_ID" in example_s3_key:
         print("\nERROR: Please update 'example_s3_key' in the script with a real S3 key for one of your videos!\n")
         exit()

    # 1. Download the video file temporarily
    local_video_path = download_s3_video(S3_BUCKET_NAME, example_s3_key, TEMP_AUDIO_DIR)

    if local_video_path:
        # 2. Perform transcription
        transcription_result = transcribe_audio(local_video_path)

        if transcription_result:
            print("\n--- Transcription Results ---")
            # Print the full text
            print("\nFull Text:")
            print(transcription_result["text"])

            # Print segments with timestamps
            print("\nSegments:")
            for segment in transcription_result["segments"]:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                print(f"[{start:.2f}s -> {end:.2f}s] {text}")
        else:
            print("Transcription failed.")

        # # 3. Clean up: Delete the temporary local video file
        # try:
        #     os.remove(local_video_path)
        #     print(f"\nRemoved temporary file: {local_video_path}")
        # except OSError as e:
        #     print(f"\nWarning: Could not remove temporary file {local_video_path}: {e}")
    else:
        print(f"Could not process video {example_s3_key} due to download failure.")

    print("\nScript finished.")