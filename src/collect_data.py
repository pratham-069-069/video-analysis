# src/collect_data.py

import os
import json
import subprocess # To run yt-dlp
import pandas as pd
import numpy as np
import isodate # For parsing duration
import googleapiclient.discovery
import googleapiclient.errors
import boto3 # AWS SDK
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import shutil # For finding yt-dlp executable

# --- Load Environment Variables ---
load_dotenv('../.env') # Load from .env file in the parent directory

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- Basic Configuration ---
# Replace with your NEW chosen video IDs
VIDEO_IDS = ["ZdU3rWin0EQ", "mheHbVev1CU", "ul2HKsUpeH4", "bcGxg3c1HE8", "9wUg2xK5Yxo"]

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
COMMENTS_DIR = os.path.join(DATA_DIR, 'comments')
ENGAGEMENT_DIR = os.path.join(DATA_DIR, 'engagement')
VIDEO_TEMP_DIR = os.path.join(DATA_DIR, 'videos') # Temp storage for downloads

os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(COMMENTS_DIR, exist_ok=True)
os.makedirs(ENGAGEMENT_DIR, exist_ok=True)
os.makedirs(VIDEO_TEMP_DIR, exist_ok=True)

# --- YouTube API Setup ---
try:
    api_service_name = "youtube"
    api_version = "v3"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=YOUTUBE_API_KEY)
except Exception as e:
    print(f"Error initializing YouTube API: {e}")
    print("Ensure your YOUTUBE_API_KEY is correct in the .env file.")
    exit() # Exit if API can't be setup

# --- AWS S3 Setup ---
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
    print(f"Successfully connected to S3 bucket: {S3_BUCKET_NAME}")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        print(f"Error: S3 bucket '{S3_BUCKET_NAME}' not found. Please create it in region '{AWS_REGION}'.")
    elif error_code == '403':
        print(f"Error: Access denied to S3 bucket '{S3_BUCKET_NAME}'. Check IAM user permissions.")
    else:
        print(f"Error connecting to S3 bucket '{S3_BUCKET_NAME}': {e}")
    print("Ensure AWS credentials and bucket name/region in .env are correct and the IAM user has S3 permissions.")
    exit() # Exit if S3 setup fails
except Exception as e:
    print(f"An unexpected error occurred during S3 setup: {e}")
    exit()

# --- Helper Functions ---

def get_video_metadata(video_id):
    # (No changes needed from previous version)
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        if response['items']:
            return response['items'][0]
        else:
            print(f"No metadata found for video ID: {video_id}")
            return None
    except googleapiclient.errors.HttpError as e:
        print(f"An API error occurred for metadata {video_id}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching metadata for {video_id}: {e}")
        return None


def get_video_comments(video_id, max_results=50, order_by="relevance"): # Default to relevance
    # (Added order_by parameter, default to relevance)
    comments = []
    print(f"Fetching {order_by} comments for {video_id}...")
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100), # API max is 100 per page
            textFormat="plainText",
            order=order_by # Use parameter here
        )
        response = request.execute()

        while response and len(comments) < max_results:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'comment_id': item['id'],
                    'author': comment['authorDisplayName'],
                    'published_at': comment['publishedAt'],
                    'updated_at': comment['updatedAt'],
                    'text': comment['textDisplay'],
                    'like_count': comment['likeCount'],
                    'total_reply_count': item['snippet']['totalReplyCount']
                })

            # Check if there's a next page and if we need more results
            if 'nextPageToken' in response and len(comments) < max_results:
                next_page_token = response['nextPageToken']
                request = youtube.commentThreads().list(
                     part="snippet",
                     videoId=video_id,
                     maxResults=min(max_results - len(comments), 100),
                     pageToken=next_page_token,
                     textFormat="plainText",
                     order=order_by # Use parameter here too
                )
                response = request.execute()
            else:
                break # No more pages or max_results reached
        return comments
    except googleapiclient.errors.HttpError as e:
        if 'disabled comments' in str(e).lower():
             print(f"Comments are disabled for video ID: {video_id}")
        elif 'forbidden' in str(e).lower():
             print(f"API request forbidden for comments {video_id}. Check API key permissions/quotas.")
        else:
             print(f"An API error occurred for comments {video_id}: {e}")
        return [] # Return empty list if comments disabled or error
    except Exception as e:
        print(f"An unexpected error occurred fetching comments for {video_id}: {e}")
        return []

def download_video_local(video_id, output_dir):
    """
    Downloads video using yt-dlp, aiming for best mp4 video + best m4a audio,
    merges them into mp4, and renames the final file to video_id.mp4.
    Returns the path to the final video_id.mp4 file or None if failed.
    """
    # Check if yt-dlp exists
    ytdlp_executable = shutil.which("yt-dlp")
    if not ytdlp_executable:
        print("\nERROR: yt-dlp command not found.")
        print("Please install yt-dlp using pip:")
        print("  pip install yt-dlp")
        print("Or ensure it's in your system's PATH.\n")
        return None

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    # Temporary output template that yt-dlp uses during download/merge
    temp_output_template = os.path.join(output_dir, f"{video_id}_temp.%(ext)s")
    # Desired final path after successful download and rename
    desired_final_path = os.path.join(output_dir, f"{video_id}.mp4")

    # Format specifier: Best MP4 video + Best M4A audio, fallback to best MP4 overall
    format_specifier = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]/best[ext=mp4]/best'

    command = [
        ytdlp_executable,
        '-f', format_specifier,
        '-o', temp_output_template,      # Use temporary template
        '--merge-output-format', 'mp4',  # Force merge to mp4 container
        '--quiet',                       # Suppress yt-dlp progress output
        '--no-warnings',                 # Suppress yt-dlp warnings
        '--no-abort-on-error',           # Try to continue if some formats fail
        video_url
    ]

    print(f"Attempting to download+merge: {video_url}...")
    actual_downloaded_path = None # To track the file yt-dlp actually creates

    try:
        # Remove existing temp/final files for this ID to ensure clean download
        for f in os.listdir(output_dir):
            if f.startswith(f"{video_id}_temp.") or f == f"{video_id}.mp4":
                try:
                    os.remove(os.path.join(output_dir, f))
                    # print(f"Removed existing file: {f}") # Optional debug
                except OSError:
                    pass # Ignore if removal fails

        # Run yt-dlp process
        process = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8') # Added encoding

        # --- Find the successfully downloaded/merged file ---
        # yt-dlp might name the final merged file based on the template,
        # but it's safer to look for the expected final name directly if merge was forced.
        if os.path.exists(desired_final_path):
            actual_downloaded_path = desired_final_path
            print(f"Successfully downloaded and merged to: {os.path.basename(desired_final_path)}")
            return actual_downloaded_path # Return the clean, final path
        else:
             # Fallback: check if a temp file exists (maybe merge failed?)
             found_temp = None
             for f in os.listdir(output_dir):
                 if f.startswith(f"{video_id}_temp."):
                      found_temp = os.path.join(output_dir, f)
                      break
             if found_temp:
                 print(f"Warning: Merge might have failed. Found temp file: {os.path.basename(found_temp)}")
                 print("Attempting to rename temp file...")
                 try:
                     os.rename(found_temp, desired_final_path)
                     print(f"Successfully renamed temp file to {os.path.basename(desired_final_path)}")
                     return desired_final_path
                 except OSError as e:
                     print(f"Error renaming temp file: {e}. Download failed.")
                     return None
             else:
                 print(f"Error: Download finished but final file '{os.path.basename(desired_final_path)}' not found.")
                 return None

    except subprocess.CalledProcessError as e:
        print(f"Error executing yt-dlp for {video_id}:")
        # Limit printing potentially long stderr/stdout unless needed
        print(f"Return code: {e.returncode}")
        if e.stderr: print(f"Stderr: {e.stderr[:500]}...") # Print first 500 chars
        if e.stdout: print(f"Stdout: {e.stdout[:500]}...")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during video download for {video_id}: {e}")
        return None


def upload_to_s3(local_file_path, bucket, s3_key):
    # (No changes needed from previous version)
    if not local_file_path or not os.path.exists(local_file_path):
         print(f"Skipping S3 upload for {s3_key}, local file missing at '{local_file_path}'.")
         return None

    print(f"Uploading {os.path.basename(local_file_path)} to s3://{bucket}/{s3_key}...")
    try:
        s3_client.upload_file(local_file_path, bucket, s3_key)
        print(f"Successfully uploaded to s3://{bucket}/{s3_key}")
        s3_url = f"s3://{bucket}/{s3_key}"
        return s3_url
    except ClientError as e:
        print(f"Error uploading {local_file_path} to S3: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during S3 upload for {local_file_path}: {e}")
        return None


def create_synthetic_engagement(video_id, duration_iso):
    # (No changes needed from previous version)
    try:
        duration_sec = isodate.parse_duration(duration_iso).total_seconds()
    except Exception as e:
        print(f"Could not parse duration '{duration_iso}' for {video_id}. Using default 180s. Error: {e}")
        duration_sec = 180

    if duration_sec <= 0:
        print(f"Warning: Duration for {video_id} is zero or negative ({duration_sec}s). Generating empty engagement data.")
        return pd.DataFrame(columns=['video_id', 'timestamp_sec', 'cumulative_likes', 'cumulative_views'])

    num_events = max(10, min(int(duration_sec / 5), 100))
    timestamps = np.sort(np.random.uniform(0, duration_sec, num_events))
    likes = np.random.randint(0, 3, size=num_events).cumsum()
    views = np.random.randint(1, 10, size=num_events).cumsum()

    engagement_data = pd.DataFrame({
        'video_id': video_id,
        'timestamp_sec': timestamps,
        'cumulative_likes': likes,
        'cumulative_views': views
    })
    return engagement_data


# --- Main Execution Logic ---
all_engagement_data = []
processed_videos = {} # Keep track of metadata and S3 URLs

# Check if yt-dlp exists before starting the loop
if not shutil.which("yt-dlp"):
    print("\nFATAL ERROR: yt-dlp command not found. Cannot download videos.")
    print("Please install yt-dlp using pip:")
    print("  pip install yt-dlp")
    print("Or ensure it's in your system's PATH.\n")
    exit()

for video_id in VIDEO_IDS:
    print(f"\nProcessing Video ID: {video_id}")
    processed_videos[video_id] = {'s3_url': None, 'metadata_path': None, 'comments_path': None, 'engagement_generated': False}

    # 1. Get Metadata
    metadata = get_video_metadata(video_id)
    if metadata:
        metadata_path = os.path.join(METADATA_DIR, f"{video_id}_metadata.json")
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f: # Added encoding
                json.dump(metadata, f, indent=4, ensure_ascii=False) # Added ensure_ascii=False
            print(f"  - Metadata saved to {metadata_path}")
            processed_videos[video_id]['metadata_path'] = metadata_path
        except IOError as e:
            print(f"  - Error saving metadata for {video_id}: {e}")

        # 2. Get Comments
        comments = get_video_comments(video_id, max_results=50, order_by="relevance") # Get top 50 comments
        if comments:
            comments_path = os.path.join(COMMENTS_DIR, f"{video_id}_comments.json")
            try:
                with open(comments_path, 'w', encoding='utf-8') as f: # Added encoding
                    json.dump(comments, f, indent=4, ensure_ascii=False) # Added ensure_ascii=False
                print(f"  - {len(comments)} comments saved to {comments_path}")
                processed_videos[video_id]['comments_path'] = comments_path
            except IOError as e:
                 print(f"  - Error saving comments for {video_id}: {e}")
        else:
            print(f"  - No comments found or retrieved for {video_id}")

        # 3. Download Video Locally (includes merge and rename)
        local_video_path = download_video_local(video_id, VIDEO_TEMP_DIR) # Should return path like ".../video_id.mp4"

        # 4. Upload to S3
        if local_video_path:
            # Use the clean filename for the S3 key
            s3_object_key = f"raw_videos/{os.path.basename(local_video_path)}" # e.g., raw_videos/VIDEO_ID.mp4
            s3_url = upload_to_s3(local_video_path, S3_BUCKET_NAME, s3_object_key)
            if s3_url:
                processed_videos[video_id]['s3_url'] = s3_url
            # Delete local video file after successful upload
            try:
                os.remove(local_video_path)
                print(f"  - Removed local file: {local_video_path}")
            except OSError as e:
                print(f"  - Warning: Could not remove local file {local_video_path}: {e}")
        else:
            print(f"  - Skipping S3 upload for {video_id} due to download failure.")


        # 5. Create Synthetic Engagement Data (using metadata if available)
        if 'contentDetails' in metadata and 'duration' in metadata['contentDetails']:
             duration_iso = metadata['contentDetails']['duration']
             engagement_df = create_synthetic_engagement(video_id, duration_iso)
             if not engagement_df.empty:
                 all_engagement_data.append(engagement_df)
                 print(f"  - Generated synthetic engagement data ({len(engagement_df)} events).")
                 processed_videos[video_id]['engagement_generated'] = True
             else:
                 print(f"  - Generated empty engagement data for {video_id}.")

        else:
            print(f"  - Could not generate engagement data due to missing duration in metadata.")

    else:
        print(f"  - Skipping further processing for {video_id} due to metadata fetch error.")


# Combine and save all engagement data
if all_engagement_data:
    final_engagement_df = pd.concat(all_engagement_data, ignore_index=True)
    engagement_csv_path = os.path.join(ENGAGEMENT_DIR, "synthetic_engagement.csv")
    try:
        final_engagement_df.to_csv(engagement_csv_path, index=False, encoding='utf-8') # Added encoding
        print(f"\nSynthetic engagement data for all videos saved to {engagement_csv_path}")
    except IOError as e:
        print(f"\nError saving combined engagement data to {engagement_csv_path}: {e}")
else:
    print("\nNo engagement data was generated.")

print("\n--- Data Collection Summary ---")
for vid, status in processed_videos.items():
    print(f"Video ID: {vid}")
    print(f"  Metadata: {'Saved' if status['metadata_path'] else 'Failed/Skipped'}")
    print(f"  Comments: {'Saved' if status['comments_path'] else 'Failed/Skipped/None'}")
    print(f"  S3 Upload: {status['s3_url'] if status['s3_url'] else 'Failed/Skipped'}") # URL should now end in .mp4
    print(f"  Engagement: {'Generated' if status['engagement_generated'] else 'Failed/Skipped'}")

print("\nPhase 1: Data collection and S3 upload script finished.")

