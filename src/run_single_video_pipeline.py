# src/run_single_video_pipeline.py

import os
import json
import pandas as pd
from dotenv import load_dotenv
import logging
import shutil # For cleaning up temp dir

# --- Import project modules ---
# This relies on the __init__.py files making 'src' a package
from src.persistence import s3_utils, mongo_utils
from src.processing import video, audio, text

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
log = logging.getLogger(__name__) # Logger for this script

# --- Load Environment Variables ---
# Find the .env file relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    log.info(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
else:
    log.warning(f".env file not found at {dotenv_path}. Relying on environment variables.")

# --- Get Configuration from Environment ---
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "video_analysis_db")

# --- Directories ---
TEMP_PROCESSING_DIR = os.path.join(project_root, "data", "temp_processing") # Single temp dir

# --- Main Processing Function ---
# Inside process_single_video function (Corrected)
def process_single_video(video_id, s3_key, s3_client, mongo_db):
    """
    Runs the full processing pipeline for a single video and saves results to MongoDB.
    """
    log.info(f"--- Starting processing for Video ID: {video_id} ---")

    # --- CORRECTED CHECK ---
    # 1. Check mongo_db explicitly first
    if mongo_db is None:
         log.error("Missing required argument: mongo_db object is None. Aborting.")
         return False
    # 2. Check other arguments (s3_client could also be None checked explicitly if desired)
    if not all([video_id, s3_key, s3_client]): # Removed mongo_db from all()
        log.error("Missing required arguments (video_id, s3_key, or s3_client). Aborting.")
        return False
    # --- END CORRECTED CHECK ---

    # Ensure temp directory exists and is clean for this video
    video_temp_dir = os.path.join(TEMP_PROCESSING_DIR, video_id)
    if os.path.exists(video_temp_dir):
        shutil.rmtree(video_temp_dir) # Remove old temp data if exists
    os.makedirs(video_temp_dir, exist_ok=True)

    local_video_path = None # Initialize path
    success_flag = True # Track overall success
    # ... rest of function ...

    try:
        # 1. Download Video
        local_video_path = s3_utils.download_s3_object(s3_client, S3_BUCKET_NAME, s3_key, video_temp_dir)
        if not local_video_path:
            log.error(f"Failed to download video {s3_key}. Skipping further processing for {video_id}.")
            return False # Cannot proceed without video

        # --- Save basic video info to Mongo ---
        # (Could fetch metadata here too if needed, or assume it's done elsewhere)
        video_doc = {'video_id': video_id, 's3_key': s3_key, 's3_bucket': S3_BUCKET_NAME}
        mongo_utils.save_video_metadata(mongo_db, video_doc)

        # 2. Detect Scenes
        log.info(f"Starting scene detection for {video_id}...")
        scene_timestamps = video.detect_scenes(local_video_path)
        # Save even if empty list
        if not mongo_utils.save_scene_data(mongo_db, video_id, scene_timestamps):
            log.warning(f"Failed to save scene data for {video_id} to MongoDB.")
            success_flag = False # Mark as partial failure

        # 3. Transcribe Audio
        log.info(f"Starting audio transcription for {video_id}...")
        transcription_result = audio.transcribe_audio(local_video_path) # Uses default 'tiny.en' model

        transcript_segments_with_sentiment = [] # Initialize list for combined data

        if transcription_result and "segments" in transcription_result:
            log.info(f"Transcription successful for {video_id}. Analyzing sentiment...")
            # 4. Analyze Transcript Sentiment (using Transformers)
            sentiment_pipeline = text.get_sentiment_pipeline() # Load/get cached pipeline
            if sentiment_pipeline:
                for i, segment in enumerate(transcription_result["segments"]):
                    segment_text = segment.get('text', '').strip()
                    sentiment = text.analyze_sentiment_transformer(segment_text, sentiment_pipeline)
                    transcript_segments_with_sentiment.append({
                        'start': segment.get('start'),
                        'end': segment.get('end'),
                        'text': segment_text,
                        'sentiment_label': sentiment.get('label'),
                        'sentiment_score': sentiment.get('score')
                    })
                    # Optional: Log progress
                    # if (i + 1) % 20 == 0:
                    #     log.info(f"Analyzed sentiment for {i+1} transcript segments...")
                log.info(f"Sentiment analysis complete for {len(transcript_segments_with_sentiment)} transcript segments.")

                # Save transcript+sentiment to Mongo
                if not mongo_utils.save_transcript_segments(mongo_db, video_id, transcript_segments_with_sentiment):
                     log.warning(f"Failed to save transcript data for {video_id} to MongoDB.")
                     success_flag = False
            else:
                log.error(f"Sentiment pipeline failed to load. Cannot analyze transcript sentiment for {video_id}.")
                # Save transcript without sentiment? Or mark as failed? For now, just log.
                success_flag = False
        else:
            log.warning(f"Transcription failed or produced no segments for {video_id}.")
            # Optionally save an empty list to indicate processing attempted but failed
            if not mongo_utils.save_transcript_segments(mongo_db, video_id, []):
                log.warning(f"Failed to save empty transcript marker for {video_id} to MongoDB.")
            success_flag = False # Mark as partial failure

        # 5. Analyze Comment Sentiment
        # (Assuming comments JSON already collected by collect_data.py)
        log.info(f"Loading and analyzing comments for {video_id}...")
        comments_path = os.path.join(project_root, "data", "comments", f"{video_id}_comments.json")
        comment_sentiments = []
        if os.path.exists(comments_path):
            try:
                with open(comments_path, 'r', encoding='utf-8') as f:
                    comments_data = json.load(f)

                sentiment_pipeline = text.get_sentiment_pipeline() # Ensure pipeline is loaded
                if sentiment_pipeline:
                    for i, comment in enumerate(comments_data):
                        comment_text = comment.get('text', '').strip()
                        sentiment = text.analyze_sentiment_transformer(comment_text, sentiment_pipeline)
                        comment_sentiments.append({
                            'comment_id': comment.get('comment_id'),
                            'author': comment.get('author'),
                            'published_at': comment.get('published_at'),
                            'text': comment_text,
                            'sentiment_label': sentiment.get('label'),
                            'sentiment_score': sentiment.get('score')
                            # Add other original comment fields if needed
                        })
                        # Optional: Log progress
                        # if (i + 1) % 10 == 0:
                        #      log.info(f"Analyzed sentiment for {i+1} comments...")
                    log.info(f"Sentiment analysis complete for {len(comment_sentiments)} comments.")

                    # Save comments+sentiment to Mongo
                    if not mongo_utils.save_comment_sentiments(mongo_db, video_id, comment_sentiments):
                        log.warning(f"Failed to save comment data for {video_id} to MongoDB.")
                        success_flag = False
                else:
                    log.error(f"Sentiment pipeline failed to load. Cannot analyze comment sentiment for {video_id}.")
                    success_flag = False

            except (json.JSONDecodeError, IOError) as e:
                log.error(f"Error reading or parsing comments file {comments_path}: {e}", exc_info=True)
                # Optionally save empty comments? For now, just log.
            except Exception as e:
                 log.error(f"Unexpected error processing comments for {video_id}: {e}", exc_info=True)

        else:
            log.warning(f"Comments file not found for {video_id} at {comments_path}. Skipping comment analysis.")
            # Optionally save empty comments? For now, just log.

        log.info(f"--- Finished processing for Video ID: {video_id} ---")
        return success_flag

    except Exception as e:
        log.error(f"An unexpected critical error occurred during processing for {video_id}: {e}", exc_info=True)
        return False # Indicate failure

    finally:
        # 6. Clean up temporary file ALWAYS
        if local_video_path and os.path.exists(local_video_path):
            try:
                # Remove the whole temp dir for this video
                shutil.rmtree(video_temp_dir)
                # log.info(f"Removed temporary directory: {video_temp_dir}")
            except OSError as e:
                log.warning(f"Could not remove temporary directory {video_temp_dir}: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    log.info("--- Starting Single Video Pipeline Test ---")

    # --- Initialization ---
    s3_client = s3_utils.get_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    mongo_client = mongo_utils.get_mongo_client(MONGO_CONNECTION_STRING)

    if not s3_client or not mongo_client:
        log.fatal("Failed to initialize S3 or MongoDB client. Exiting.")
        exit()

    mongo_db = mongo_utils.get_mongo_database(mongo_client, MONGO_DB_NAME)
    # Corrected check
    if mongo_db is None:
        log.fatal(f"Failed to get MongoDB database '{MONGO_DB_NAME}'. Exiting.")
        mongo_client.close() # Close client if db fails
        exit()

    # --- Select Video to Process ---
    # Use one of the VIDEO_IDs you downloaded with the *updated* collect_data.py
    TEST_VIDEO_ID = "bcGxg3c1HE8"  # <<< REPLACE WITH A VALID ID FROM YOUR LATEST RUN
    # Construct the expected S3 key (assuming collect_data saved as VIDEO_ID.mp4)
    TEST_S3_KEY = f"raw_videos/{TEST_VIDEO_ID}.mp4"

    # --- Execute Processing ---
    overall_success = process_single_video(TEST_VIDEO_ID, TEST_S3_KEY, s3_client, mongo_db)

    if overall_success:
        log.info(f"Processing for video {TEST_VIDEO_ID} completed successfully (or with warnings). Check logs and MongoDB.")
    else:
        log.error(f"Processing for video {TEST_VIDEO_ID} failed. Check logs for details.")

    # --- Cleanup ---
    if mongo_client:
        mongo_client.close()
        log.info("MongoDB connection closed.")

    log.info("--- Single Video Pipeline Test Finished ---")