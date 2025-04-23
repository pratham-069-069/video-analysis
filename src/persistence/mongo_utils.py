# src/persistence/mongo_utils.py

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError
from datetime import datetime, timezone # Use timezone-aware UTC
import logging

# Configure basic logging if not configured by the main script yet
# This helps if these utils are imported elsewhere without explicit setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
log = logging.getLogger(__name__)

def get_mongo_client(connection_string):
    """Establishes connection to MongoDB and returns the client object."""
    if not connection_string:
        log.error("MongoDB connection string not provided.")
        return None
    try:
        # Add timeout options for robustness
        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=5000, # 5 sec server selection timeout
            connectTimeoutMS=10000,        # 10 sec connection timeout
            socketTimeoutMS=10000          # 10 sec socket timeout
            )
        # Use hello command for modern MongoDB versions
        client.admin.command('hello')
        log.info("Successfully connected to MongoDB.")
        return client
    except (ConnectionFailure, ConfigurationError) as e:
        log.error(f"MongoDB connection/configuration failed: {e}", exc_info=True)
        return None
    except Exception as e:
         log.error(f"An unexpected error occurred connecting to MongoDB: {e}", exc_info=True)
         return None

def get_mongo_database(client, db_name):
    """Gets a database object from the client."""
    if not client:
        log.error("Cannot get database: MongoDB client is not valid.")
        return None
    if not db_name:
        log.error("Cannot get database: Database name not provided.")
        return None
    try:
        db = client[db_name]
        return db
    except Exception as e:
        log.error(f"Error getting MongoDB database '{db_name}': {e}", exc_info=True)
        return None

def save_video_metadata(db, video_data):
    """Saves or updates video metadata in the 'videos' collection."""
    # --- CORRECTED CHECK ---
    if db is None:
        log.error("Cannot save video metadata: DB object invalid.")
        return False
    # --- END CORRECTED CHECK ---
    if not video_data or not isinstance(video_data, dict) or 'video_id' not in video_data:
        log.error(f"Invalid video_data provided to save_video_metadata: {video_data}")
        return False

    collection = db['videos']
    video_id = video_data['video_id']
    video_data['last_updated'] = datetime.now(timezone.utc) # Add/update timestamp

    try:
        result = collection.update_one(
            {'video_id': video_id},
            {'$set': video_data},
            upsert=True
        )
        # Optional: More detailed logging if needed for debugging
        # log.info(f"Video metadata for {video_id} saved/updated. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")
        return True
    except OperationFailure as e:
        log.error(f"MongoDB OperationFailure saving video metadata for {video_id}: {e.details}", exc_info=True)
        return False
    except Exception as e:
        log.error(f"Unexpected error saving video metadata for {video_id}: {e}", exc_info=True)
        return False

def save_scene_data(db, video_id, scene_timestamps):
    """Saves scene change timestamps for a video, replacing existing data."""
    # --- CORRECTED CHECK ---
    if db is None:
        log.error("Cannot save scene data: DB object invalid.")
        return False
    # --- END CORRECTED CHECK ---
    if not video_id:
        log.error("Cannot save scene data: video_id missing.")
        return False
    if not isinstance(scene_timestamps, list):
         log.error(f"Invalid scene_timestamps for {video_id}: Must be a list.")
         return False

    collection = db['scenes']
    doc = {
        'video_id': video_id,
        'scene_change_timestamps_sec': scene_timestamps,
        'last_updated': datetime.now(timezone.utc)
    }
    try:
        result = collection.replace_one({'video_id': video_id}, doc, upsert=True)
        # Optional: More detailed logging
        # log.info(f"Scene data for {video_id} saved/replaced. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")
        return True
    except OperationFailure as e:
        log.error(f"MongoDB OperationFailure saving scene data for {video_id}: {e.details}", exc_info=True)
        return False
    except Exception as e:
        log.error(f"Unexpected error saving scene data for {video_id}: {e}", exc_info=True)
        return False

def save_transcript_segments(db, video_id, transcript_segments):
    """Saves transcript segments (incl. sentiment) for a video. Deletes old before inserting."""
    # --- CORRECTED CHECK ---
    if db is None:
        log.error("Cannot save transcript: DB object invalid.")
        return False
    # --- END CORRECTED CHECK ---
    if not video_id:
        log.error("Cannot save transcript: video_id missing.")
        return False
    if not isinstance(transcript_segments, list):
        log.error(f"Invalid transcript_segments for {video_id}: Must be a list.")
        return False

    collection = db['transcripts']
    timestamp = datetime.now(timezone.utc)

    # Add video_id and timestamp to each segment before inserting
    processed_segments = []
    for i, segment in enumerate(transcript_segments):
        if isinstance(segment, dict):
             segment_doc = segment.copy() # Avoid modifying original list dicts
             segment_doc['video_id'] = video_id
             segment_doc['segment_index'] = i # Add an index for ordering
             segment_doc['last_updated'] = timestamp
             processed_segments.append(segment_doc)
        else:
             log.warning(f"Skipping invalid segment data for {video_id} at index {i}: {segment}")

    if not processed_segments and transcript_segments: # Check if processing failed, but input list was not empty
         log.error(f"No valid transcript segments processed for {video_id}, but input was not empty.")
         # Consider returning False here as something went wrong in processing before saving
         # return False # Or proceed to delete/insert empty

    try:
        # Delete existing segments for this video first to prevent duplicates
        del_result = collection.delete_many({'video_id': video_id})
        # Optional: log.info(f"Deleted {del_result.deleted_count} old transcript segments for {video_id}.")

        # Insert new segments if any exist
        if processed_segments:
            insert_result = collection.insert_many(processed_segments)
            # Optional: log.info(f"Inserted {len(insert_result.inserted_ids)} new transcript segments for {video_id}.")
            pass
        else:
            # Optional: log.info(f"No new transcript segments to insert for {video_id}.")
            pass

        return True # Return True even if no segments were inserted (e.g., empty transcript)
    except OperationFailure as e:
        log.error(f"MongoDB OperationFailure saving transcript for {video_id}: {e.details}", exc_info=True)
        return False
    except Exception as e:
        log.error(f"Unexpected error saving transcript for {video_id}: {e}", exc_info=True)
        return False

def save_comment_sentiments(db, video_id, comment_sentiments):
    """Saves comments with sentiment for a video. Deletes old before inserting."""
    # --- CORRECTED CHECK ---
    if db is None:
        log.error("Cannot save comments: DB object invalid.")
        return False
    # --- END CORRECTED CHECK ---
    if not video_id:
        log.error("Cannot save comments: video_id missing.")
        return False
    if not isinstance(comment_sentiments, list):
        log.error(f"Invalid comment_sentiments for {video_id}: Must be a list.")
        return False

    collection = db['comments'] # Use a dedicated collection for comments with sentiment
    timestamp = datetime.now(timezone.utc)

    processed_comments = []
    for i, comment_data in enumerate(comment_sentiments):
         if isinstance(comment_data, dict):
             comment_doc = comment_data.copy()
             comment_doc['video_id'] = video_id
             comment_doc['last_updated'] = timestamp
             processed_comments.append(comment_doc)
         else:
             log.warning(f"Skipping invalid comment data for {video_id} at index {i}: {comment_data}")

    if not processed_comments and comment_sentiments: # Check if processing failed but input list was not empty
        log.error(f"No valid comments processed for {video_id}, but input was not empty.")
        # Consider returning False here
        # return False

    try:
        # Delete existing comments for this video in this collection
        del_result = collection.delete_many({'video_id': video_id})
        # Optional: log.info(f"Deleted {del_result.deleted_count} old comments for {video_id}.")

        if processed_comments:
            insert_result = collection.insert_many(processed_comments)
            # Optional: log.info(f"Inserted {len(insert_result.inserted_ids)} new comments for {video_id}.")
            pass
        else:
            # Optional: log.info(f"No new comments to insert for {video_id}.")
            pass

        return True # Return True even if no comments were inserted
    except OperationFailure as e:
        log.error(f"MongoDB OperationFailure saving comments for {video_id}: {e.details}", exc_info=True)
        return False
    except Exception as e:
        log.error(f"Unexpected error saving comments for {video_id}: {e}", exc_info=True)
        return False