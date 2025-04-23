# src/persistence/s3_utils.py

import os
import boto3
from botocore.exceptions import ClientError
import logging # Use logging instead of just print

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def get_s3_client(aws_access_key_id, aws_secret_access_key, region_name):
    """Initializes and returns an S3 client."""
    if not all([aws_access_key_id, aws_secret_access_key, region_name]):
        log.error("AWS credentials or region not fully provided.")
        return None
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        log.info(f"S3 client initialized for region {region_name}.")
        # Optional: Add a check like head_bucket here if needed, but might be better in main script
        return s3_client
    except Exception as e:
        log.error(f"Error initializing S3 client: {e}", exc_info=True)
        return None

def download_s3_object(s3_client, bucket, s3_key, local_dir):
    """Downloads an object from S3 to a local temporary directory."""
    if not s3_client:
        log.error("Cannot download S3 object: S3 client not initialized.")
        return None
    if not bucket or not s3_key:
        log.error("Cannot download S3 object: Bucket name or S3 key is missing.")
        return None

    local_filename = os.path.basename(s3_key)
    local_filepath = os.path.join(local_dir, local_filename)
    # Ensure local directory exists before attempting download
    try:
        os.makedirs(local_dir, exist_ok=True)
    except OSError as e:
        log.error(f"Error creating local directory {local_dir}: {e}", exc_info=True)
        return None

    log.info(f"Attempting to download s3://{bucket}/{s3_key} to {local_filepath}...")
    try:
        s3_client.download_file(bucket, s3_key, local_filepath)
        log.info(f"Successfully downloaded object to {local_filepath}")
        return local_filepath
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == "404":
            log.error(f"S3 object s3://{bucket}/{s3_key} not found.")
        elif error_code == "403":
            log.error(f"Access Denied downloading s3://{bucket}/{s3_key}. Check permissions.")
        else:
            log.error(f"ClientError downloading object from S3: {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Unexpected error during S3 download: {e}", exc_info=True)
        return None

def upload_to_s3(s3_client, local_file_path, bucket, s3_key):
    """Uploads a file to an S3 bucket."""
    if not s3_client:
        log.error("Cannot upload to S3: S3 client not initialized.")
        return None
    if not bucket or not s3_key:
        log.error("Cannot upload to S3: Bucket name or S3 key is missing.")
        return None
    if not local_file_path or not os.path.exists(local_file_path):
         log.warning(f"Skipping S3 upload for {s3_key}, local file missing at '{local_file_path}'.")
         return None

    log.info(f"Uploading {os.path.basename(local_file_path)} to s3://{bucket}/{s3_key}...")
    try:
        s3_client.upload_file(local_file_path, bucket, s3_key)
        s3_url = f"s3://{bucket}/{s3_key}"
        log.info(f"Successfully uploaded to {s3_url}")
        return s3_url
    except ClientError as e:
        log.error(f"ClientError uploading {local_file_path} to S3: {e}", exc_info=True)
        return None
    except Exception as e:
        log.error(f"Unexpected error during S3 upload for {local_file_path}: {e}", exc_info=True)
        return None