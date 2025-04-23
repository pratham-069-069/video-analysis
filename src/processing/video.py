# src/processing/video.py

import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Default Scene detection parameters (can be overridden)
DEFAULT_FRAME_DIFFERENCE_THRESHOLD = 0.4 # Start tuning here again
DEFAULT_RESIZE_WIDTH = 320
DEFAULT_BLUR_KERNEL = (7, 7)
DEFAULT_PIXEL_THRESHOLD = 20

def detect_scenes(video_path,
                  threshold=DEFAULT_FRAME_DIFFERENCE_THRESHOLD,
                  resize_width=DEFAULT_RESIZE_WIDTH,
                  blur_kernel=DEFAULT_BLUR_KERNEL,
                  pixel_threshold=DEFAULT_PIXEL_THRESHOLD,
                  min_scene_duration_sec=0.5): # Heuristic to avoid rapid detections
    """
    Opens a video file and detects scene changes using frame differencing.
    Returns a list of timestamps (in seconds) where scene changes are detected.
    """
    scene_change_timestamps = []
    if not os.path.exists(video_path):
        log.error(f"Video file not found at {video_path}")
        return scene_change_timestamps

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error(f"Could not open video file {video_path}")
        return scene_change_timestamps

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: # Check for invalid FPS
        log.warning(f"Invalid FPS ({fps}) detected for {os.path.basename(video_path)}. Assuming 30 FPS.")
        fps = 30 # Assume a default FPS

    min_scene_duration_frames = int(min_scene_duration_sec * fps) # Convert min duration to frames
    last_scene_cut_frame = -min_scene_duration_frames # Initialize to allow detection at start

    prev_frame_gray = None
    frame_count = 0
    detected_count = 0

    log.info(f"Processing video: {os.path.basename(video_path)} at {fps:.2f} FPS for scene detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # log.info("Finished processing video frames.")
            break # End of video

        # --- Frame Processing for Scene Detection ---
        try:
            # 1. Resize for speed (optional)
            if resize_width and frame.shape[1] > resize_width:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                new_height = int(resize_width * aspect_ratio)
                current_frame_resized = cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                current_frame_resized = frame

            # 2. Convert to Grayscale
            current_frame_gray = cv2.cvtColor(current_frame_resized, cv2.COLOR_BGR2GRAY)

            # 3. Blur slightly to reduce noise
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, blur_kernel, 0)

            # 4. Compare with previous frame (if exists)
            if prev_frame_gray is not None:
                # Check if enough frames passed since last cut
                if frame_count >= last_scene_cut_frame + min_scene_duration_frames:
                    frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
                    _, diff_thresh = cv2.threshold(frame_diff, pixel_threshold, 255, cv2.THRESH_BINARY)
                    changed_pixels_ratio = np.sum(diff_thresh > 0) / diff_thresh.size

                    if changed_pixels_ratio > threshold:
                        timestamp_sec = frame_count / fps
                        scene_change_timestamps.append(timestamp_sec)
                        last_scene_cut_frame = frame_count # Record when this cut happened
                        detected_count += 1
                        # log.info(f"  Scene change detected at frame {frame_count} ({timestamp_sec:.2f}s) - Ratio: {changed_pixels_ratio:.3f}")

            # Update previous frame
            prev_frame_gray = current_frame_gray
            frame_count += 1

            # Optional: Display processing progress less frequently
            # if frame_count % 500 == 0:
            #     log.info(f"  Processed {frame_count} frames...")

        except cv2.error as e:
             log.error(f"OpenCV error processing frame {frame_count}: {e}", exc_info=True)
             # Optionally decide whether to break or try to continue
             break # Stop processing on OpenCV error for safety
        except Exception as e:
             log.error(f"Unexpected error processing frame {frame_count}: {e}", exc_info=True)
             break # Stop processing on other errors

    cap.release() # Release the video capture object
    log.info(f"Detected {detected_count} potential scene changes for {os.path.basename(video_path)}.")
    return scene_change_timestamps