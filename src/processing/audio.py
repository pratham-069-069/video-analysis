# src/processing/audio.py

import whisper
import os
import time
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Suppress specific known warnings from Whisper if needed
# warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe')
warnings.filterwarnings("ignore") # Suppress all user warnings for cleaner logs for now


# Whisper Model Selection
DEFAULT_WHISPER_MODEL = "tiny.en" # Faster for testing
# DEFAULT_WHISPER_MODEL = "base.en" # Good balance

# Global variable to cache the loaded model
# Be cautious with globals in more complex scenarios (e.g., multiprocessing)
_whisper_model_cache = {}

def load_whisper_model(model_name=DEFAULT_WHISPER_MODEL):
    """Loads a Whisper model, caching it for efficiency."""
    global _whisper_model_cache
    if model_name in _whisper_model_cache:
        # log.info(f"Using cached Whisper model '{model_name}'.")
        return _whisper_model_cache[model_name]

    log.info(f"Loading Whisper model '{model_name}'...")
    start_time = time.time()
    try:
        # Consider specifying download_root if needed
        model = whisper.load_model(model_name)
        load_time = time.time() - start_time
        log.info(f"Model '{model_name}' loaded in {load_time:.2f} seconds.")
        _whisper_model_cache[model_name] = model # Cache the loaded model
        return model
    except Exception as e:
        log.error(f"Error loading Whisper model '{model_name}': {e}", exc_info=True)
        log.error("Ensure the model name is correct, dependencies (like torch) are installed,")
        log.error("and you have internet access if downloading for the first time.")
        return None


def transcribe_audio(video_path, model_name=DEFAULT_WHISPER_MODEL):
    """
    Transcribes the audio from a video file using Whisper.
    Returns the structured transcription result including segments.
    """
    if not os.path.exists(video_path):
        log.error(f"Video file not found at {video_path} for transcription.")
        return None

    model = load_whisper_model(model_name)
    if not model:
        log.error("Transcription failed: Whisper model could not be loaded.")
        return None # Model loading failed

    log.info(f"Starting transcription for: {os.path.basename(video_path)} using model '{model_name}'...")
    start_time = time.time()
    try:
        # Process with fp16=False for better CPU compatibility. Set verbose=None for default progress.
        # Consider language= "en" if using multilingual model but expect english
        result = model.transcribe(video_path, fp16=False, verbose=None) # verbose=None is whisper default
        end_time = time.time()
        log.info(f"Transcription finished in {end_time - start_time:.2f} seconds.")
        # result dictionary contains 'text' and 'segments' keys
        if "segments" not in result or not isinstance(result["segments"], list):
             log.warning(f"Transcription result for {os.path.basename(video_path)} might be incomplete or malformed: 'segments' key missing or not a list.")
             # Still return result, calling code might handle it
        return result
    except Exception as e:
        log.error(f"Error during transcription: {e}", exc_info=True)
        log.error("Ensure ffmpeg is installed correctly and accessible in your PATH.")
        return None