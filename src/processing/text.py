# src/processing/text.py

# Using Transformers for higher accuracy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch # Or tensorflow if using TF models
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
# Choose a robust sentiment model (check Hugging Face Hub for options)
# Examples:
# SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" # Good default, faster
# SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest" # Trained on tweets, potentially better for comments
# SENTIMENT_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis" # Another tweet-focused one
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


# --- Caching for loaded pipeline ---
_sentiment_pipeline_cache = {}

def get_sentiment_pipeline(model_name=DEFAULT_SENTIMENT_MODEL):
    """Loads and caches a Hugging Face sentiment analysis pipeline."""
    global _sentiment_pipeline_cache
    if model_name in _sentiment_pipeline_cache:
        # log.info(f"Using cached sentiment pipeline for model '{model_name}'.")
        return _sentiment_pipeline_cache[model_name]

    log.info(f"Loading sentiment analysis pipeline for model '{model_name}'...")
    start_time = time.time()
    try:
        # Check if GPU is available, otherwise use CPU
        device = 0 if torch.cuda.is_available() else -1 # device=0 for first GPU, -1 for CPU
        log.info(f"Using device: {'GPU 0' if device == 0 else 'CPU'}")

        # Load tokenizer and model to potentially handle truncation if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Using device explicitly in pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device # Explicitly set device
            # Consider adding truncation=True if texts might exceed model limits
            # truncation=True, max_length=512 # Example, check model's max length
        )
        load_time = time.time() - start_time
        log.info(f"Sentiment pipeline loaded in {load_time:.2f} seconds.")
        _sentiment_pipeline_cache[model_name] = sentiment_pipeline
        return sentiment_pipeline
    except Exception as e:
        log.error(f"Error loading sentiment pipeline '{model_name}': {e}", exc_info=True)
        log.error("Ensure model name is correct, libraries (transformers, torch/tf) are installed,")
        log.error("and you have internet access if downloading for the first time.")
        return None

def analyze_sentiment_transformer(text, sentiment_pipeline):
    """
    Analyzes the sentiment of text using a loaded Transformers pipeline.
    Returns a dictionary with 'label' ('POSITIVE'/'NEGATIVE') and 'score'.
    """
    if not sentiment_pipeline:
        log.error("Cannot analyze sentiment: Pipeline not loaded.")
        return {'label': 'ERROR', 'score': 0.0}
    if not isinstance(text, str) or not text.strip():
        return {'label': 'NEUTRAL', 'score': 0.0} # Assign neutral for empty

    try:
        # The pipeline returns a list, usually with one dictionary
        # Example: [{'label': 'POSITIVE', 'score': 0.9998}]
        # Handle potential truncation issues if text is very long and truncation isn't enabled
        # Some models have input length limits (e.g., 512 tokens)
        max_length = sentiment_pipeline.tokenizer.model_max_length
        if len(sentiment_pipeline.tokenizer.encode(text)) > max_length:
             log.warning(f"Input text exceeds model's max length ({max_length}). Truncating text for sentiment analysis.")
             # Simple truncation - more sophisticated chunking might be needed for long texts
             truncated_text = sentiment_pipeline.tokenizer.decode(
                 sentiment_pipeline.tokenizer.encode(text, max_length=max_length, truncation=True)
             )
             result = sentiment_pipeline(truncated_text)[0]
        else:
             result = sentiment_pipeline(text)[0]

        return result

    except Exception as e:
        log.error(f"Error during sentiment analysis for text: '{text[:50]}...' - {e}", exc_info=True)
        return {'label': 'ERROR', 'score': 0.0} # Return error indicator