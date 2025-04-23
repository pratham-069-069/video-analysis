# src/analyze_text.py

import os
import json
import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv

# --- Load Environment Variables (Optional for this script, but good practice) ---
load_dotenv('../.env')

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_DIR = os.path.join(BASE_DIR, "data")
COMMENTS_DIR = os.path.join(DATA_DIR, 'comments')

# --- Helper Function ---

def analyze_sentiment(text):
    """
    Analyzes the sentiment of a given text using TextBlob.
    Returns polarity (-1.0 to +1.0) and subjectivity (0.0 to 1.0).
    """
    if not isinstance(text, str) or not text.strip():
        # Return neutral sentiment for empty or non-string input
        return 0.0, 0.0
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity
    except Exception as e:
        print(f"Error analyzing sentiment for text: '{text[:50]}...' - {e}")
        return 0.0, 0.0 # Return neutral on error

def load_comments_from_json(video_id, comments_dir):
    """Loads comments from the JSON file for a given video ID."""
    comments_path = os.path.join(comments_dir, f"{video_id}_comments.json")
    try:
        with open(comments_path, 'r', encoding='utf-8') as f:
            comments_data = json.load(f)
        # Convert to DataFrame for easier handling, ensure 'text' column exists
        df = pd.DataFrame(comments_data)
        if 'text' not in df.columns:
             print(f"Warning: 'text' column not found in {comments_path}. Returning empty DataFrame.")
             return pd.DataFrame(columns=['comment_id', 'text']) # Return empty with expected columns
        # Select only relevant columns, handle potential missing comment_id
        if 'comment_id' not in df.columns:
            df['comment_id'] = [f"comment_{i}" for i in range(len(df))] # Generate placeholder IDs
        return df[['comment_id', 'text']]

    except FileNotFoundError:
        print(f"Error: Comments file not found at {comments_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {comments_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading comments for {video_id}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Text Sentiment Analysis Script ---")

    # --- Select Video to Process ---
    # Use the SAME video ID for which you have the transcript segments below
    TEST_VIDEO_ID = "bcGxg3c1HE8" # <<< MAKE SURE THIS MATCHES THE TRANSCRIPT BELOW

    # --- Placeholder for Transcript Segments ---
    # Copy and paste the 'segments' list output from running process_audio.py
    # for the TEST_VIDEO_ID above.
    # Example structure: [{'start': 0.0, 'end': 1.92, 'text': ' Segment text...'}, ...]
    placeholder_transcript_segments = [
        {'start': 0.0, 'end': 1.92, 'text': ' 2024 is the year of elections,'},
        {'start': 1.92, 'end': 4.2, 'text': ' but at least 50 countries around the world'},
        {'start': 4.2, 'end': 5.12, 'text': ' earned the polls.'},
        {'start': 5.12, 'end': 8.28, 'text': " 2024 is a year of what is, but democracies."},
        {'start': 8.28, 'end': 11.0, 'text': " This year's results ended in a strong showing"},
        {'start': 11.0, 'end': 14.76, 'text': ' for the far right across much of the European Union.'},
        {'start': 14.76, 'end': 17.32, 'text': ' Democracy has been surprisingly resilient.'},
        {'start': 17.32, 'end': 22.04, 'text': ' The 2024 presidential campaign went from deja vu to utter chaos.'},
        {'start': 22.04, 'end': 23.12, 'text': ' Look what happened.'}, {'start': 23.12, 'end': 24.04, 'text': ' This is crazy.'},
        {'start': 30.0, 'end': 44.96, 'text': ' At a time when the world is so turbulent,'},
        {'start': 44.96, 'end': 47.2, 'text': ' the Olympics allows people around the globe'},
        {'start': 47.2, 'end': 49.08, 'text': ' to share some common experiences.'},
        {'start': 49.08, 'end': 51.76, 'text': " But it's athletes creating memorable moments"},
        {'start': 51.76, 'end': 53.84, 'text': ' that really got people talking this time.'},
        {'start': 53.84, 'end': 55.72, 'text': " I'm not sure what to do."},
        {'start': 55.72, 'end': 58.52, 'text': " I'm not sure what to do."},
        {'start': 58.52, 'end': 60.68, 'text': " I'm not sure what to do."},
        {'start': 60.68, 'end': 62.16, 'text': " I'm not sure what to do."},
        {'start': 62.16, 'end': 64.36, 'text': " I'm not sure what to do."},
        {'start': 64.36, 'end': 67.72, 'text': ' Rafael Nadal has announced his retirement from professional tennis.'},
        {'start': 67.72, 'end': 70.52, 'text': " Interesting women's sports has never been higher."},
        {'start': 70.52, 'end': 72.52, 'text': " I didn't know that that was happening."},
        {'start': 72.52, 'end': 74.88, 'text': " I've seen it, yeah."},
        {'start': 74.88, 'end': 77.36, 'text': " This summer's music scene has been dominated"},
        {'start': 77.36, 'end': 80.48, 'text': ' by a wave of young women singers described a crime.'},
        {'start': 80.48, 'end': 81.12, 'text': ' Me.'},
        {'start': 81.12, 'end': 82.16, 'text': ' Very junior.'},
        {'start': 82.16, 'end': 83.2, 'text': ' Very bright folk.'},
        {'start': 83.2, 'end': 86.04, 'text': ' Beyonce is making history the first black woman'},
        {'start': 86.04, 'end': 89.88, 'text': ' to top the Billboard country charts.'},
        {'start': 89.88, 'end': 93.32, 'text': ' Bitcoin just hit $100,000.'},
        {'start': 93.32, 'end': 94.88, 'text': ' Andflation is still too high.'},
        {'start': 94.88, 'end': 96.92, 'text': " I'm looking for a man in finance."},
        {'start': 96.92, 'end': 97.56, 'text': ' Tristan.'},
        {'start': 97.56, 'end': 100.16, 'text': ' A multi-state manhunt suspect involved'},
        {'start': 100.16, 'end': 102.28, 'text': ' in the murder of United Health Care CEO.'},
        {'start': 102.28, 'end': 104.44, 'text': ' hottest cold-blooded killer in America.'},
        {'start': 104.44, 'end': 108.24, 'text': ' Another drone type object traversing this airspace.'},
        {'start': 108.24, 'end': 110.36, 'text': ' A Microsoft outage is causing problems'},
        {'start': 110.36, 'end': 112.84, 'text': ' for airlines, banks, government agencies,'},
        {'start': 112.88, 'end': 114.16, 'text': ' and companies around the world.'},
        {'start': 114.16, 'end': 118.28, 'text': ' Two Boeing astronauts stranded on the international space station.'},
        {'start': 118.28, 'end': 120.48, 'text': " That's just the way it goes sometimes."},
        {'start': 120.48, 'end': 122.36, 'text': " 1.24 is we're not fine this year."},
        {'start': 122.36, 'end': 125.6, 'text': " You know, we're postponed to a 25."},
        {'start': 125.6, 'end': 128.0, 'text': ' It is virtually certain 2024.'},
        {'start': 128.0, 'end': 130.56, 'text': " Could be the planet's hottest year on record."},
        {'start': 130.56, 'end': 132.16, 'text': " We can see that we're not ready"},
        {'start': 132.16, 'end': 134.36, 'text': " for the storms that we're ever saving now."},
        {'start': 134.36, 'end': 136.2, 'text': " And it's going to be worse in the future."},
        {'start': 136.2, 'end': 143.56, 'text': ' Those in the industry call it a permacrisis,'},
        {'start': 143.56, 'end': 146.92, 'text': ' a sense that the world is latching from warning stream events.'},
        {'start': 146.92, 'end': 147.72, 'text': ' To another.'},
        {'start': 147.72, 'end': 151.36, 'text': " Russia's war in Ukraine enters its third years of thousands"},
        {'start': 151.36, 'end': 153.42, 'text': ' at the displaced across the Haitian Cam'},
        {'start': 153.42, 'end': 155.64, 'text': ' and authoritarian crisis in Sudan is worsening.'},
        {'start': 155.64, 'end': 158.44, 'text': " South Korea's parliament voting to impeach President"},
        {'start': 158.44, 'end': 162.16, 'text': ' Yunsuki over his short-lived declaration of martial law.'},
        {'start': 162.16, 'end': 164.92, 'text': ' Israel has expanded its bombardment of Lebanon'},
        {'start': 164.92, 'end': 166.16, 'text': ' striking new areas.'},
        {'start': 166.16, 'end': 169.88, 'text': ' The alleged that Netanyahu Gallant used starvation'},
        {'start': 169.88, 'end': 171.48, 'text': ' as a tool of war.'},
        {'start': 171.48, 'end': 172.48, 'text': ' Yeah.'},
        {'start': 172.48, 'end': 173.48, 'text': ' Oh, yeah.'},
        {'start': 176.0, 'end': 179.44, 'text': ' We did not get justice once again.'},
        {'start': 179.44, 'end': 181.76, 'text': " I don't like that this is a fact of life."},
        {'start': 181.76, 'end': 182.48, 'text': ' Screw you.'},
        {'start': 182.48, 'end': 184.36, 'text': ' If you think this is a fact of life,'},
        {'start': 184.36, 'end': 185.8, 'text': ' then I should just get over.'},
        {'start': 189.16, 'end': 192.36, 'text': ' The mental state in the Middle East.'},
        {'start': 192.36, 'end': 196.96, 'text': ' Decades of savage horrific rule over in a matter of days.'},
        {'start': 196.96, 'end': 198.56, 'text': ' We have so much hope.'},
        {'start': 198.56, 'end': 201.8, 'text': ' We hope Blackboard with their and with their future for Syria.'},
        {'start': 201.8, 'end': 204.04, 'text': ' History has been made in Mexico.'},
        {'start': 204.04, 'end': 206.88, 'text': " Claudia Shambal has been elected as the country's first"},
        {'start': 206.88, 'end': 209.4, 'text': ' in a female press.'},
        {'start': 209.4, 'end': 211.44, 'text': ' Cameroon has become the first country'},
        {'start': 211.44, 'end': 213.96, 'text': ' to roll out a new malaria vaccine.'},
        {'start': 213.96, 'end': 217.2, 'text': ' France has become the only country to guarantee abortion'},
        {'start': 217.2, 'end': 220.48, 'text': ' as a constitutional right.'},
        {'start': 220.48, 'end': 225.64, 'text': " For so long, I've always wanted to be different."},
        {'start': 225.64, 'end': 231.0, 'text': ' And now I realize I just need to pee myself.'},
        {'start': 231.0, 'end': 234.04, 'text': " We don't want to bring children into a world"},
        {'start': 234.04, 'end': 238.12, 'text': " where from the very beginning they believe everything's hopeless."},
        {'start': 238.12, 'end': 242.12, 'text': ' When if we get together, if we roll up our sleeves'},
        {'start': 242.12, 'end': 247.4, 'text': ' and take action, each of us doing what we feel is important,'},
        {'start': 247.4, 'end': 251.24, 'text': ' then there is hope for the future.'},
        {'start': 251.24, 'end': 252.2, 'text': ' No!'},
        {'start': 252.2, 'end': 253.2, 'text': ' No!'},
        {'start': 259.2, 'end': 263.04, 'text': " It's important to remember that we are not allowed a fear, is?"},
        {'start': 263.04, 'end': 264.04, 'text': " And that's OK."},
        {'start': 269.44, 'end': 270.36, 'text': ' Thanks for watching.'},
        {'start': 270.36, 'end': 273.36, 'text': ' This was our 11th end of year wrap-up video.'},
        {'start': 273.36, 'end': 276.36, 'text': ' It was only possible because of you, our viewers.'},
        {'start': 276.36, 'end': 278.32, 'text': ' But especially our members.'},
        {'start': 278.32, 'end': 279.2, 'text': ' Do you like what we do?'},
        {'start': 279.2, 'end': 281.04, 'text': " And you'd like to support that work?"},
        {'start': 281.04, 'end': 283.48, 'text': ' Go to Vox.com slash memberships.'},
        {'start': 283.48, 'end': 284.28, 'text': ' Thank you.'},
        {'start': 284.28, 'end': 287.44, 'text': ' And happy new year from the Vox video team.'}
    ] # End of placeholder_transcript_segments list

    # --- Analyze Transcript ---
    print(f"\n--- Analyzing Transcript Segments for Video: {TEST_VIDEO_ID} ---")
    print("Polarity: -1 (Negative) to +1 (Positive)")
    print("Subjectivity: 0 (Objective) to 1 (Subjective)")
    print("-" * 60)
    transcript_sentiments = []
    for i, segment in enumerate(placeholder_transcript_segments):
        text = segment.get('text', '') # Get text, default to empty string if missing
        polarity, subjectivity = analyze_sentiment(text)
        transcript_sentiments.append({
            'video_id': TEST_VIDEO_ID,
            'segment_index': i,
            'start_time': segment.get('start'),
            'end_time': segment.get('end'),
            'text': text.strip(), # Remove leading/trailing whitespace
            'polarity': polarity,
            'subjectivity': subjectivity
        })
        # Print results neatly
        print(f"[{segment.get('start'):>6.2f}s -> {segment.get('end'):>6.2f}s] Pol: {polarity:>+6.3f}, Subj: {subjectivity:.3f} | Text: {text.strip()[:80]}...")

    # --- Analyze Comments ---
    print(f"\n--- Analyzing Comments for Video: {TEST_VIDEO_ID} ---")
    comments_df = load_comments_from_json(TEST_VIDEO_ID, COMMENTS_DIR)
    comment_sentiments = []
    if comments_df is not None and not comments_df.empty:
        print(f"Loaded {len(comments_df)} comments.")
        print("-" * 60)
        # Use .itertuples() for slightly better performance than iterrows
        for row in comments_df.itertuples(index=False):
            comment_id = row.comment_id
            text = row.text
            polarity, subjectivity = analyze_sentiment(text)
            comment_sentiments.append({
                'video_id': TEST_VIDEO_ID,
                'comment_id': comment_id,
                'text': text,
                'polarity': polarity,
                'subjectivity': subjectivity
            })
            # Print results neatly
            print(f"Comment ID: {comment_id[:15]}... | Pol: {polarity:>+6.3f}, Subj: {subjectivity:.3f} | Text: {text[:80]}...")
    elif comments_df is not None and comments_df.empty:
         print("Comment DataFrame loaded but is empty.")
    else:
        print("Could not load or process comments.")

    # Note: In a real pipeline, transcript_sentiments and comment_sentiments
    # would be saved to MongoDB or prepared for loading into Snowflake.
    print("\nSentiment analysis finished.")
    # Example: Show first few results stored in lists
    # print("\nSample Transcript Sentiment Data Structure:")
    # print(transcript_sentiments[:2])
    # print("\nSample Comment Sentiment Data Structure:")
    # print(comment_sentiments[:2])

    print("\nScript finished.")


    