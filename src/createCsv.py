import csv
import os
import pandas as pd
from src.utility.utility_text import parse_all_transcriptions, parse_emotion_labels

# Set base path
base_path = "data"

for i in range(1,6):
    # Step 1: Parsing transcriptions
    print(f"Parsing transcription files from Session{i}...")
    transcriptions = parse_all_transcriptions(base_path, i)
    print(f"Parsed {len(transcriptions)} utterances.")

    # Step 2: Load emotion labels
    print("Loading emotion labels...")
    emotion_labels = parse_emotion_labels(base_path, i)
    print(f"Loaded {len(emotion_labels)} emotion labels.")

    # Step 3: Merge into DataFrame
    df = pd.DataFrame(transcriptions)
    df['emotion'] = df['id'].map(emotion_labels)

    # Step 4: Drop utterances with no emotion label (optional)
    df = df.dropna(subset=['emotion'])

    # Step 5: Save to CSV
    output_dir = os.path.join(base_path, f'Session{i}/output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'session{i}_transcriptions_with_emotions.csv')

    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Saved final dataset to: {output_file}")
