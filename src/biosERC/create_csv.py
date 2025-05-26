import os
import pandas as pd
from collections import defaultdict

def create_csv(data_path):
    # Initialize a list to store CSV rows
    rows = []
    
    # Loop over Session1 to Session5
    for session_num in range(1, 6):
        session_name = f"Session{session_num}"
        transcriptions_path = os.path.join(data_path, session_name, "dialog", "transcriptions")

        # Iterate through each session file (e.g., Ses01.csv, Ses02.csv, ...)
        for filename in sorted(os.listdir(transcriptions_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(transcriptions_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Create a conversation text from the utterances
                dialogue_id = filename.replace(".txt", "")
                conversation = []
                speakers_in_session = set()

                for line in lines:
                    # Example line: Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.
                    if ": " not in line:
                        continue  # skip malformed lines
                    meta, text = line.split(": ", 1)
                    parts = meta.strip().split("_")
                    if len(parts) < 3:
                        continue  # unexpected line
                    speaker_tag = parts[-1][0]  # 'F' or 'M'
                    speakers_in_session.add(speaker_tag)
                    conversation.append(f"{speaker_tag}: {text.strip()}")

                conversation_text = "\n".join(conversation)

                # Create one row per speaker for this session
                for speaker in speakers_in_session:
                    rows.append({
                        "dialogue_id": dialogue_id,
                        "speaker": speaker,
                        "conversation_text": conversation_text
                    })

    # Save to CSV
    output_df = pd.DataFrame(rows)
    output_df.to_csv("src/biosERC/iemocap_prompts.csv", index=False)

    print("Prompt CSV saved to iemocap_prompts.csv")

import pandas as pd
import csv

import csv

def clean_semicolons(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        writer.writerow(['dialogue_id', 'speaker', 'description'])  # write header

        for row in reader:
            if not row:
                continue

            line = row[0]
            # Try to split the first part as ID, speaker, then assume the rest is description
            parts = line.split(',', 2)
            if len(parts) != 3:
                continue  # skip malformed rows

            dialogue_id = parts[0].strip().strip('"')
            speaker = parts[1].strip().strip('"')
            description = parts[2].strip().strip('"')

            # Replace all semicolons in description with commas
            cleaned_description = description.replace(";", ",")

            writer.writerow([dialogue_id, speaker, cleaned_description])

    print(f"Cleaned file written to: {output_path}")

# Example usage:
clean_semicolons(
    input_path="src/biosERC/iemocap_spk_bios.csv",
    output_path="src/biosERC/iemocap_spk_bios_clean.csv"
)


data_path = "data/IEMOCAP/IEMOCAP_full_release"  # Adjust to your directory
#create_csv(data_path)