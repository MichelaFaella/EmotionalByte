import os
import re


def parse_transcriptions_file(file_path):
    utterances = []  # List to store all parsed utterances

    # Regular expression to extract utterance ID, start time, end time, and text
    pattern = r'(\w+) \[(\d+\.\d+)-(\d+\.\d+)\]: (.+)'

    # Open the transcription file
    with open(file_path, 'r') as file:
        for line in file:
            # Try to match each line with the expected pattern
            match = re.match(pattern, line.strip())
            if match:
                # Extract matched groups: ID, start time, end time, and the spoken text
                utterance_id, start, end, text = match.groups()

                # Get the speaker gender (F or M) from the utterance ID
                speaker = utterance_id.split('_')[-1][0]  # e.g., "F000" -> "F"

                # Create a dictionary with the parsed data
                utterances.append({
                    'id': utterance_id,
                    'speaker': speaker,
                    'start_time': float(start),
                    'end_time': float(end),
                    'text': text
                })

    # Return the list of parsed utterances
    return utterances


def parse_all_transcriptions(base_path, current_session):
    all_data = []
    session = current_session  # Currently we are only parsing Session1 for testing

    # Construct the path to the transcriptions folder for Sessions
    path = os.path.join(base_path, f"Session{session}", "transcriptions")

    # Loop through all files in the transcriptions folder
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            full_path = os.path.join(path, file_name)

            # Parse the transcription file using the previously defined function
            data = parse_transcriptions_file(full_path)
            all_data.extend(data)

    return all_data


def parse_emotion_labels(base_path, current_session):
    emotion_map = {}  # key: utterance_id, value: emotion

    session = current_session
    emo_path = os.path.join(base_path, f"Session{session}", "EmoEvaluation")

    for file_name in os.listdir(emo_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(emo_path, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    match = re.search(r'\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\w+)', line)
                    if match:
                        utt_id = match.group(3)
                        emotion = match.group(4)
                        emotion_map[utt_id] = emotion
    return emotion_map
