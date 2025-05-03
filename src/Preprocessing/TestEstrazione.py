import pickle
import numpy as np

# Path to the saved pickle file.
pickle_path = "../../data/iemocap_multimodal_features_par.pkl"

# Loading
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Extraction of variables.
video_ids = data[0]
videoSpeakers = data[1]
videoLabels = data[2]
videoText = data[3]
videoAudio = data[7]
trainVid = data[8]
testVid = data[9]

# Basic inspection.
print(f"Total number of video: {len(video_ids)}")
print(f"Train: {len(trainVid)} | Test: {len(testVid)}")
print(f"First video ID: {video_ids[0]}")
print(f"Speaker: {videoSpeakers[video_ids[0]]}")
print(f"Label: {videoLabels[video_ids[0]]}")
print(f"Embedding text (shape): {videoText[video_ids[0]].shape}")
print(f"Feature audio (shape): {videoAudio[video_ids[0]].shape}")
