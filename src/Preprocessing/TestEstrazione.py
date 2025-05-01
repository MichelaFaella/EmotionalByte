import pickle
import numpy as np

# Percorso al file pickle salvato
pickle_path = "../../data/iemocap_multimodal_features_par.pkl"

# Caricamento
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# Estrazione delle variabili
video_ids = data[0]
videoSpeakers = data[1]
videoLabels = data[2]
videoText = data[3]
videoAudio = data[7]
trainVid = data[8]
testVid = data[9]

# Ispezione di base
print(f"Numero totale video: {len(video_ids)}")
print(f"Train: {len(trainVid)} | Test: {len(testVid)}")
print(f"Primo video ID: {video_ids[0]}")
print(f"Speaker: {videoSpeakers[video_ids[0]]}")
print(f"Label: {videoLabels[video_ids[0]]}")
print(f"Embedding testo (shape): {videoText[video_ids[0]].shape}")
print(f"Feature audio (shape): {videoAudio[video_ids[0]].shape}")
