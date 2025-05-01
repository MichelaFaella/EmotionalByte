import os
import pickle

import opensmile
from transformers import RobertaTokenizer, RobertaModel


import torch

root = "data/IEMOCAP/"
sessions = [f"Session{i}" for i in range(1, 3)]


videoText = {}
videoAudio = {}
videoSpeakers = {}
videoLabels = {}
trainVid, testVid = [], []

# happy, sad, angry, neutral, frustrated, excited, surprised, fearful, disgusted, indefinite, other
label_map = {'hap': 0, 'sad': 1, 'ang': 2, 'neu': 3, 'fru':4, 'exc':5,
             'sur':6, 'fea':7, 'dis':8,'xxx':9, 'oth':10}

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
model.eval()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv01b,
    feature_level=opensmile.FeatureLevel.Functionals
)

for session in sessions:
    print(f"Session: {session}")
    session_path = root + session
    emo_dir = session_path + "/EmoEvaluation"
    wav_dir = session_path + "/wav"
    trans_dir = session_path + "/transcriptions"

    for emo_file, wav_fold in zip(os.listdir(emo_dir) , os.listdir(wav_dir)):
        print(f"{emo_file}: {wav_fold}")
        wav_fold_path = wav_dir + "/"  + wav_fold
        idx = 0
        if not emo_file.endswith(".txt"):
            continue
        with open(emo_dir + "/" + emo_file) as f:
            for textline in f:
                if textline.startswith("["):
                    print(f"{idx}")
                    parts = textline.strip().split()
                    vid = parts[3]
                    emotion = parts[4]
                    if emotion not in label_map:
                        #continue
                        print(f"{emotion} not in label_map")

                    # 1. Testo
                    text = None
                    with open(trans_dir + "/" + emo_file, encoding='utf8') as textfile:
                        for tline in textfile:
                            if vid in tline:
                                text = tline.split(":", 1)[-1].strip()
                                break

                    if text is None:
                        print(f" Testo non trovato per {vid} in {emo_file}")
                        continue  # salta questo sample

                    # ora è sicuro tokenizzare
                    inputs = tokenizer(text, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
                    videoText.setdefault(vid, []).append(outputs.numpy())


                    # 2. Audio
                    wav = os.listdir(wav_fold_path)[idx]
                    wav_path = wav_dir + "/" + wav_fold + "/" + wav
                    idx = idx + 1

                    try:
                        features = smile.process_file(wav_path)  # pandas DataFrame
                        audio_feat = features.iloc[0].to_numpy()  # Vettore numpy
                        videoAudio.setdefault(vid, []).append(audio_feat)
                    except Exception as e:
                        print(f"Errore nell'estrazione audio per {vid}: {e}")


                    # 3. Speaker
                    speaker = 'M' if vid[7] == 'M' else 'F'
                    videoSpeakers.setdefault(vid, []).append(speaker)


                    # 4. Label
                    videoLabels.setdefault(vid, []).append(label_map[emotion])


                    # 5. Divisione in train/test
                    if session == "Session1":
                        trainVid.append(vid)
                    else:
                        testVid.append(vid)

print("Numero video in train:", len(trainVid))
print("Numero video in test:", len(testVid))
print("Esempi:", trainVid[:5])

# Salvataggio finale
with open("iemocap_multimodal_features.pkl", "wb") as f:
    pickle.dump(
        (list(videoText.keys()), videoSpeakers, videoLabels, videoText,
         None, None, None, videoAudio, trainVid, testVid),
        f
    )


