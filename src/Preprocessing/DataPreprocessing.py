import os
import pickle

import opensmile
from transformers import RobertaTokenizer, RobertaModel


import torch

root = "./data/IEMOCAP/"
sessions = [f"Session{i}" for i in range(1, 6)]


videoTextConv = {}
videoAudioConv = {}
videoSpeakersConv = {}
videoLabelsConv = {}
videoTimeConv = {}

videoText = {}
videoAudio = {}
videoSpeakers = {}
videoLabels = {}


trainVid, testVid = [], []

# happy, sad, angry, neutral, frustrated, excited, surprised, fearful, disgusted, indefinite, other
label_map_10 = {'hap': 0, 'sad': 1, 'ang': 2, 'neu': 3, 'fru':4, 'exc':5,
             'sur':6, 'fea':7, 'dis':8,'xxx':9, 'oth':10}

label_map_6 = {'hap': 0, 'sad': 1, 'ang': 2, 'neu': 3, 'fru':4, 'exc':0,
             'sur':5, 'fea':5, 'dis':5,'xxx':5, 'oth':5}

label_map = label_map_6

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
model.eval()

smile = opensmile.Smile(
    #feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_set=opensmile.FeatureSet.IS10_paraling,
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
                    print(f'VID: {vid}')
                    emotion = parts[4]
                    if emotion not in label_map:
                        #continue
                        print(f"{emotion} not in label_map")

                    # 1. TEXT
                    text = None
                    with open(trans_dir + "/" + emo_file, encoding='utf8') as textfile:
                        for tline in textfile:
                            if vid in tline:
                                text = tline.split(":", 1)[-1].strip()
                                break

                    if text is None:
                        print(f" Testo non trovato per {vid} in {emo_file}")
                        continue  # skip this sample

                    # now it is safe to tokenize
                    inputs = tokenizer(text, return_tensors="pt")

                    with torch.no_grad():
                        outputs = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
                    videoTextConv.setdefault(vid, []).append(outputs.numpy())


                    # 2. AUDIO
                    wav = os.listdir(wav_fold_path)[idx]
                    print(f'WAV: {wav}')
                    wav_path = wav_dir + "/" + wav_fold + "/" + wav
                    idx = idx + 1

                    try:
                        features = smile.process_file(wav_path)  # pandas DataFrame
                        audio_feat = features.iloc[0].to_numpy()  # numpy vector
                        videoAudioConv.setdefault(vid, []).append(audio_feat)
                    except Exception as e:
                        print(f"Errore nell'estrazione audio per {vid}: {e}")


                    # 3. Speaker
                    speaker = [1,0] if vid.split("_")[-1][0] == 'M' else [0,1]
                    videoSpeakersConv.setdefault(vid, []).append(speaker)


                    # 4. Label
                    videoLabelsConv.setdefault(vid, []).append(label_map[emotion])

                    # 5. Time
                    time = parts[0][1:]
                    time = float(time)
                    videoTimeConv.setdefault(vid, []).append(time)


            sortedKey = sorted(videoTimeConv, key=videoTimeConv.get)

            videoTextConv = {k: videoTextConv[k] for k in sortedKey}
            videoAudioConv = {k: videoAudioConv[k] for k in sortedKey}
            videoSpeakersConv = {k: videoSpeakersConv[k] for k in sortedKey}
            videoLabelsConv = {k: videoLabelsConv[k] for k in sortedKey}


            convID = emo_file.split(".")[0]
            videoText.setdefault(convID, []).extend(videoTextConv.values())
            videoAudio.setdefault(convID, []).extend(videoAudioConv.values())
            videoSpeakers.setdefault(convID, []).extend(videoSpeakersConv.values()) #[['F'] , ['M']]
            videoLabels.setdefault(convID, []).extend(videoLabelsConv.values())

            videoTextConv = {}
            videoAudioConv = {}
            videoSpeakersConv = {}
            videoLabelsConv = {}
            videoTimeConv = {}


        # 5. Train/Test division
        if session != "Session5":
            trainVid.append(convID)
        else:
            testVid.append(convID)

print("#Conversation in train:", len(trainVid))
print("#Conversation in Test:", len(testVid))


# Save file pickle
with open("./data/iemocap_multimodal_features_6labels.pkl", "wb") as f:
    pickle.dump(
        (list(videoText.keys()), videoSpeakers, videoLabels, videoText,
         None, None, None, videoAudio, trainVid, testVid),
        f
    )
