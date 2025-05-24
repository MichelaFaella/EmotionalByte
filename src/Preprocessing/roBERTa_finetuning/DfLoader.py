import os
import re
import pandas as pd

root = "../../data/IEMOCAP/"
sessions = [f"Session{i}" for i in range(1, 6)]

label_map = {
    'hap': 0, 'exc': 0,
    'sad': 1,
    'ang': 2,
    'neu': 3,
    'fru': 4,
    'sur': 5, 'fea': 5, 'dis': 5, 'xxx': 5, 'oth': 5
}

data = []

for session in sessions:
    session_path = root + session
    emo_dir = session_path + "/EmoEvaluation"
    wav_dir = session_path + "/wav"
    trans_dir = session_path + "/transcriptions"


    transcripts = {}
    for fname in os.listdir(trans_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(trans_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    match = re.match(r"^(.+?):\s(.+)", line)
                    if match:
                        utt_id = match.group(1).strip().split()[0]
                        text = match.group(2).strip()
                        transcripts[utt_id] = text

    for fname in os.listdir(emo_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(emo_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip().startswith("["):
                        continue

                    try:
                        parts = line.split("\t")
                        if len(parts) >= 3:
                            utt_id = parts[1].strip()
                            emotion = parts[2].strip()

                            if utt_id in transcripts:
                                if emotion in label_map:
                                    data.append({
                                        "utterance_id": utt_id,
                                        "text": transcripts[utt_id],
                                        "label": label_map[emotion]
                                    })
                    except Exception as e:
                        print(f"[ERROR] Errore nel parsing della riga: {line}\n{e}")

df = pd.DataFrame(data)
print("Totale campioni:", len(df))
print(df['label'].value_counts())
df.to_csv("iemocap_text_emotion_6labels.csv", index=False)
