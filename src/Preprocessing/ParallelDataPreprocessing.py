import os
import pickle
import opensmile
import torch
from transformers import RobertaTokenizer, RobertaModel
from concurrent.futures import ProcessPoolExecutor


class MultimodalDatasetProcessor:
    def __init__(self, root, label_map):
        self.root = root
        self.label_map = label_map

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.model.eval()

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )

    def tokenize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        return outputs.numpy()

    def extract_audio_features(self, wav_path):
        features = self.smile.process_file(wav_path)
        return features.iloc[0].to_numpy()

    def process_session(self, session):
        raise NotImplementedError("This method must be implemented in subclasses.")

    def save_features(self, path, data):
        with open(path, "wb") as f:
            pickle.dump(data, f)


class IEMOCAPProcessor(MultimodalDatasetProcessor):
    def __init__(self, root, label_map):
        super().__init__(root, label_map)
        self.sessions = [f"Session{i}" for i in range(1, 6)]

    def process_session(self, session):
        print(f"Processing {session}")
        session_path = os.path.join(self.root, session)
        emo_dir = os.path.join(session_path, "EmoEvaluation")
        wav_dir = os.path.join(session_path, "wav")
        trans_dir = os.path.join(session_path, "transcriptions")

        videoText, videoAudio, videoSpeakers, videoLabels = {}, {}, {}, {}
        vids = []

        for emo_file, wav_fold in zip(os.listdir(emo_dir), os.listdir(wav_dir)):
            if not emo_file.endswith(".txt"):
                continue
            print(f"{emo_file}")
            wav_fold_path = os.path.join(wav_dir, wav_fold)
            idx = 0

            with open(os.path.join(emo_dir, emo_file)) as f:
                for line in f:
                    if line.startswith("["):
                        parts = line.strip().split()
                        vid = parts[3]
                        emotion = parts[4]
                        if emotion not in self.label_map:
                            continue

                        # Text
                        text = None
                        with open(os.path.join(trans_dir, emo_file), encoding="utf8") as tf:
                            for tline in tf:
                                if vid in tline:
                                    text = tline.split(":", 1)[-1].strip()
                                    break
                        if text is None:
                            continue
                        videoText[vid] = self.tokenize_text(text)

                        # Audio
                        try:
                            wav = os.listdir(wav_fold_path)[idx]
                            wav_path = os.path.join(wav_fold_path, wav)
                            videoAudio[vid] = self.extract_audio_features(wav_path)
                            idx += 1
                        except Exception as e:
                            print(f"Audio error {vid}: {e}")
                            continue

                        # Speaker and label
                        speaker = 'M' if vid[7] == 'M' else 'F'
                        videoSpeakers[vid] = speaker
                        videoLabels[vid] = self.label_map[emotion]
                        vids.append(vid)

        print(f"Found {len(vids)} videos in {session}")
        return session, vids, videoText, videoAudio, videoSpeakers, videoLabels

    def process_all_sessions(self, save_path="iemocap_multimodal_features_par.pkl"):
        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.process_session, self.sessions))

        results.sort(key=lambda x: x[0])
        videoText, videoAudio, videoSpeakers, videoLabels = {}, {}, {}, {}
        trainVid, testVid = [], []

        for session, vids, vText, vAudio, vSpeakers, vLabels in results:
            videoText.update(vText)
            videoAudio.update(vAudio)
            videoSpeakers.update(vSpeakers)
            videoLabels.update(vLabels)
            if session != "Session5":
                trainVid.extend(vids)
            else:
                testVid.extend(vids)

        print("Number of video in train:", len(trainVid))
        print("Number of video in test:", len(testVid))

        self.save_features(save_path, (
            list(videoText.keys()), videoSpeakers, videoLabels, videoText,
            None, None, None, videoAudio, trainVid, testVid
        ))


if __name__ == "__main__":
    label_map = {
        'hap': 0, 'sad': 1, 'ang': 2, 'neu': 3, 'fru': 4, 'exc': 5,
        'sur': 6, 'fea': 7, 'dis': 8, 'xxx': 9, 'oth': 10
    }
    processor = IEMOCAPProcessor(root="IEMOCAP", label_map=label_map)
    processor.process_all_sessions()

