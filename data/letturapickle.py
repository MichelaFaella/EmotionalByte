import pickle
import pprint

with open("data/iemocap_multimodal_features_6labels.pkl", "rb") as f:
    objects = pickle.load(f)
    pprint.pprint(objects)


