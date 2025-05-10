import pickle
import pprint

with open("IEMOCAP_features.pkl", "rb") as f:
    objects = pickle.load(f)
    pprint.pprint(objects)


