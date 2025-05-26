import torch
import pandas as pd
from transformers import pipeline
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.eval()

@torch.no_grad()
def create_bio_embedding(description):

    inputs = tokenizer(description, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs)
    h_desc = outputs.last_hidden_state[:, 0, :]  # CLS token

    return h_desc.squeeze(0)

def build_speaker_description_embeddings(prompt_csv_path):
    prompt_df = pd.read_csv(prompt_csv_path, delimiter=";", engine="python")
    prompt_df.columns = prompt_df.columns.str.strip()  
    prompt_df = prompt_df.dropna(subset=["dialogue_id", "speaker", "description"])
    
    speaker_embeddings = {}

    for row in prompt_df.itertuples():
        dialogue_id = getattr(row, "dialogue_id").strip()
        speaker = getattr(row, "speaker").strip()
        description = getattr(row, "description").strip()

        h_desc = create_bio_embedding(description)
        
        # Store the vector for later use in the model
        speaker_embeddings[(dialogue_id, speaker)] = h_desc.detach()  # detach to avoid keeping graph

    return speaker_embeddings

# Precompute the speaker descriptions
speaker_embeddings = build_speaker_description_embeddings(prompt_csv_path="src/biosERC/iemocap_spk_bios.csv")

torch.save(speaker_embeddings, "src/biosERC/speaker_bios.pt")