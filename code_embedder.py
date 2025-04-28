from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from similarity_functions import cosine_similarity

# --- Embedders ---
class CodeEmbedder:
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).to(
            self.device)
        self.max_length = max_length
        # default sim_fn
        self.sim_fn = cosine_similarity

    def get_embedding(self, code_text):
        inputs = self.tokenizer(code_text, return_tensors="pt", truncation=True,
                                max_length=self.max_length, padding="max_length").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.cpu().numpy()[0]