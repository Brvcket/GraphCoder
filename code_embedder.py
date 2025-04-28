from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

class CodeEmbedder:
    def __init__(self, model_name="Salesforce/codet5p-110m-embedding", max_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # we ask for return_dict so we can inspect the fields if it's a ModelOutput
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            return_dict=True,
            output_hidden_states=False
        ).to(self.device)
        self.max_length = max_length
        self.sim_fn = None

    def get_embedding(self, code_text: str) -> np.ndarray:
        # 1) Tokenize
        inputs = self.tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        ).to(self.device)

        # 2) Forward
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3) Peel out the tensor
        if isinstance(outputs, torch.Tensor):
            # e.g. Salesforce/codet5p-110m-embedding returns a plain tensor [B, D]
            emb_tensor = outputs
        elif getattr(outputs, "pooler_output", None) is not None:
            # e.g. many classification-style Transformers
            emb_tensor = outputs.pooler_output
        else:
            # fallback: mean‐pool the last hidden states
            last_hidden = outputs.last_hidden_state        # [B, L, D]
            mask = inputs.attention_mask.unsqueeze(-1)     # [B, L, 1]
            summed = (last_hidden * mask).sum(dim=1)       # [B, D]
            counts = mask.sum(dim=1).clamp(min=1e-9)       # [B, 1]
            emb_tensor = summed / counts                  # [B, D]

        # 4) To CPU + numpy + take first batch item
        return emb_tensor.cpu().numpy()[0]
