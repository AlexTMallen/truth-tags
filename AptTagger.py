from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
import pandas as pd
from typing import Iterable

class AptTagger:
    def __init__(self, model_name="data/apt-tagger-deberta-xlarge-mnli", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
        self.device = device

    def tag(self, text, return_mask=True):
        """
        texts: list of strings or string
        returns: [list of] masks, where each mask is a list of 1s and 0s of length
                `len(text)`, where 1 indicates a tag at the token ending at that index
                e.g. "I run." -> [0 0 0 0 1 0], corresponding to "I run[[APT]]."
        """
        if not isinstance(text, str):
            if isinstance(text, Iterable):
                return [self.tag(t) for t in text]
            else:
                raise ValueError(f"Expected string or  of strings, got {type(text)}")
        
        probs, offset_mapping = self._predict(text)
        preds_by_tok = probs.argmax(axis=-1).flatten()
        tag_mask = self._tokens_to_chars(preds_by_tok, offset_mapping, text)
        if return_mask:
            return tag_mask
        return self.mask_to_idxs(tag_mask)
        
    def _predict(self, text):
        encodings = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt").to(self.device)
        offset_mapping = encodings.pop("offset_mapping")[0]
        with torch.no_grad():
            output = self.model(**encodings)
        logits = output.logits
        probs = torch.softmax(logits, dim=2).cpu().numpy()
        return probs, offset_mapping
    
    def _tokens_to_chars(self, preds_by_tok, offset_mapping, text):
        preds_by_char = np.zeros(len(text), dtype=int)
        for pred_by_tok, (start, end) in zip(preds_by_tok, offset_mapping):
            if pred_by_tok != 0:
                if start == end:
                    print("Warning: token with no characters tagged as APT, skipping")
                preds_by_char[end - 1] = pred_by_tok
        return preds_by_char
    
    @staticmethod
    def mask_to_idxs(mask):
        """ Returns a list of indices where mask is 1 """
        return np.where(mask == 1)[0].tolist()
    
    @staticmethod
    def annotate(text, mask):
        """ Inserts labels starting from [[1]] at each index where mask is 1 """
        bits = []
        num_apt = 0
        for i, c in enumerate(text):
            if c == ")" and mask[i] != 1 and i != 0 and mask[i - 1] == 1:
                # if the previous char tag comes right before a closing paren, move it outside,
                # unless there is another tag right after the closing paren
                # NOTE: This is a hack, otherwise LMs get confused what span the tag is for
                bits.insert(-1, c)  # insert ")" before the last apt tag
            else:
                bits.append(c)
                
            if mask[i] == 1:
                num_apt += 1
                
                bits.append(f"[[{num_apt}]]")
        return "".join(bits)