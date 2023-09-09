from typing import Any
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import nltk
import os
from tqdm import tqdm
import faiss
import json

class Retriever:
    def __init__(self, encoder_name="BAAI/bge-small-en", encoder_device="cuda:0", index_device="cpu", use_IVF=True, pretrained_index_path=None):
        # TODO: implement reranking
        self.encoder_device = encoder_device
        self.index_device = index_device if isinstance(index_device, int) or index_device == "cpu" else int(index_device.split(":")[-1])
        self.encoder_name = encoder_name
        self.model = None
        self.tokenizer = None
        self.index = None
        self.use_IVF = use_IVF
        self.pretrained_index_path = pretrained_index_path

        self.load_model()
        if self.pretrained_index_path is not None:
            print("Loading pretrained index...", end=" ")
            self.index = faiss.read_index(self.pretrained_index_path)
            print("done.")
        else:
            self.load_embeddings()

    def load_model(self):
        print("Loading model...", end=" ")
        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(self.encoder_name, device_map={"": self.encoder_device})
        print("done.")

    def load_embeddings(self):
        dim = {"BAAI/bge-large-en": 1024, "BAAI/bge-small-en": 384}[self.encoder_name]
        embeddings_dir = f"retrieval/20230601.en.wiki_embeddings_{self.encoder_name.split('/')[-1]}"

        if self.index_device == "cpu":
            self.index = faiss.IndexFlatIP(dim)
        else:                
            config = faiss.GpuIndexFlatConfig()
            config.useFloat16 = True
            config.device = self.index_device
            self.index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dim, config)
        
        if self.use_IVF:
            print("Training IVF index...", end=" ")
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(self.index, dim, nlist)

            n_samples = 5
            files = np.random.choice(os.listdir(embeddings_dir), n_samples)
            vecs = []
            for file in files:
                vecs.append(np.load(os.path.join(embeddings_dir, file)).astype(np.float16))
            vecs = np.concatenate(vecs)
            np.random.shuffle(vecs)
            self.index.train(vecs)
            self.index.nprobe = 5
            print("done.")

        files = os.listdir(embeddings_dir)
        # make sure to add keys to index in order
        files = sorted(files, key=lambda file: int(file.split(".")[0]))
        for file in tqdm(files, desc="Adding embeddings to index"):
            vecs = np.load(os.path.join(embeddings_dir, file)).astype(np.float16)
            self.index.add(vecs)

    def embed(self, texts):
        """Embeds a list of texts using the model."""
        encoded_chunks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_chunks.to(self.encoder_device))
            # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    
    def __call__(self, text: str, k: int = 1, threshold: float = 0.8) -> list:
        """Retrieves k texts from the dataset that are most similar to the input text.
        Args:
            text (str): Text to retrieve similar texts for.
            k (int, optional): Maximum number of texts to retrieve.
            threshold (float, optional): Threshold for the cosine similarity.
        Returns:
            List of texts that are most similar to the input text, sorted by similarity.
        """
        embeddings = self.embed([text])
        scores, indices = self.index.search(embeddings, k)
        scores, indices = scores[0], indices[0]
        indices = indices[scores > threshold]
        scores = scores[scores > threshold]
        retr_texts = [self.get_chunk(i) for i in indices]
        return retr_texts, scores
    
    def get_chunk(self, i: int) -> str:
        """Returns the text at index i from the dataset of wikipedia chunks."""
        batch_size = 200_000
        remainder = i % batch_size
        fname = f"{i - remainder}.json"
        with open(os.path.join("retrieval/wiki_chunks", fname)) as f:
            text = json.load(f)[remainder]
        return text
    
    @staticmethod
    def trim_chunk(chunk):
        """Trims away sentence fragments from the beginning and end of a chunk."""
        chunk = chunk.strip()
        sents = nltk.sent_tokenize(chunk)

        punc_chars = ".!?"
        if sents[-1][-1] not in punc_chars:
            try:
                chunk = chunk.removesuffix(sents[-1])
            except ValueError:
                pass
        try:
            chunk = chunk.removeprefix(sents[0])
        except ValueError:
            pass
        return chunk.strip()
    
    def get_metadata(self):
        return {
            "encoder_name": self.encoder_name,
            "use_IVF": self.use_IVF,
        }
