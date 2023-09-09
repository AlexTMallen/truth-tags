import numpy as np
import json
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import torch
import os

import torch.multiprocessing as mp


chunks_dir = "retrieval/wiki_chunks/"

def make_chunks():
    os.mkdir(chunks_dir)
    print("Splitting wiki chunks...", end=" ")
    ds = load_dataset("graelo/wikipedia", "20230601.en", split="train")
    wiki_chunks = []
    for i in tqdm(range(len(ds))):
        text = ds[i]["text"]
        words = text.split(" ")
        chunk_size = 100
        for j in range(0, len(words), chunk_size):
            wiki_chunks.append(" ".join(words[j:j+chunk_size]))
    print("done.")

    batch_size = 200_000
    batches = np.array_split(wiki_chunks, len(wiki_chunks) // batch_size)
    for i, batch in tqdm(enumerate(batches)):
        with open(chunks_dir + f"{i}.json", "w") as f:
            json.dump(batch.tolist(), f)


def embed_wikipedia(model_name, devices, batch_size):
    world_size = len(devices)
    if not os.path.exists(chunks_dir):
        make_chunks()
    mp.spawn(embed_wikipedia_worker, args=(model_name, devices, batch_size, world_size), nprocs=world_size, join=True)


def embed_wikipedia_worker(rank, model_name, devices, batch_size, world_size):
    
    
    device = torch.device(f"cuda:{devices[rank]}")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device})

    
    save_interval = 5  # batches

    for fname in tqdm(os.listdir(chunks_dir)[rank::world_size], position=rank):
        batch_num = int(fname.split(".")[0])
        embed_path = f"retrieval/20230601.en.wiki_embeddings_{model_name.split('/')[-1]}/{batch_num}.npy"
        path = os.path.join(chunks_dir, fname)
        with open(path) as f:
            wiki_chunks = json.load(f)
            
        sentence_embeddings = np.zeros((len(wiki_chunks), model.config.hidden_size))
        for i in range(0, len(wiki_chunks), batch_size):
            encoded_chunks = tokenizer(wiki_chunks[i:i + batch_size], padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = model(**encoded_chunks.to(device))
                # Perform pooling. In this case, cls pooling.
                embeddings = model_output[0][:, 0]
            # normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sentence_embeddings[i:i + batch_size, :] = embeddings.cpu().numpy()

            if (i // batch_size) % save_interval == 0:
                np.save(embed_path, sentence_embeddings.astype(np.float16))

        np.save(embed_path, sentence_embeddings.astype(np.float16))
    print(f"Process {rank} finished and saved embeddings to {embed_path}")

if __name__ == "__main__":
    os.chdir("/mnt/ssd-2/spar/alexm/dlk-benchmarking/auto-labeling")
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3], required=False)
    parser.add_argument("--model-name", default="BAAI/bge-large-en", type=str, required=False)
    parser.add_argument("--batch-size", default=256, type=int, required=False)
    args = parser.parse_args()
    
    embed_wikipedia(args.model_name, args.devices, args.batch_size)
