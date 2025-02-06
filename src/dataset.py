import torch
import pandas as pd
import numpy as np
import swifter
import joblib
from pathlib import Path
import os

import models
import models.query_embedder, models.doc_embedder
from util import devices, mini

dirname = os.path.dirname(__file__)

CHUNK_SIZE = 10000

device = devices.get_device()

class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.prepped = {}

    def __len__(self):
        return len(self.data.index)
    
    def __get_irrelevant_doc_embedding(self, row):
        for i in range(0, 100):
            other_row = self.data.sample(1).iloc[0]
            if (other_row['query'] != row['query']) and (other_row['doc_ref'] != row['doc_ref']):
                return models.doc_embedder.get_embeddings_for_doc(other_row['doc_text'])
        raise ValueError("No non-relevant result found in random 100 row sample. Weird.")

    def __prepare_row(self, row):
        return pd.Series({
            'query_embeddings': models.query_embedder.get_embeddings_for_query(row['query']),
            'relevant_doc_embeddings': models.doc_embedder.get_embeddings_for_doc(row['doc_text']),
            'irrelevant_doc_embeddings': self.__get_irrelevant_doc_embedding(row)
        })
    
    def __get_chunk(self, chunk_idx: int):
        if chunk_idx not in self.prepped:
            Path(os.path.join(dirname, './data/chunks')).mkdir(parents=True, exist_ok=True)
            chunk_filepath = os.path.join(dirname, f"./data/chunks/chunk-{chunk_idx}.generated.pt")
            try:
                if mini.is_mini():
                    raise FileNotFoundError('CACHE MISS: Mini mode, not loading')
                chunk = joblib.load(chunk_filepath)
                print('CACHE HIT: Got existing chunk from file...')
                self.prepped[chunk_idx] = chunk
                return chunk
            except:
                print(f"Preloading data chunk {chunk_idx}...")
                rows = self.data[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
                chunk = rows.swifter.apply(self.__prepare_row, axis=1)
                self.prepped[chunk_idx] = chunk
                joblib.dump(chunk, chunk_filepath)
                print('> Done')
        return self.prepped[chunk_idx]

    def __getitem__(self, idx: int) -> dict[str, str]:
        chunk_idx = idx // CHUNK_SIZE

        chunk = self.__get_chunk(chunk_idx)

        idx_in_chunk = idx % CHUNK_SIZE
        
        return chunk.iloc[idx_in_chunk]

def pad_batch_values(values: list):
    if type(values) is not list:
        raise ValueError("Input values must be a list (batches)")
    if type(values[0]) is not list:
        raise ValueError(f"Input values must be a list (batches) of lists (tokens in batch), got type {type(values[0])} at values[0]")
    if type(values[0][0]) is not np.ndarray:
        raise ValueError(f"Input values must be a list (batches) of lists (tokens in batch) of numpy arrays (token embeddings), got {type(values[0][0])} at values[0][0]")
    if not isinstance(values[0][0], np.ndarray) or values[0][0].dtype != np.float32:
        raise ValueError(f"Input values must be a list (batches) of lists (tokens in batch) of numpy arrays of dtype float32 (token embeddings), got dtype {type(values[0][0].dtype)}")
    tensors = [torch.tensor(np.array(item)).to(device) for item in values]

    padded_values = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0).float()
    original_lengths = torch.tensor([len(x) for x in values])

    return padded_values, original_lengths

def collate_two_tower_batch(batch: list):
    query_embeddings, query_embedding_lengths = pad_batch_values([item['query_embeddings'] for item in batch])
    relevant_doc_embeddings, relevant_doc_embedding_lengths = pad_batch_values([item['relevant_doc_embeddings'] for item in batch])
    irrelevant_doc_embeddings, irrelevant_doc_embedding_lengths = pad_batch_values([item['irrelevant_doc_embeddings'] for item in batch])

    return {
        'query_embeddings': query_embeddings,
        'query_embedding_lengths': query_embedding_lengths,
        'relevant_doc_embeddings': relevant_doc_embeddings,
        'relevant_doc_embedding_lengths': relevant_doc_embedding_lengths,
        'irrelevant_doc_embeddings': irrelevant_doc_embeddings,
        'irrelevant_doc_embedding_lengths': irrelevant_doc_embedding_lengths
    }
