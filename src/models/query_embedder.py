import nltk
import numpy as np
from models import vectors

nltk.download('punkt_tab')

EMBEDDING_DIM = vectors.EMBEDDING_DIM

def get_embeddings_for_query(query: str) -> list:
    word_vectors = vectors.get_vecs()
    tokens = nltk.word_tokenize(query)
    embeddings = [word_vectors[token] if token in word_vectors else word_vectors['<UNK>'] for token in tokens]
    return embeddings
