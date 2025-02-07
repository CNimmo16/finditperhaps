import pandas as pd
import torch

from util import artifacts, constants, chroma, devices
import models
import dataset
import models.query_embedder, models.query_projector, models.doc_projector, models.vectors

MAX_RESULTS = 5

device = devices.get_device()

query_projector = None
docs = None

def load_model_and_docs():
    global query_projector, docs
    if query_projector is None or docs is None:
        query_projector = models.query_projector.Model().to(device)

        query_state_dict = artifacts.load_artifact('query-projector-weights', 'model')

        query_projector.load_state_dict(query_state_dict)
        query_projector.eval()

        docs = pd.read_csv(constants.DOCS_PATH)

    return query_projector, docs

def get_random_query():
    sample_queries = pd.read_csv(constants.SAMPLE_QUERIES_PATH)

    query = sample_queries.sample(1).iloc[0]['query']

    return query

def get_doc_encoding(doc_projector: models.doc_projector.Model, doc_text: str):
    doc_embeddings = models.doc_embedder.get_embeddings_for_doc(doc_text)

    batch, lengths = dataset.pad_batch_values([doc_embeddings])

    encoded, _ = doc_projector(batch, lengths)
    encoded_list = encoded.detach().tolist()

    if (len(encoded_list) > 1):
        raise ValueError(f"Expected 1 encoded vector, got {len(encoded_list)}")
    
    encoded_item = encoded_list[0]

    return encoded_item

def search(query: str):
    query_projector, docs = load_model_and_docs()

    query_embeddings = models.query_embedder.get_embeddings_for_query(query)

    batch = [query_embeddings]

    padded_embeddings, lengths = dataset.pad_batch_values(batch)

    encoded_query_batch, _ = query_projector(padded_embeddings, lengths)
    
    encoded_query_batch_list = encoded_query_batch.detach().tolist()

    collection = chroma.client.get_collection(name="docs")
    nearest_docs = collection.query(
        query_embeddings=encoded_query_batch_list,
        n_results=5,
    )
    nearest_doc_refs = nearest_docs['ids'][0]
    nearest_docs = [docs[docs['doc_ref'] == id].iloc[0].to_dict() for id in nearest_doc_refs]

    return nearest_docs
