import pandas as pd
import torch
import numpy as np
import swifter

from util import artifacts, constants, chroma, devices
import models
import models.doc_embedder, models.doc_projector, models.vectors
import inference

device = devices.get_device()

def main():
    torch.no_grad()

    models.vectors.get_vecs()

    print('Loading data...')

    data = pd.read_csv(constants.DOCS_PATH)

    print('Removing duplicate documents...')

    data = data.drop_duplicates(subset=['doc_ref'])

    print('Loading doc projector...')

    doc_projector = models.doc_projector.Model().to(device)

    doc_state_dict = artifacts.load_artifact('doc-projector-weights', 'model')

    doc_projector.load_state_dict(doc_state_dict)
    doc_projector.eval()

    print('Deleting existing cache...')

    collection = chroma.client.get_or_create_collection(name="docs")

    chroma.client.delete_collection(name="docs")

    print('Encoding documents...')

    collection = chroma.client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

    BATCH_SIZE = 1000
    num_of_batches = len(data) // BATCH_SIZE
    batches = np.array_split(data, num_of_batches)
    for index, batch in enumerate(batches):
        batch = batch.swifter.progress_bar(enable=True, desc=f"Encoding batch {index} of {len(batches)}").apply(lambda row: pd.Series({
            'doc_ref': row['doc_ref'],
            'doc_embedding': inference.get_doc_encoding(doc_projector, row['doc_text'])
        }), axis=1)

        print('Storing encodings for batch...')

        collection.add(
            ids=batch['doc_ref'].tolist(),
            embeddings=batch['doc_embedding'].tolist()
        )

        print('> Done')
