import pandas as pd
import torch
import numpy as np
import swifter

from util import artifacts, constants, chroma, devices
import models
import models.doc_embedder, models.doc_projector, models.vectors
import dataset

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

    def get_doc_encoding(row):
        doc_embeddings = models.doc_embedder.get_embeddings_for_doc(row['doc_text'])

        batch, lengths = dataset.pad_batch_values([doc_embeddings])

        encoded, _ = doc_projector(batch, lengths)
        encoded_list = encoded.detach().tolist()

        if (len(encoded_list) > 1):
            raise ValueError(f"Expected 1 encoded vector, got {len(encoded_list)}")
        
        encoded_item = encoded_list[0]

        return pd.Series({
            'doc_ref': row['doc_ref'],
            'doc_embedding': encoded_item
        })

    collection = chroma.client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

    BATCH_SIZE = 1000
    num_of_batches = len(data) // BATCH_SIZE
    batches = np.array_split(data, num_of_batches)
    for index, batch in enumerate(batches):
        print(f"Encoding batch {index} of {len(batches)}")

        batch = batch.swifter.apply(get_doc_encoding, axis=1)

        print('Storing encodings for batch...')

        collection.add(
            ids=batch['doc_ref'].tolist(),
            embeddings=batch['doc_embedding'].tolist()
        )

        print('> Done')
