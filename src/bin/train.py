import os
from pathlib import Path
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing
import torch
import tqdm
from typing import TypedDict
import swifter
import numpy as np
import wandb
import sklearn
import itertools
import spacy

import models
import models.doc_projector, models.query_projector, models.doc_embedder, models.query_embedder, models.vectors
import dataset
import inference
from util import devices, artifacts, constants

EPOCHS = 100
LEARNING_RATE = 0.0002
MARGIN = 0.2
BATCH_SIZE = 64
EARLY_STOP_AFTER = 3

torch.manual_seed(16)

def main():
    data = pd.read_csv(constants.TRAINING_DATA_PATH)

    print(f"INFO: Running for {len(data)} training rows")

    wandb.init(project='search', name='search')
    wandb.config = {
        "training_data_size": len(data),
        "learning_rate": LEARNING_RATE,
        "query_hidden_layer_dimensions": models.query_projector.QUERY_HIDDEN_LAYER_DIMENSION,
        "doc_hidden_layer_dimensions": models.doc_projector.DOC_HIDDEN_LAYER_DIMENSION,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS
    }

    device = devices.get_device()

    print(f"INFO: Using device: {device.type}")

    train, val = sklearn.model_selection.train_test_split(data, test_size=0.2, random_state=16)

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    train_loader = torch.utils.data.DataLoader(dataset.TwoTowerDataset(train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.collate_two_tower_batch)
    val_loader = torch.utils.data.DataLoader(dataset.TwoTowerDataset(val), batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.collate_two_tower_batch)

    query_projector = models.query_projector.Model().to(device)
    doc_projector = models.doc_projector.Model().to(device)

    calc_loss = torch.nn.TripletMarginWithDistanceLoss(margin=MARGIN, distance_function=lambda query, doc: 1 - torch.nn.functional.cosine_similarity(query, doc)).to(device)

    all_params = list(query_projector.parameters()) + list(doc_projector.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LEARNING_RATE)

    val_loss_failed_to_improve_for_epochs = 0
    best_val_loss = float('inf')
    best_query_state_dict = None
    best_doc_state_dict = None
    
    models.vectors.get_vecs()

    def get_epoch_weight_path(epoch, query_or_doc):
        return os.path.join(constants.DATA_PATH, f"epoch-weights/{query_or_doc}-weights_epoch-{epoch + 1}.generated.pt")
    
    val_doc1_samples = val.sample(100, random_state=1)
    val_doc2_samples = val.sample(100, random_state=2)
    val_samples = list(zip(val_doc1_samples.iterrows(), val_doc2_samples.iterrows()))
    val_samples = [(row1[1], row2[1]) for row1, row2 in val_samples]

    print('Initializing spacy for validation')
    nlp = spacy.load('en_core_web_lg')
    print('> Done')

    for epoch in range(EPOCHS):

        query_projector.train()
        doc_projector.train()

        train_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):

            optimizer.zero_grad()

            query_outputs, _ = query_projector(batch['query_embeddings'], batch['query_embedding_lengths'])
            relevant_doc_outputs, _ = doc_projector(batch['relevant_doc_embeddings'], batch['relevant_doc_embedding_lengths'])
            irrelevant_doc_outputs, _ = doc_projector(batch['irrelevant_doc_embeddings'], batch['irrelevant_doc_embedding_lengths'])

            loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)

        query_projector.eval()
        doc_projector.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                query_outputs, _ = query_projector(batch['query_embeddings'], batch['query_embedding_lengths'])
                relevant_doc_outputs, _ = doc_projector(batch['relevant_doc_embeddings'], batch['relevant_doc_embedding_lengths'])
                irrelevant_doc_outputs, _ = doc_projector(batch['irrelevant_doc_embeddings'], batch['irrelevant_doc_embedding_lengths'])

                loss = calc_loss(query_outputs, relevant_doc_outputs, irrelevant_doc_outputs)

                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)

        diffs = []
        for row1, row2 in val_samples:
            # compare the cosine similarity from the trained model to the similarity returned by a pretrained model from spacy
            row1_doc_output = inference.get_doc_encoding(doc_projector, row1['doc_text'])
            row2_doc_output = inference.get_doc_encoding(doc_projector, row2['doc_text'])
            doc_output_similarity = torch.nn.functional.cosine_similarity(torch.tensor([row1_doc_output]), torch.tensor([row2_doc_output])).item()
            doc_baseline_similarity = nlp(row1['doc_text']).similarity(nlp(row2['doc_text']))
            diffs.append(abs(doc_output_similarity - doc_baseline_similarity))

        avg_diff = sum(diffs) / len(diffs)

        print(f"Epoch {epoch + 1}, train loss: {round(train_loss, 6)}, val loss: {round(val_loss, 6)}, difference from baseline model: {round(avg_diff, 4)}")

        wandb.log({ 'epoch': epoch + 1, 'train-loss': train_loss, 'val_loss': val_loss })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_query_state_dict = query_projector.state_dict()
            best_doc_state_dict = doc_projector.state_dict()
            val_loss_failed_to_improve_for_epochs = 0

            Path(os.path.join(constants.DATA_PATH, "epoch-weights")).mkdir(exist_ok=True)
            torch.save(best_query_state_dict, get_epoch_weight_path(epoch, 'query'))
            torch.save(best_doc_state_dict, get_epoch_weight_path(epoch, 'doc'))
        else:
            val_loss_failed_to_improve_for_epochs += 1

        if val_loss_failed_to_improve_for_epochs == EARLY_STOP_AFTER:
            print(f"Validation loss failed to improve for {EARLY_STOP_AFTER} epochs. Early stopping now.")
            break

    query_model_save_path = os.path.join(constants.DATA_PATH, 'query-projector-weights.generated.pt')
    torch.save(best_query_state_dict, query_model_save_path)
    artifacts.store_artifact('query-projector-weights', 'model', query_model_save_path)

    doc_model_save_path = os.path.join(constants.DATA_PATH, 'doc-projector-weights.generated.pt')
    torch.save(best_doc_state_dict, doc_model_save_path)
    artifacts.store_artifact('doc-projector-weights', 'model', doc_model_save_path)

    wandb.finish()
