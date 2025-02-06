import os
import datasets as hf_datasets
import pandas as pd
import swifter

from util import constants
import models
import models.doc_embedder, models.query_embedder

def _expand_passages(row) -> hf_datasets.Dataset:
    queries = []
    doc_refs = []
    doc_texts = []
    is_selecteds = []
    for i, is_selected in enumerate(row['passages']['is_selected']):
        queries.append(row['query'])
        doc_refs.append(row['passages']['url'][i])
        doc_texts.append(row['passages']['passage_text'][i])
        is_selecteds.append(is_selected)

    return pd.DataFrame({
        'query': queries,
        'doc_ref': doc_refs,
        'doc_text': doc_texts,
        'is_selected': is_selecteds
    })

def main():
    print('Loading dataset...')

    splits = hf_datasets.load_dataset("microsoft/ms_marco", "v1.1")

    all_data = []
    for split in splits:
        rows = splits[split].to_list()
        all_data.extend(rows)
        
    marco = pd.DataFrame(all_data)

    print('Expanding passages...')

    marco = pd.concat(marco.swifter.apply(_expand_passages, axis=1).tolist(), ignore_index=True)

    print('Writing docs to file...')

    docs = marco[['doc_ref', 'doc_text']]

    docs = docs.drop_duplicates(subset=['doc_ref'])

    docs.to_csv(constants.DOCS_PATH, index=False)

    sample_queries = marco[['query']]

    sample_queries = sample_queries.drop_duplicates(subset=['query'])

    sample_queries = sample_queries.sample(1000)

    sample_queries.to_csv(constants.SAMPLE_QUERIES_PATH, index=False)

    marco.to_csv(constants.TRAINING_DATA_PATH, index=False)

    print('Done!')
