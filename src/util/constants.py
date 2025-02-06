import os
from util import mini

dirname = os.path.dirname(__file__)

DATA_PATH = os.path.join(dirname, "../../data")
DOCS_PATH = os.path.join(DATA_PATH, "docs.generated.csv")
TRAINING_DATA_PATH = os.path.join(DATA_PATH, "training-data.generated.csv")
SAMPLE_QUERIES_PATH = os.path.join(DATA_PATH, "sample-queries.generated.csv")
