#!/bin/sh

# only to be run from docker

PYTHONPATH=src python -m bin.load_to_csv
PYTHONPATH=src python src/server.py
