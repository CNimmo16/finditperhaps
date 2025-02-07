#!/bin/bash

DIR="$(dirname "$(realpath "$0")")"

pdm export --no-hashes --format=requirements --output "$DIR/requirements.txt"

docker build --platform="linux/amd64" -t "cameronnimmo/ml-search" $DIR

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Build failed with exit code $exit_code. See above"
    exit $exit_code
fi

docker push cameronnimmo/ml-search
