# Finditperhaps

Document search and retrieval with deep learning (part of ML institute programme)

## Setup

1. Ensure Python is installed on the system. If you need a specific version eg 3.9, run:
```bash
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install python3.9
```
2. Install PDM with `curl -sSL https://pdm-project.org/install-pdm.py | python3 -` (you may need to add to your path after doing this to allow running the `pdm` command, see output of installation script for details)
3. Point PDM to a python interpreter - for example if installed python 3.9 in step 1 run `pdm use python3.9`. PDM will automatically create a virtual environment in the `.venv` folder
4. Run `source .venv/bin/activate` to activate the virtual environment
5. Run `pdm install` to install dependencies

## Training locally

1. Run `pdm run load` to preprocess the dataset into a csv
2. Run `pdm run train` to train the model in minimode

## Training on a GPU

1. Run `./ssh.sh`, providing ip and port when prompted to open vscode on the GPU remotely
2. Follow steps from [setup](#setup) to install PDM and python on the GPU
3. Run `FULLRUN=1 pdm run load` to preprocess the dataset into a csv
4. Run `FULLRUN=1 pdm run train` to train the model in full mode with all the data

## Running inference locally

1. Run `docker compose -f docker-compose.dev.yml up` to spin up the chroma (vector database) instance
2. Run `pdm run cache` to run a script that stores the encoded vectors for each document in chroma
3. Run `pdm run serve` to launch the web server. It should open on http://localhost:8080

### Overriding the weights used

By default inference is run using model weights downloaded from wandb (see `src/util/artifacts.py`). Override these by setting env variables, for example to override the weights for the projector during caching you could run `DOC_PROJECTOR_WEIGHTS_PATH=data/epoch-weights/doc-weights_epoch-30.generated.pt pdm run cache`

## Deployment

1. Run `pdm run build` to build the server docker image and push it to docker hub
2. Open `inventory.ini` and update to reflect the ip and port of the server you want to deploy to
3. Run `pdm run ansible` to run the `playbook.yml` file which should ssh to the remote server and launch a chroma instance and the server. 

Note that the server will take a long time to startup because it needs to run the document caching logic from `pdm run cache` to insert the document vectors into chroma before it can handle requests. You can check its progress by ssh-ing to the server and running `sudo docker compose -f /root/mlx/docker-compose.yml logs server`
