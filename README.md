# MLX course

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

## Running scripts

Run `pdm run <script-from-pyproject.toml>`
