FROM python:3.11-slim-bullseye

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

# latest sqlite3 for chroma
RUN pip install --no-cache-dir pysqlite3-binary

COPY . /code

EXPOSE 8000

CMD PYTHONPATH=src python -m bin.load_to_csv && python src/server.py
