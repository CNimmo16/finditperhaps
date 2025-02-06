try:
    # handles outdated sqlite3 in docker image
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    # if no pysqlite3 installed via pip assume default sqlite3 is up to date enough for chroma
    pass

import os
import chromadb

CHROMA_HOST = os.environ.get('CHROMA_HOST', 'localhost')
CHROMA_PORT = os.environ.get('CHROMA_PORT', '8000')

client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
client.heartbeat()
