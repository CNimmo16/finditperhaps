from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Annotated
import logging

from bin.cache_docs import main as cache_docs
import inference
from models import vectors
from util import chroma

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(dirname, "web/templates"))

static_dir = os.path.join(dirname, "web/static")

app.mount(static_dir, StaticFiles(directory=static_dir), name="static")

vectors.get_vecs()

print('Checking if docs have been cached...')
try:
    docs = chroma.client.get_collection(name="docs")
    count = docs.count()
    print(f"Docs collection already exists, skipping caching. (doc count: {count})")
except Exception:
    print('Docs not cached. Storing vectors now.')
    cache_docs()
    print('> Done')

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.j2"
    )

@app.post("/search", response_class=RedirectResponse)
async def root(request: Request, query: Annotated[str, Form()]):
    return RedirectResponse(url=f"/results?query={query}", status_code=302)

@app.get("/results", response_class=HTMLResponse)
async def root(request: Request, query: str):
    try:
        results = inference.search(query)
        for result in results:
            result['summary'] = result['doc_text'].replace("\\r\\n", "")[0:200]
        return templates.TemplateResponse(
            request=request, name="index.j2", context={"query": query, "results": results}
        )
    except Exception:
        logging.exception("Query failed")
        return templates.TemplateResponse(
            request=request, name="index.j2", context={"query": query, "error": "Something went wrong"}
        )


@app.get("/lucky", response_class=RedirectResponse)
async def root(request: Request):
    query = inference.get_random_query()

    return RedirectResponse(url=f"/results?query={query}", status_code=302)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
