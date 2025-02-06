import models
import models.query_embedder, models.query_projector, models.vectors
import inference

def main():
    models.vectors.get_vecs()

    query = input("Enter a query (or blank to quit): ")

    if not query:
        print('Goodbye!')
        return

    results = inference.search(query)

    for result in results:
        print(f"- {result['doc_ref']}")

    main()

main()
