import pandas as pd
import os
import chromadb
from chromadb.utils import embedding_functions

CHUNKS_PATH = "data/chunks/chunks.csv"
VECTOR_DB_DIR = "vectorstore"
COLLECTION_NAME = "medicine_embeddings"

BATCH_SIZE = 5000


def load_chunks():
    df = pd.read_csv(CHUNKS_PATH)
    print("Chunks loaded:", len(df))
    return df


def create_vector_store(texts, ids):

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # Insert in batches
    for i in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]

        collection.add(
            documents=batch_texts,
            ids=batch_ids
        )

        print(f"Inserted batch {i} → {i + len(batch_texts)}")


def main():

    df = load_chunks()

    texts = df["text"].tolist()
    ids = df["chunk_id"].astype(str).tolist()

    create_vector_store(texts, ids)


if __name__ == "__main__":
    main()