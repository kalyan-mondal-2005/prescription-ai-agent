import pandas as pd
import os

# Paths
INPUT_PATH = "data/processed/clean_data.csv"
OUTPUT_PATH = "data/chunks/chunks.csv"

CHUNK_SIZE = 1


def load_clean_data(path):
    try:
        df = pd.read_csv(path)
        print(" Clean data loaded")
        return df
    except Exception as e:
        print(" Error loading clean data:", e)
        return None


def create_text_representation(df):

    documents = []

    for _, row in df.iterrows():

        text = (
            f"Medicine Name: {row['Medicine_Name']}. "
            f"Strength: {row['Strength']}. "
            f"Used for: {row['Use_Case']}. "
            f"Alternative medicines: {row['Alternative']}. "
            f"Stock Available: {row['Stock']}. "
            f"Dosage: {row['Dosage_Instruction']}."
        )

        documents.append(text)

    return documents


def chunk_documents(documents, chunk_size):

    chunks = []

    for i in range(0, len(documents), chunk_size):

        chunk = " ".join(documents[i:i + chunk_size])

        chunks.append({
            "chunk_id": len(chunks) + 1,
            "text": chunk
        })

    return pd.DataFrame(chunks)


def save_chunks(df, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    print(f" Chunks saved to {path}")


def main():

    df = load_clean_data(INPUT_PATH)

    if df is None:
        return

    documents = create_text_representation(df)

    chunk_df = chunk_documents(documents, CHUNK_SIZE)

    save_chunks(chunk_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()