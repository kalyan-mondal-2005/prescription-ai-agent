import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import ollama

# -----------------------------
# Configuration
# -----------------------------

VECTOR_DB_PATH = "vectorstore"
COLLECTION_NAME = "medicine_embeddings"

TOP_K = 300

# -----------------------------
# Initialize ChromaDB
# -----------------------------

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# -----------------------------
# Search Vector DB
# -----------------------------

def retrieve_context(query):

    results = collection.query(
        query_texts=[query],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    # print(documents)

    return documents


# -----------------------------
# Build Prompt
# -----------------------------

def build_prompt(query, context_docs):

    context = "\n".join(context_docs)

    prompt = f"""
You are a helpful medical shop assistant.

Use the following medicine information to answer the user.

Context:
{context}

User Question:
{query}

Instructions:
- If medicine is available mention stock.
- If out of stock suggest alternatives.
- Give dosage guidance if available.
- Answer clearly and briefly.
"""

    return prompt


# -----------------------------
# Ask Ollama LLM
# -----------------------------

def ask_llm(prompt):

    response = ollama.chat(
        # model="mistral",
        model="phi3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# -----------------------------
# Extra Feature: Stock Check
# -----------------------------

def check_stock(context_docs):

    for doc in context_docs:
        if "Stock Available: Yes" in doc or "Stock: Yes" in doc:
            return "Medicine is available in stock."

    return "Medicine may be out of stock."


# -----------------------------
# Extra Feature: Alternatives
# -----------------------------

def find_alternatives(context_docs):

    alternatives = []

    for doc in context_docs:

        if "Alternative" in doc:

            parts = doc.split("Alternative")

            if len(parts) > 1:
                alt = parts[1].split(".")[0]
                alternatives.append(alt.strip())

    return list(set(alternatives))


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🏥 Hospital & Medical Shop AI Agent")

st.write("Ask about medicines, dosage, alternatives, or availability.")

query = st.text_input("Enter your question:")

if st.button("Ask AI"):

    if query:

        with st.spinner("Searching medicines..."):

            context_docs = retrieve_context(query)

        with st.spinner("Generating response..."):

            prompt = build_prompt(query, context_docs)

            answer = ask_llm(prompt)

        st.subheader("AI Response")

        st.write(answer)

        # Show retrieved documents
        # with st.expander("Retrieved Medicine Information"):

        #     for doc in context_docs:
        #         st.write(doc)

        # Stock info
        st.subheader("Stock Status")

        st.write(check_stock(context_docs))

        # Alternatives
        alts = find_alternatives(context_docs)

        if alts:
            st.subheader("Possible Alternatives")

            for a in alts:
                st.write("-", a)

    else:
        st.warning("Please enter a question.")