#import os
import pickle
import faiss
import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)


# In[32]:


model = genai.GenerativeModel("gemini-2.5-flash")


# In[33]:


embedder = SentenceTransformer("all-MiniLM-L6-v2")


# In[34]:


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# In[35]:


def load_pdf_file(path):
    text = ""

    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"❌ Failed to open PDF: {path}")
        print(e)
        return text

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception as e:
            print(f"⚠️ Skipping page {i} due to error:", e)
            continue

    return text


# In[36]:


def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# In[57]:


def build_index(
    file_paths,
    index_file="faiss.index",
    docs_file="docs.pkl"
):
    all_chunks = []

    for path in file_paths:
        # Load file based on type
        if path.endswith(".txt"):
            text = load_text_file(path)
            chunks = chunk_text(text, chunk_size=300, overlap=50)

        elif path.endswith(".pdf"):
            text = load_pdf_file(path)
            chunks = chunk_text(text, chunk_size=150, overlap=30)  # smaller for PDF

        else:
            continue

        # Store chunk WITH SOURCE metadata
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": path
            })

    print(f"Total chunks: {len(all_chunks)}")

    # Create embeddings ONLY from text
    embeddings = embedder.encode(
        [c["text"] for c in all_chunks],
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index + docs (with metadata)
    faiss.write_index(index, index_file)
    with open(docs_file, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Index built and saved.")


# In[58]:


files = [
    "ml_notes.txt",
    "ai_book.txt"
]

build_index(files)


# In[59]:


def retrieve(
    query,
    top_k=5,
    index_file="faiss.index",
    docs_file="docs.pkl",
    source_filter=None
):
    index = faiss.read_index(index_file)

    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k * 3)  # overfetch

    with open(docs_file, "rb") as f:
        docs = pickle.load(f)

    results = []
    for i in indices[0]:
        doc = docs[i]
        if source_filter:
            if source_filter in doc["source"]:
                results.append(doc)
        else:
            results.append(doc)

        if len(results) == top_k:
            break

    return results


# In[60]:


def rag_query(
    query,
    top_k=3,
    index_file="faiss.index",
    docs_file="docs.pkl"
):
    # 🔥 FIRST: try PDF only
    retrieved_docs = retrieve(
        query,
        top_k=top_k,
        index_file=index_file,
        docs_file=docs_file,
        source_filter="ai_book.pdf"
    )

    # Fallback: all docs
    if not retrieved_docs:
        retrieved_docs = retrieve(
            query,
            top_k=top_k,
            index_file=index_file,
            docs_file=docs_file
        )

    context = "\n".join([d["text"] for d in retrieved_docs])

    prompt = f"""
Answer the question using only the following context.
If the context is not enough, say you don't know.

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)   # replace with your actual LLM call

    return response.text,retrieved_docs


# In[61]:


prompt, docs = rag_query("Types of Artificial Intelligence")

for d in docs:
    print(d["source"])
    print(d["text"][:200])
    print("-" * 50)

print(prompt)


# In[ ]:





# In[ ]:




