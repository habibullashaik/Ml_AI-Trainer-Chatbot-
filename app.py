import streamlit as st

# Import YOUR existing functions (unchanged)
from core_logic import build_index, rag_query

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📘 AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🤖 AI Knowledge Assistant</h1>
    <p style='text-align: center; font-size:18px;'>
    Ask questions from your own <b>PDF & TXT knowledge base</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

if st.sidebar.button("🔄 Build / Rebuild Knowledge Index"):
    with st.spinner("Indexing documents..."):
        build_index([
            "ml_notes.txt",
            "ml_again_notes.txt",
            "ai_book.pdf"
        ])
    st.sidebar.success("Index built successfully!")

st.sidebar.markdown("---")
st.sidebar.markdown("📂 **Data Sources**")
st.sidebar.markdown("- TXT Notes")
st.sidebar.markdown("- AI Book (PDF)")

# ---------------- MAIN UI ----------------
query = st.text_input(
    "💬 Ask a question from your knowledge base:",
    placeholder="e.g. What is Machine Learning?"
)

col1, col2 = st.columns([1, 1])

if st.button("🚀 Ask AI"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            prompt, docs = rag_query(query)

        # -------- ANSWER --------
        st.subheader("🧠 Answer")
        st.markdown(prompt.split("Question:")[0].strip())

        # -------- SOURCES --------
        with st.expander("📚 Retrieved Sources"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Source {i}: {d['source']}**")
                st.write(d["text"][:500] + "...")
                st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    Built with ❤️ using Streamlit + FAISS + RAG
    </p>
    """,
    unsafe_allow_html=True
)
