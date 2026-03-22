from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)