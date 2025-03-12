from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Create a vector store
def create_vectorstore(doc_splits):
    vectorstore = FAISS.from_documents(documents=doc_splits, embedding=HuggingFaceEmbeddings())
    return vectorstore
