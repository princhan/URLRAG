from langchain.tools.retriever import create_retriever_tool

# Create a retriever tool
def create_retriever_tool_from_vectorstore(vectorstore):
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "retriever_vector_db_blog", "Search and run information about Bank Accounts or Bank Products")
    return retriever_tool
