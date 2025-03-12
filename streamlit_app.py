import os
import streamlit as st
from document_loader import load_documents, split_documents
from vectorstore import create_vectorstore
from retriever_tool import create_retriever_tool_from_vectorstore
from state_graph import create_graph

def main():
    st.title("Document Q&A with Langchain")

    url = st.text_input("Enter the URL:")
    api_key = st.text_input("Enter your Groq API Key:", type="password")

    if st.button("Load Documents"):
        # Set the Groq API Key in environment variables
        os.environ["GROQ_API_KEY"] = api_key

        # Load and process documents
        urls = [url]
        docs_list = load_documents(urls)
        doc_splits = split_documents(docs_list)

        # Create vector store and retriever tool
        vectorstore = create_vectorstore(doc_splits)
        retriever_tool = create_retriever_tool_from_vectorstore(vectorstore)

        # Create the state graph
        graph = create_graph([retriever_tool])

        st.session_state.graph = graph
        st.success("Documents loaded successfully!")

    if 'graph' in st.session_state:
        question = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            graph = st.session_state.graph

            # Define input messages for the workflow
            inputs = {"messages": [("user", question)]}
            
            # Stream the output of the graph for the input messages
            for output in graph.stream(inputs):
                if 'generate' in output:
                    st.write(output['generate']['messages'][0])
                else:
                    st.write(output)

if __name__ == "__main__":
    main()
