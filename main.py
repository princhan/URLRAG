import os
from document_loader import load_documents, split_documents
from vectorstore import create_vectorstore
from retriever_tool import create_retriever_tool_from_vectorstore
from state_graph import create_graph

def main():
    # Ask for the URL and API Key
    url = input("Enter the URL: ")
    api_key = input("Enter your Groq API Key: ")

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

    while True:
        # Ask the user for a question
        question = input("Enter your question: ")

        if question.lower() in ['exit', 'quit']:
            break

        # Define input messages for the workflow
        inputs = {"messages": [("user", question)]}

        # Stream the output of the graph for the input messages
        for output in graph.stream(inputs):
            if 'generate' in output:
                print(output['generate']['messages'][0])
            else:
                print(output)

if __name__ == "__main__":
    main()
