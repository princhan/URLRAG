from langchain_groq import ChatGroq

# Initialize the language model
def initialize_language_model():
    return ChatGroq(model="qwen-2.5-32b")
