import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the Groq API key in the environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
