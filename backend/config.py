import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROJECT_NAME = os.getenv("PROJECT_NAME", "RAG Chat App")
