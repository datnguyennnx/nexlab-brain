from langfuse import Langfuse
from .config import settings

# Initialize the Langfuse client with credentials from your environment settings.
# This client is a singleton and can be imported across the application.
langfuse = Langfuse(
    secret_key=settings.LANGFUSE_SECRET_KEY,
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    host=settings.LANGFUSE_HOST,
    debug=False # Set to True for verbose SDK logging
) 