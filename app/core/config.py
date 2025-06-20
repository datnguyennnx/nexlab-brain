from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    
    # Langfuse credentials
    LANGFUSE_HOST: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str

    class Config:
        env_file = ".env"

settings = Settings() 