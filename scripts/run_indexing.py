import asyncio
import os
import sys
from dotenv import load_dotenv
from loguru import logger
from datetime import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.core.processor import DocumentProcessor
from app.models.document_store import DocumentStore
from app.services.embedding_service import embedding_service
from app.core.config import settings

# Load environment variables from .env file
load_dotenv()

# Ensure DATABASE_URL is available
if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL not found in environment variables or .env file.")

# Create an async engine
engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def index_documents():
    """
    Main asynchronous function to process and index documents.
    """
    logger.info("Starting indexing process...")
    processor = DocumentProcessor()
    docs_path = "documents"
    
    # 1. Process documents from the filesystem
    document_chunks = processor.process_documents(docs_path)
    if not document_chunks:
        logger.warning("No documents found to index.")
        return

    logger.info(f"Found {len(document_chunks)} documents to process.")
    
    # 2. Extract content for embedding
    content_to_embed = [chunk.content for chunk in document_chunks]
    
    # 3. Generate embeddings in a batch
    logger.info("Generating embeddings for all documents... (This may take a while)")
    embeddings = embedding_service.encode(content_to_embed)
    logger.info("Embeddings generated successfully.")

    # 4. Store in database
    async with AsyncSessionFactory() as session:
        logger.info("Connecting to database and saving documents...")
        db_objects = []
        for chunk, embedding in zip(document_chunks, embeddings):
            # Create the SQLAlchemy model instance
            doc_record = DocumentStore(
                id=chunk.chunk_id,
                content=chunk.content,
                content_length=len(chunk.content),
                embedding=embedding.tolist(), # Convert numpy array to list for pgvector
                filename=chunk.metadata.get('filename'),
                file_path=chunk.metadata.get('file_path'),
                document_type=chunk.metadata.get('document_type'),
                created_at=chunk.metadata.get('created_at'),
                modified_at=chunk.metadata.get('modified_at'),
                metadata_={k: v.isoformat() if isinstance(v, datetime) else v for k, v in chunk.metadata.items()}
            )
            db_objects.append(doc_record)
        
        # Add all objects to the session and commit
        session.add_all(db_objects)
        await session.commit()
        logger.success(f"Successfully indexed {len(db_objects)} documents.")

async def main():
    # Before running, clear the table to avoid duplicates
    async with AsyncSessionFactory() as session:
        logger.info("Clearing existing documents from the database...")
        await session.execute(text("DELETE FROM document_store"))
        await session.commit()
    await index_documents()

if __name__ == "__main__":
    # Setup logger
    logger.add("logs/indexing.log", rotation="500 MB")
    asyncio.run(main()) 
 
 
 
 
 
 
 
 
 
 
 
 
 
 