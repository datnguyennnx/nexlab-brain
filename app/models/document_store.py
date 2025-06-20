import uuid
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from sqlalchemy_utils import TSVectorType
from ..models.base import Base

class DocumentStore(Base):
    __tablename__ = 'document_store'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    content_length = Column(Integer)
    embedding = Column(Vector(768))  # Dimension for bkai-foundation-models/vietnamese-bi-encoder is 768
    
    # Metadata fields
    filename = Column(String)
    file_path = Column(String)
    document_type = Column(String) # e.g., 'md', 'json'
    subject = Column(String)
    category = Column(String)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    modified_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Full-text search vector
    # This will be automatically updated based on the 'content' column
    # The language 'vietnamese' might not be available by default in all postgres installations.
    # It might require installing a dictionary. 'simple' is a safe fallback.
    content_tsv = Column(TSVectorType('content', regconfig='pg_catalog.simple'))

    # Store other metadata in a JSON field for flexibility
    metadata_ = Column("metadata", JSON)

    def __repr__(self):
        return f"<DocumentStore(id={self.id}, filename='{self.filename}')>" 
 
 
 
 
 
 
 
 
 
 
 
 
 
 