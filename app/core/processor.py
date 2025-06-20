import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
from datetime import datetime

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class DocumentProcessor:
    def process_documents(self, docs_path: str) -> List[DocumentChunk]:
        """
        Processes all .md and .json files in a directory, treating each file as one chunk.
        """
        chunks = []
        for filename in os.listdir(docs_path):
            file_path = os.path.join(docs_path, filename)
            if not os.path.isfile(file_path):
                continue
            
            try:
                file_extension = os.path.splitext(filename)[1].lower()
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    continue
                
                stat = os.stat(file_path)
                metadata = {
                    'filename': filename,
                    'file_path': file_path,
                    'document_type': file_extension,
                    'created_at': datetime.fromtimestamp(stat.st_ctime),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime),
                }

                if file_extension == '.json':
                    # For JSON, we can pretty-print it to make the content more readable
                    try:
                        parsed_json = json.loads(content)
                        content = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # If it's not valid JSON, just use the raw content
                        pass
                
                chunks.append(DocumentChunk(content=content, metadata=metadata))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        return chunks 
 
 
 
 
 
 
 
 
 
 
 
 
 
 