"""
Cardiology RAG Vector Database Handler using Milvus and Gemini
This module handles PDF document processing, embeddings, and vector storage with persistent indexing.
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    MilvusClient
)

# Load environment variables
load_dotenv()


class CardiologyVectorDB:
    """Handles vector database operations for cardiology documents using Milvus and Gemini."""

    def __init__(self, collection_name: str = "cardiology_rag",
                 milvus_host: str = "localhost",
                 milvus_port: str = "19530",
                 index_metadata_file: str = "index_metadata.json"):
        """Initialize the vector database handler."""
        self.collection_name = collection_name
        self.collection = None
        self.index_metadata_file = index_metadata_file

        # Initialize Gemini API
        self.api_key = os.getenv("API_KEY_GEMINI")
        if not self.api_key:
            raise ValueError(
                "API_KEY_GEMINI not found in environment variables")

        genai.configure(api_key=self.api_key)

        # Connect to Milvus
        connections.connect("default", host=milvus_host, port=milvus_port)
        print(f"✅ Connected to Milvus at {milvus_host}:{milvus_port}")

        self._setup_collection()

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file to detect changes."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"❌ Error calculating file hash: {str(e)}")
            return ""

    def _load_index_metadata(self) -> Dict[str, Any]:
        """Load metadata about processed documents."""
        if os.path.exists(self.index_metadata_file):
            try:
                with open(self.index_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Warning: Could not load index metadata: {str(e)}")
        return {"processed_files": {}, "collection_stats": {}}

    def _save_index_metadata(self, metadata: Dict[str, Any]):
        """Save metadata about processed documents."""
        try:
            with open(self.index_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving index metadata: {str(e)}")

    def _is_file_processed(self, file_path: str) -> bool:
        """Check if a file has already been processed and indexed."""
        metadata = self._load_index_metadata()

        if not os.path.exists(file_path):
            return False

        file_key = os.path.abspath(file_path)
        current_hash = self._get_file_hash(file_path)

        if file_key in metadata["processed_files"]:
            stored_hash = metadata["processed_files"][file_key].get(
                "file_hash", "")
            return stored_hash == current_hash

        return False

    def _setup_collection(self):
        """Set up the Milvus collection for storing document embeddings."""
        # Check if collection already exists
        if utility.has_collection(self.collection_name):
            print(f"📦 Using existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)

            # Ensure the collection is loaded
            try:
                self.collection.load()
                print("✅ Collection is loaded and ready for use")

                # Display collection stats
                stats = self.get_collection_stats()
                if stats.get("total_entities", 0) > 0:
                    print(
                        f"📊 Collection contains {stats['total_entities']} document chunks")
                    return
                else:
                    print("⚠️ Collection exists but is empty")
            except Exception as e:
                print(f"⚠️ Warning loading existing collection: {str(e)}")
        else:
            print(f"🆕 Creating new collection: {self.collection_name}")

        # Define schema with proper field specifications
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,
                        dim=768),  # Gemini embedding dimension
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="chunk_id", dtype=DataType.INT64)
        ]

        schema = CollectionSchema(
            fields, "Cardiology RAG collection for storing document embeddings")

        # Create collection only if it doesn't exist
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name, schema)
            print(f"📦 Created collection: {self.collection_name}")

            # Create index for better search performance
            index_params = {
                "metric_type": "COSINE",  # Cosine similarity for text embeddings
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index("embedding", index_params)
            print("🔗 Created index for embeddings")
        else:
            self.collection = Collection(self.collection_name)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini API."""
        try:
            # Use Gemini's text embedding model
            model = 'models/text-embedding-004'
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Error generating embedding: {str(e)}")
            return [0.0] * 768  # Return zero vector on error

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search queries using Gemini API."""
        try:
            model = 'models/text-embedding-004'
            result = genai.embed_content(
                model=model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"❌ Error generating query embedding: {str(e)}")
            return [0.0] * 768

    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Load and extract text from PDF file."""
        documents = []

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(
                    f"📄 Processing PDF with {len(pdf_reader.pages)} pages...")

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only process pages with text
                        documents.append({
                            'text': text,
                            'source': os.path.basename(pdf_path),
                            'page_num': page_num + 1
                        })

        except Exception as e:
            print(f"❌ Error reading PDF: {str(e)}")
            return []

        print(f"✅ Extracted text from {len(documents)} pages")
        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size

            # If we're not at the end of the text, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundaries
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundaries
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end

            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end

        return chunks

    def process_documents(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF documents into chunks with metadata."""
        documents = self.load_pdf(pdf_path)
        processed_docs = []

        for doc in documents:
            chunks = self.chunk_text(doc['text'])

            for chunk_id, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Only process meaningful chunks
                    processed_docs.append({
                        'text': chunk,
                        'source': doc['source'],
                        'page_num': doc['page_num'],
                        'chunk_id': chunk_id
                    })

        print(f"✅ Processed {len(processed_docs)} text chunks")
        return processed_docs

    def add_documents(self, pdf_path: str) -> bool:
        """Process and add documents to the vector database with persistent indexing."""
        try:
            # Check if file has already been processed
            if self._is_file_processed(pdf_path):
                print(f"✅ File already processed and up to date: {pdf_path}")
                stats = self.get_collection_stats()
                print(
                    f"📊 Collection contains {stats['total_entities']} document chunks")

                # Ensure collection is loaded
                if not self.collection.is_loaded:
                    self.collection.load()
                    print("⏳ Collection loaded and ready for search")

                return True

            print(f"📚 Processing documents from: {pdf_path}")

            # Process the PDF into chunks
            processed_docs = self.process_documents(pdf_path)

            if not processed_docs:
                print("❌ No documents to process")
                return False

            # Generate embeddings and prepare data for insertion
            texts = []
            embeddings = []
            sources = []
            page_nums = []
            chunk_ids = []

            print(
                f"🔄 Generating embeddings for {len(processed_docs)} chunks...")

            for i, doc in enumerate(processed_docs):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(processed_docs)}")

                embedding = self.generate_embedding(doc['text'])

                texts.append(doc['text'])
                embeddings.append(embedding)
                sources.append(doc['source'])
                page_nums.append(doc['page_num'])
                chunk_ids.append(doc['chunk_id'])

            # Insert data into Milvus
            entities = [texts, embeddings, sources, page_nums, chunk_ids]

            insert_result = self.collection.insert(entities)
            self.collection.flush()  # Ensure data is written

            print(
                f"📥 Added {len(insert_result.primary_keys)} document chunks to vector database")

            # Load collection for searching
            self.collection.load()
            print("⏳ Collection loaded and ready for search")

            # Update metadata to mark file as processed
            self._mark_file_as_processed(pdf_path, len(processed_docs))

            return True

        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False

    def _mark_file_as_processed(self, file_path: str, chunk_count: int):
        """Mark a file as processed in the metadata."""
        metadata = self._load_index_metadata()

        file_key = os.path.abspath(file_path)
        file_hash = self._get_file_hash(file_path)

        metadata["processed_files"][file_key] = {
            "file_hash": file_hash,
            "chunk_count": chunk_count,
            "processed_date": str(Path(file_path).stat().st_mtime),
            "file_size": os.path.getsize(file_path)
        }

        # Update collection stats
        stats = self.get_collection_stats()
        metadata["collection_stats"] = stats

        self._save_index_metadata(metadata)
        print(f"💾 Updated index metadata for: {os.path.basename(file_path)}")

    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents similar to the query."""
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)

            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }

            # Perform search
            results = self.collection.search(
                [query_embedding],
                "embedding",
                search_params,
                limit=top_k,
                output_fields=["text", "source", "page_num", "chunk_id"]
            )

            # Format results
            formatted_results = []
            for result in results[0]:
                formatted_results.append({
                    "text": result.entity.get('text'),
                    "source": result.entity.get('source'),
                    "page_num": result.entity.get('page_num'),
                    "chunk_id": result.entity.get('chunk_id'),
                    "score": float(result.distance)
                })

            return formatted_results

        except Exception as e:
            print(f"❌ Error searching documents: {str(e)}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}

            stats = {
                "total_entities": self.collection.num_entities,
                "collection_name": self.collection_name,
                "schema": {
                    "fields": [field.name for field in self.collection.schema.fields],
                    "description": self.collection.schema.description
                }
            }
            return stats

        except Exception as e:
            return {"error": str(e)}

    def get_index_status(self) -> Dict[str, Any]:
        """Get detailed status of the index and processed files."""
        metadata = self._load_index_metadata()
        collection_stats = self.get_collection_stats()

        return {
            "index_metadata_file": self.index_metadata_file,
            "collection_name": self.collection_name,
            "processed_files_count": len(metadata.get("processed_files", {})),
            "processed_files": metadata.get("processed_files", {}),
            "collection_stats": collection_stats,
            "metadata_file_exists": os.path.exists(self.index_metadata_file)
        }

    def force_reindex(self, pdf_path: str) -> bool:
        """Force reprocessing of a document even if it's already indexed."""
        print(f"🔄 Force reindexing: {pdf_path}")

        # Remove from metadata
        metadata = self._load_index_metadata()
        file_key = os.path.abspath(pdf_path)

        if file_key in metadata["processed_files"]:
            del metadata["processed_files"][file_key]
            self._save_index_metadata(metadata)
            print(
                f"🗑️ Removed {os.path.basename(pdf_path)} from index metadata")

        # Process the document
        return self.add_documents(pdf_path)

    def list_processed_files(self) -> Dict[str, Any]:
        """Get list of all processed files and their metadata."""
        metadata = self._load_index_metadata()
        return metadata.get("processed_files", {})

    def cleanup(self):
        """Clean up resources."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            # Optionally clean up metadata file
            print(f"🧹 Cleaned up collection: {self.collection_name}")
        if os.path.exists(self.index_metadata_file):
            os.remove(self.index_metadata_file)
            print(f"🧹 Cleaned up metadata file: {self.index_metadata_file}")

        connections.disconnect("default")
        print("👋 Disconnected from Milvus")
