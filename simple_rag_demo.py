"""
Simple RAG (Retrieval-Augmented Generation) example using Milvus.
This demonstrates how to store and search text embeddings for RAG applications.
"""

import numpy as np
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)


class SimpleRAGWithMilvus:
    def __init__(self, collection_name="rag_collection", host="localhost", port="19530"):
        """Initialize the RAG system with Milvus."""
        self.collection_name = collection_name
        self.collection = None

        # Connect to Milvus
        connections.connect("default", host=host, port=port)
        print(f"✅ Connected to Milvus at {host}:{port}")

        self._setup_collection()

    def _setup_collection(self):
        """Set up the Milvus collection for storing documents."""
        # Drop existing collection if it exists
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🗑️ Dropped existing collection: {self.collection_name}")

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            # Using 384 dim for sentence transformers
            FieldSchema(name="embedding",
                        dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=200)
        ]

        schema = CollectionSchema(
            fields, "RAG collection for storing document embeddings")

        # Create collection
        self.collection = Collection(self.collection_name, schema)
        print(f"📦 Created collection: {self.collection_name}")

        # Create index
        index_params = {
            "metric_type": "IP",  # Inner Product for cosine similarity
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
        print("🔗 Created index for embeddings")

    def simple_text_embedding(self, text):
        """
        Simple text embedding function (for demonstration).
        In a real RAG system, you would use a proper embedding model like:
        - sentence-transformers
        - OpenAI embeddings
        - Hugging Face transformers
        """
        # This is a very simple hash-based embedding for demonstration
        # Replace this with a real embedding model in production
        words = text.lower().split()
        embedding = np.zeros(384)

        for i, word in enumerate(words[:384]):
            # Simple hash-based embedding (not suitable for production)
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def add_documents(self, documents):
        """Add documents to the vector database."""
        entities = [
            [doc["text"] for doc in documents],          # text field
            [self.simple_text_embedding(doc["text"])
             for doc in documents],  # embedding field
            [doc.get("source", "unknown") for doc in documents]  # source field
        ]

        insert_result = self.collection.insert(entities)
        self.collection.flush()  # Ensure data is written
        print(f"📥 Added {len(insert_result.primary_keys)} documents")

        # Load collection for searching
        self.collection.load()
        print("⏳ Collection loaded and ready for search")

    def search_similar_documents(self, query, top_k=3):
        """Search for documents similar to the query."""
        query_embedding = self.simple_text_embedding(query)

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            [query_embedding],
            "embedding",
            search_params,
            limit=top_k,
            output_fields=["text", "source"]
        )

        return results[0]

    def rag_query(self, query, top_k=3):
        """Perform a RAG query: retrieve relevant documents and format response."""
        print(f"🔍 Searching for: '{query}'")

        # Retrieve similar documents
        search_results = self.search_similar_documents(query, top_k)

        print(f"\n📚 Found {len(search_results)} relevant documents:")

        context_docs = []
        for i, result in enumerate(search_results):
            print(f"\n  📄 Document {i+1} (Score: {result.distance:.4f}):")
            print(f"     Source: {result.entity.get('source')}")
            print(f"     Text: {result.entity.get('text')[:200]}...")

            context_docs.append({
                "text": result.entity.get('text'),
                "source": result.entity.get('source'),
                "score": result.distance
            })

        return context_docs

    def cleanup(self):
        """Clean up resources."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"🧹 Cleaned up collection: {self.collection_name}")

        connections.disconnect("default")
        print("👋 Disconnected from Milvus")


def demo_rag_system():
    """Demonstrate the RAG system with sample documents."""

    # Sample documents for demonstration
    sample_documents = [
        {
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make predictions from data.",
            "source": "AI_Basics.pdf"
        },
        {
            "text": "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently for similarity search.",
            "source": "Database_Guide.pdf"
        },
        {
            "text": "Retrieval-Augmented Generation combines information retrieval with large language models to provide more accurate and contextual responses.",
            "source": "RAG_Overview.pdf"
        },
        {
            "text": "Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight containers.",
            "source": "DevOps_Manual.pdf"
        },
        {
            "text": "Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way.",
            "source": "NLP_Introduction.pdf"
        },
        {
            "text": "Deep learning uses neural networks with multiple layers to learn complex patterns in data for tasks like image recognition and language understanding.",
            "source": "Deep_Learning_Fundamentals.pdf"
        }
    ]

    print("🚀 Starting RAG System Demo with Milvus")

    # Initialize RAG system
    rag_system = SimpleRAGWithMilvus()

    try:
        # Add documents to the system
        print("\n📚 Adding sample documents...")
        rag_system.add_documents(sample_documents)

        # Perform some queries
        queries = [
            "What is machine learning?",
            "How do vector databases work?",
            "Tell me about containers and Docker",
            "What is deep learning?"
        ]

        for query in queries:
            print(f"\n{'='*60}")
            results = rag_system.rag_query(query, top_k=2)
            print(f"{'='*60}")

            # In a real RAG system, you would now send these context documents
            # along with the query to a language model (like GPT, Claude, etc.)
            # to generate a comprehensive answer

        print(f"\n🎉 RAG Demo completed successfully!")
        print("\n💡 Next steps for a production RAG system:")
        print("  1. Replace simple_text_embedding() with a real embedding model")
        print("  2. Add text preprocessing and chunking")
        print("  3. Integrate with a language model (OpenAI, Anthropic, etc.)")
        print("  4. Add more sophisticated retrieval strategies")
        print("  5. Implement re-ranking and filtering")

    finally:
        # Clean up
        rag_system.cleanup()


if __name__ == "__main__":
    demo_rag_system()
