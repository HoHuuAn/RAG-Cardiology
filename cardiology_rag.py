"""
Cardiology RAG (Retrieval-Augmented Generation) System
Main application for querying cardiology documents using Milvus and Gemini.
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from cardiology_vector_db import CardiologyVectorDB

# Load environment variables
load_dotenv()


class CardiologyRAG:
    """Main RAG system for cardiology Q&A using retrieved documents."""

    def __init__(self):
        """Initialize the RAG system."""
        # Initialize Gemini API
        self.api_key = os.getenv("API_KEY_GEMINI")
        if not self.api_key:
            raise ValueError(
                "API_KEY_GEMINI not found in environment variables")

        genai.configure(api_key=self.api_key)

        # Initialize vector database
        self.vector_db = CardiologyVectorDB()
        # Initialize Gemini model for text generation
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        print("🚀 Cardiology RAG system initialized successfully!")

    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the given query."""
        print(f"🔍 Searching for relevant context for: '{query}'")

        results = self.vector_db.search_similar_documents(query, top_k=top_k)

        if results:
            print(f"📚 Found {len(results)} relevant document chunks")
            for i, result in enumerate(results):
                print(
                    f"  📄 Document {i+1}: Page {result['page_num']}, Score: {result['score']:.4f}")
        else:
            print("❌ No relevant documents found")

        return results

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context for the LLM."""
        if not retrieved_docs:
            return ""

        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(
                f"Context {i+1} (Page {doc['page_num']}, Score: {doc['score']:.3f}):\n"
                f"{doc['text']}\n"
            )

        return "\n".join(context_parts)

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Gemini with retrieved context."""
        # Create a comprehensive prompt
        system_prompt = """You are a knowledgeable cardiology assistant. Your role is to provide accurate, helpful information about cardiovascular health, heart diseases, treatments, and related medical topics.

Instructions:
1. Use the provided context from medical documents to answer the question accurately
2. If the context doesn't contain enough information, acknowledge this limitation
3. Provide clear, well-structured answers that are easy to understand
4. Include relevant medical terminology but explain complex terms when necessary
5. Always emphasize that your response is for educational purposes and recommend consulting healthcare professionals for medical advice
6. If asked about symptoms, treatments, or medical procedures, be comprehensive but remind users to seek professional medical care

Context from Cardiology Documents:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided above."""

        prompt = system_prompt.format(context=context, query=query)

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def query(self, question: str, top_k: int = 5, show_sources: bool = True) -> Dict[str, Any]:
        """Main query method that performs RAG."""
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_context(question, top_k=top_k)

        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the cardiology documents to answer your question. Please try rephrasing your question or ask about topics related to cardiovascular health, heart diseases, or cardiac treatments.",
                "sources": [],
                "context_used": ""
            }

        # Step 2: Format context
        context = self.format_context(retrieved_docs)

        # Step 3: Generate response
        print("🤖 Generating response with Gemini...")
        answer = self.generate_response(question, context)

        # Step 4: Prepare response
        response = {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
            "context_used": context
        }

        # Step 5: Display results
        print(f"\n💡 Answer:")
        print(answer)

        if show_sources:
            print(f"\n📚 Sources used:")
            for i, doc in enumerate(retrieved_docs):
                print(
                    f"  📄 Source {i+1}: {doc['source']} (Page {doc['page_num']}, Chunk {doc['chunk_id']})")
                print(f"     Relevance Score: {doc['score']:.4f}")
                print(f"     Preview: {doc['text'][:150]}...")
                print()

        return response

    def interactive_session(self):
        """Start an interactive Q&A session."""
        print("\n🏥 Welcome to the Cardiology RAG System!")
        print("Ask questions about cardiovascular health, heart diseases, treatments, and more.")
        print("Type 'quit', 'exit', or 'q' to stop.\n")

        while True:
            try:
                question = input("❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Thank you for using the Cardiology RAG System!")
                    break

                self.query(question)
                print(f"\n{'-'*60}\n")

            except KeyboardInterrupt:
                print("\n👋 Session ended by user.")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        db_stats = self.vector_db.get_collection_stats()
        index_status = self.vector_db.get_index_status()

        return {
            "vector_database": db_stats,
            "index_status": index_status,
            "model": "gemini-1.5-flash",
            "embedding_model": "models/text-embedding-004",
            "embedding_dimension": 768
        }

    def show_index_status(self):
        """Display detailed index status information."""
        status = self.vector_db.get_index_status()

        print(f"\n📊 Index Status Report")
        print(f"{'='*50}")
        print(f"Collection Name: {status['collection_name']}")
        print(f"Processed Files: {status['processed_files_count']}")
        print(
            f"Documents in Collection: {status['collection_stats'].get('total_entities', 0)}")
        print(f"Metadata File: {status['index_metadata_file']}")
        print(f"Metadata Exists: {status['metadata_file_exists']}")

        if status['processed_files']:
            print(f"\n📁 Processed Files:")
            for file_path, file_info in status['processed_files'].items():
                file_name = os.path.basename(file_path)
                print(f"  • {file_name}")
                print(f"    Chunks: {file_info.get('chunk_count', 'N/A')}")
                print(f"    Size: {file_info.get('file_size', 'N/A')} bytes")
                print(f"    Hash: {file_info.get('file_hash', 'N/A')[:8]}...")
        else:
            print(f"\n📁 No processed files found")

    def force_reindex_documents(self, pdf_path: str = None):
        """Force reindexing of documents."""
        if pdf_path is None:
            pdf_path = "docs/cardiology-explained.pdf"

        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False

        print(f"🔄 Force reindexing documents from: {pdf_path}")
        return self.vector_db.force_reindex(pdf_path)


def main():
    """Main function to run the RAG system."""
    try:
        # Initialize the RAG system
        rag = CardiologyRAG()

        # Show index status
        rag.show_index_status()

        # Check if documents are loaded
        stats = rag.get_system_stats()
        if stats["vector_database"].get("total_entities", 0) == 0:
            print(
                "\n📚 No documents found in the database. Loading cardiology documents...")

            pdf_path = "docs/cardiology-explained.pdf"
            if os.path.exists(pdf_path):
                success = rag.vector_db.add_documents(pdf_path)
                if not success:
                    print("❌ Failed to load documents. Please check the PDF file.")
                    return
            else:
                print(f"❌ PDF file not found: {pdf_path}")
                print(
                    "Please ensure the cardiology-explained.pdf file is in the docs/ folder.")
                return
        else:
            print(
                f"\n✅ Database is ready with {stats['vector_database']['total_entities']} document chunks")

        # Check command line arguments for special commands
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()

            if command == "--status":
                # Show detailed status
                rag.show_index_status()
                return
            elif command == "--reindex":
                # Force reindexing
                pdf_path = sys.argv[2] if len(
                    sys.argv) > 2 else "docs/cardiology-explained.pdf"
                rag.force_reindex_documents(pdf_path)
                return
            elif command == "--help":
                print("\n🔧 Available commands:")
                print("  python cardiology_rag.py                    # Interactive mode")
                print(
                    "  python cardiology_rag.py 'your question'    # Single query mode")
                print(
                    "  python cardiology_rag.py --status           # Show index status")
                print(
                    "  python cardiology_rag.py --reindex [path]   # Force reindex documents")
                print("  python cardiology_rag.py --help             # Show this help")
                return
            else:
                # Single query mode
                query = " ".join(sys.argv[1:])
                rag.query(query)
        else:
            # Interactive mode
            rag.interactive_session()

    except Exception as e:
        print(f"❌ Error initializing RAG system: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure Milvus is running (run: .\\standalone.bat start)")
        print("2. Check if the .env file contains API_KEY_GEMINI")
        print("3. Ensure docs/cardiology-explained.pdf exists")
        print("4. Try running with --help for more options")


if __name__ == "__main__":
    main()
