#!/usr/bin/env python3
"""
Index Management Script for Cardiology RAG System
This script helps manage the persistent document index.
"""

import os
import sys
import argparse
from pathlib import Path
from cardiology_vector_db import CardiologyVectorDB


def show_status(vector_db):
    """Show detailed index status."""
    status = vector_db.get_index_status()

    print(f"\n📊 Cardiology RAG Index Status")
    print(f"{'='*60}")
    print(f"Collection Name: {status['collection_name']}")
    print(f"Processed Files: {status['processed_files_count']}")
    print(
        f"Documents in Collection: {status['collection_stats'].get('total_entities', 0)}")
    print(f"Metadata File: {status['index_metadata_file']}")
    print(f"Metadata Exists: {'✅' if status['metadata_file_exists'] else '❌'}")

    if status['processed_files']:
        print(f"\n📁 Processed Files Details:")
        for file_path, file_info in status['processed_files'].items():
            file_name = os.path.basename(file_path)
            file_exists = "✅" if os.path.exists(file_path) else "❌"

            print(f"\n  📄 {file_name} {file_exists}")
            print(f"     Path: {file_path}")
            print(f"     Chunks: {file_info.get('chunk_count', 'N/A')}")
            print(f"     Size: {file_info.get('file_size', 'N/A')} bytes")
            print(f"     Hash: {file_info.get('file_hash', 'N/A')}")
            print(f"     Processed: {file_info.get('processed_date', 'N/A')}")
    else:
        print(f"\n📁 No processed files found in metadata")

    # Show collection schema
    collection_stats = status['collection_stats']
    if 'schema' in collection_stats:
        print(f"\n🗂️ Collection Schema:")
        schema = collection_stats['schema']
        print(f"   Fields: {', '.join(schema.get('fields', []))}")
        print(f"   Description: {schema.get('description', 'N/A')}")


def add_document(vector_db, pdf_path):
    """Add a document to the index."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False

    print(f"📚 Adding document: {pdf_path}")
    success = vector_db.add_documents(pdf_path)

    if success:
        print(f"✅ Successfully added document to index")
        show_status(vector_db)
    else:
        print(f"❌ Failed to add document")

    return success


def force_reindex(vector_db, pdf_path):
    """Force reindexing of a document."""
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return False

    print(f"🔄 Force reindexing: {pdf_path}")
    success = vector_db.force_reindex(pdf_path)

    if success:
        print(f"✅ Successfully reindexed document")
        show_status(vector_db)
    else:
        print(f"❌ Failed to reindex document")

    return success


def clear_index(vector_db):
    """Clear the entire index."""
    print("⚠️ This will delete all indexed documents and metadata!")
    confirm = input("Type 'yes' to confirm: ").strip().lower()

    if confirm != 'yes':
        print("❌ Operation cancelled")
        return

    try:
        vector_db.cleanup()
        print("✅ Index cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing index: {str(e)}")


def search_test(vector_db, query):
    """Test search functionality."""
    print(f"🔍 Testing search with query: '{query}'")

    try:
        results = vector_db.search_similar_documents(query, top_k=3)

        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"\n📄 Result {i+1}:")
                print(f"   Score: {result['score']:.4f}")
                print(f"   Source: {result['source']}")
                print(f"   Page: {result['page_num']}")
                print(f"   Chunk: {result['chunk_id']}")
                print(f"   Text: {result['text'][:200]}...")
        else:
            print("❌ No results found")
    except Exception as e:
        print(f"❌ Search error: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage Cardiology RAG Index")

    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Status command
    subparsers.add_parser('status', help='Show index status')

    # Add document command
    add_parser = subparsers.add_parser('add', help='Add document to index')
    add_parser.add_argument('file', help='Path to PDF file')

    # Force reindex command
    reindex_parser = subparsers.add_parser(
        'reindex', help='Force reindex document')
    reindex_parser.add_argument('file', help='Path to PDF file')

    # Clear command
    subparsers.add_parser('clear', help='Clear entire index (DESTRUCTIVE)')

    # Search test command
    search_parser = subparsers.add_parser(
        'search', help='Test search functionality')
    search_parser.add_argument('query', help='Search query')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        # Initialize vector database
        print("🚀 Connecting to Milvus...")
        vector_db = CardiologyVectorDB()

        if args.command == 'status':
            show_status(vector_db)

        elif args.command == 'add':
            add_document(vector_db, args.file)

        elif args.command == 'reindex':
            force_reindex(vector_db, args.file)

        elif args.command == 'clear':
            clear_index(vector_db)

        elif args.command == 'search':
            search_test(vector_db, args.query)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Milvus is running")
        print("2. Check your .env file has API_KEY_GEMINI")
        print("3. Ensure you have the required Python packages installed")


if __name__ == "__main__":
    main()
