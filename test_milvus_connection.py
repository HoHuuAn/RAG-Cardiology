"""
Test script to verify Milvus Docker installation and basic operations.
Make sure Milvus is running before executing this script.
"""

import time
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
import numpy as np


def test_milvus_connection():
    """Test connection to Milvus and perform basic operations."""

    print("🚀 Testing Milvus Docker Installation...")

    try:
        # Connect to Milvus
        print("📡 Connecting to Milvus...")
        connections.connect("default", host="localhost", port="19530")
        print("✅ Successfully connected to Milvus!")

        # Check if Milvus is ready
        print("🔍 Checking Milvus status...")
        print(f"Milvus version: {utility.get_server_version()}")

        # Create a test collection
        collection_name = "test_collection"

        # Check if collection exists and drop it
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️ Dropped existing collection: {collection_name}")

        # Define collection schema
        print("📋 Creating collection schema...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64,
                        is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]

        schema = CollectionSchema(
            fields, "Test collection for Milvus setup verification")

        # Create collection
        print(f"📦 Creating collection: {collection_name}")
        collection = Collection(collection_name, schema)
        print("✅ Collection created successfully!")

        # Create index
        print("🔗 Creating index...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        print("✅ Index created successfully!")

        # Insert test data
        print("📥 Inserting test data...")
        entities = [
            [i for i in range(10)],  # IDs
            [np.random.random(128).tolist() for _ in range(10)]  # Embeddings
        ]

        insert_result = collection.insert(entities)
        print(f"✅ Inserted {len(insert_result.primary_keys)} entities")

        # Load collection
        print("⏳ Loading collection...")
        collection.load()
        print("✅ Collection loaded!")

        # Perform a search
        print("🔍 Performing vector search...")
        search_vectors = [np.random.random(128).tolist()]
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }

        results = collection.search(
            search_vectors,
            "embedding",
            search_params,
            limit=3,
            output_fields=["id"]
        )

        print("✅ Search completed!")
        print(f"Found {len(results[0])} results:")
        for i, result in enumerate(results[0]):
            print(f"  - ID: {result.id}, Distance: {result.distance:.4f}")

        # Clean up
        print("🧹 Cleaning up...")
        utility.drop_collection(collection_name)
        print(f"✅ Dropped test collection: {collection_name}")

        print("\n🎉 All tests passed! Milvus is working correctly!")
        print("📊 Summary:")
        print("  - Connection: ✅")
        print("  - Collection operations: ✅")
        print("  - Index creation: ✅")
        print("  - Data insertion: ✅")
        print("  - Vector search: ✅")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure Docker Desktop is running")
        print("2. Make sure Milvus is started (run: .\\standalone.bat start)")
        print("3. Check if port 19530 is accessible")
        print("4. Install pymilvus: pip install pymilvus")

    finally:
        # Disconnect
        connections.disconnect("default")
        print("👋 Disconnected from Milvus")


if __name__ == "__main__":
    test_milvus_connection()
