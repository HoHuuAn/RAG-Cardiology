# Cardiology RAG System with Persistent Indexing

This enhanced RAG system now includes persistent indexing, which means you don't need to reprocess documents every time you run the system. The index saves metadata about processed files and only reprocesses them if they have changed.

## 🚀 Quick Start

### 1. First Time Setup
```bash
# Start Milvus (if not already running)
.\standalone.bat start

# Run the RAG system (will automatically process PDF on first run)
python cardiology_rag.py
```

### 2. Subsequent Runs
```bash
# The system will automatically detect existing index and skip processing
python cardiology_rag.py
```

## 📊 Index Management

### Check Index Status
```bash
# Using the RAG system
python cardiology_rag.py --status

# Using the index manager
python index_manager.py status
```

### Force Reindexing
```bash
# If you want to reprocess documents (e.g., after updating the PDF)
python cardiology_rag.py --reindex

# Or specify a specific file
python cardiology_rag.py --reindex "docs/cardiology-explained.pdf"

# Using the index manager
python index_manager.py reindex "docs/cardiology-explained.pdf"
```

### Search Testing
```bash
# Test search functionality
python index_manager.py search "What is heart disease?"
```

### Clear Index (Start Fresh)
```bash
# WARNING: This deletes all indexed data
python index_manager.py clear
```

## 📁 Files Created by Persistent Indexing

1. **`index_metadata.json`** - Stores metadata about processed files including:
   - File hash (to detect changes)
   - Number of chunks created
   - Processing timestamp
   - File size

2. **Milvus Collection** - The vector database collection persists between runs

## 🔍 How It Works

### File Change Detection
- The system calculates an MD5 hash of each PDF file
- If the hash matches what's stored in `index_metadata.json`, the file is considered up-to-date
- If the hash differs or file is not in metadata, it will be reprocessed

### Collection Persistence
- Milvus collections persist in the Docker volumes
- The system checks if a collection exists before creating a new one
- Existing collections are loaded and used immediately

### Metadata Management
- `index_metadata.json` tracks:
  ```json
  {
    "processed_files": {
      "C:\\path\\to\\file.pdf": {
        "file_hash": "abc123...",
        "chunk_count": 573,
        "processed_date": "1234567890.123",
        "file_size": 12345678
      }
    },
    "collection_stats": {
      "total_entities": 573,
      "collection_name": "cardiology_rag"
    }
  }
  ```

## 🛠️ Command Line Options

### RAG System Commands
```bash
python cardiology_rag.py                    # Interactive mode
python cardiology_rag.py "your question"    # Single query mode
python cardiology_rag.py --status           # Show index status
python cardiology_rag.py --reindex [path]   # Force reindex documents
python cardiology_rag.py --help             # Show help
```

### Index Manager Commands
```bash
python index_manager.py status              # Show detailed status
python index_manager.py add file.pdf        # Add new document
python index_manager.py reindex file.pdf    # Force reindex document
python index_manager.py search "query"      # Test search
python index_manager.py clear               # Clear all data
```

## 🔧 Troubleshooting

### Index is Empty After Restart
1. Check if Milvus is running: `docker ps`
2. Check if collection exists in Milvus
3. Check if `index_metadata.json` exists
4. Try force reindexing: `python cardiology_rag.py --reindex`

### File Not Being Reprocessed After Changes
1. Check file hash in `index_metadata.json`
2. Use force reindex to override: `python index_manager.py reindex file.pdf`

### Performance Issues
- Large collections may take time to load on startup
- Consider using smaller chunk sizes if memory is limited
- Monitor Milvus memory usage with `docker stats`

## 📈 Benefits of Persistent Indexing

1. **Faster Startup** - No need to reprocess documents every time
2. **Incremental Updates** - Only process changed files
3. **Development Efficiency** - Faster iteration during development
4. **Resource Savings** - Avoid repeated embedding generation
5. **Production Ready** - Suitable for production environments

## 🔄 Migration from Previous Version

If you were using the previous version without persistent indexing:

1. **Backup your work** (optional)
2. **Let the system reprocess once** - The first run will create the index
3. **Subsequent runs will be fast** - The system will use the existing index

## 📝 Example Workflow

```bash
# 1. First time - processes documents and creates index
python cardiology_rag.py
# Output: Processing 573 chunks, creating embeddings...

# 2. Second time - loads existing index
python cardiology_rag.py
# Output: File already processed and up to date

# 3. Check what's indexed
python cardiology_rag.py --status
# Shows processed files and statistics

# 4. Update PDF and reprocess
python cardiology_rag.py --reindex
# Reprocesses only if file changed

# 5. Interactive session
python cardiology_rag.py
# Ask questions using the pre-built index
```

This persistent indexing system makes the RAG system much more practical for regular use and development!
