# Cardiology RAG System 🫀

A sophisticated Retrieval-Augmented Generation (RAG) system designed for cardiology-related queries, powered by Milvus vector database and Google's Gemini AI model.

## 🌟 Features

- **Intelligent Document Processing**: Automatically processes cardiology PDFs and creates searchable embeddings
- **Persistent Indexing**: Smart caching system that avoids reprocessing unchanged documents
- **Vector Search**: High-performance similarity search using Milvus vector database
- **AI-Powered Responses**: Contextual answers generated using Google's Gemini 1.5 Flash model
- **Interactive Sessions**: Command-line interface for continuous Q&A sessions
- **Docker Integration**: Containerized Milvus setup for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Vector Database │───▶│   Gemini AI     │
│  (Cardiology)   │    │    (Milvus)     │    │   (Response)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Text Chunking & │    │ Similarity      │    │ Context-Aware   │
│ Preprocessing   │    │ Search          │    │ Generation      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **Docker Desktop** (Windows - run as administrator)
- **Google Gemini API Key**
- **Windows Subsystem for Linux 2 (WSL 2)** (recommended)

## 🚀 Quick Start

### 1. Clone and Setup

```powershell
# Navigate to your project directory
cd c:\Users\An\Desktop\datamining\RAG

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```env
API_KEY_GEMINI=your_gemini_api_key_here
```

### 3. Start Milvus Vector Database

Choose one of the following methods:

#### Method A: Docker Compose (Recommended)
```powershell
# Start all services
docker compose up -d

# Verify services are running
docker compose ps
```

#### Method B: Standalone Script
```powershell
# Start Milvus standalone
.\standalone.bat start
```

### 4. Run the RAG System

```powershell
# First run (will process documents and create index)
python cardiology_rag.py

# Interactive session
python cardiology_rag.py --interactive
```

## 📁 Project Structure

```
RAG/
├── 📄 cardiology_rag.py          # Main RAG application
├── 📄 cardiology_vector_db.py    # Vector database handler
├── 📄 simple_rag_demo.py         # Basic RAG demonstration
├── 📄 index_manager.py           # Index management utilities
├── 📄 test_milvus_connection.py  # Connection testing
├── 🐳 docker-compose.yml         # Docker services configuration
├── 📋 requirements.txt           # Python dependencies
├── 📊 index_metadata.json        # Persistent index metadata
├── 📁 docs/                      # Document storage
│   └── cardiology-explained.pdf  # Source cardiology document
├── 📁 volumes/                   # Docker persistent storage
│   └── milvus/                   # Milvus data directory
└── 📚 Documentation/
    ├── milvus_setup_guide.md     # Milvus installation guide
    ├── milvus_quick_reference.md # Quick command reference
    ├── persistent_indexing_guide.md # Index management guide
    └── sample_questions.md       # Example queries
```

## 🔧 Usage Examples

### Basic Query
```python
from cardiology_rag import CardiologyRAG

rag = CardiologyRAG()
result = rag.query("What is coronary artery disease?")
print(result['answer'])
```

### Interactive Session
```powershell
python cardiology_rag.py --interactive
```

### Index Management
```powershell
# Check index status
python index_manager.py status

# Force reindexing
python cardiology_rag.py --reindex

# Clear all indexes
python index_manager.py clear
```

## 📊 System Commands

### Milvus Management
```powershell
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f standalone

# Check container status
docker ps
```

### Database Operations
```powershell
# Test connection
python test_milvus_connection.py

# Search specific query
python index_manager.py search "heart disease symptoms"

# Check collection status
python -c "from cardiology_vector_db import CardiologyVectorDB; db = CardiologyVectorDB(); print(db.get_collection_info())"
```

## 🐛 Troubleshooting

### Common Issues

1. **Docker Engine Issues**
   ```powershell
   # Restart Docker service
   net start com.docker.service
   
   # Check if virtualization is enabled
   # Task Manager → Performance → CPU → Virtualization: Enabled
   ```

2. **Port Conflicts**
   ```powershell
   # Check what's using port 19530
   netstat -an | findstr 19530
   ```

3. **Memory Issues**
   ```powershell
   # Increase Docker memory allocation
   # Docker Desktop → Settings → Resources → Memory: 8GB+
   ```

4. **API Key Issues**
   - Ensure `.env` file exists with valid `API_KEY_GEMINI`
   - Check Gemini API quota and billing

### Reset Everything
```powershell
# Stop and remove all containers
docker compose down -v

# Remove all Docker data
docker system prune -f

# Start fresh
docker compose up -d
```

## 🧪 Testing

### Test Vector Database Connection
```powershell
python test_milvus_connection.py
```

### Test RAG Pipeline
```powershell
python simple_rag_demo.py
```

### Sample Queries
Refer to `sample_questions.md` for comprehensive test queries covering:
- Heart anatomy and function
- Cardiovascular diseases
- Risk factors and prevention
- Treatment and medications

## 🔒 Security Considerations

- Store API keys in `.env` file (never commit to version control)
- Use Docker networks for service isolation
- Regularly update dependencies for security patches
- Consider using Docker secrets for production deployments

## 📈 Performance Optimization

- **Chunk Size**: Adjust text chunking parameters for optimal retrieval
- **Vector Dimensions**: Optimize embedding dimensions based on your data
- **Index Parameters**: Tune Milvus index settings for your workload
- **Batch Processing**: Process multiple documents in batches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📚 Documentation

- [Milvus Setup Guide](milvus_setup_guide.md) - Detailed installation instructions
- [Milvus Quick Reference](milvus_quick_reference.md) - Command cheat sheet
- [Persistent Indexing Guide](persistent_indexing_guide.md) - Index management
- [Sample Questions](sample_questions.md) - Test query examples

## 🔗 Dependencies

- **pymilvus**: Vector database client
- **google-generativeai**: Gemini AI integration
- **PyPDF2**: PDF text extraction
- **sentence-transformers**: Text embeddings
- **python-dotenv**: Environment variable management
- **numpy**: Numerical computations

## 📝 License

This project is for educational and research purposes. Please ensure compliance with relevant medical information regulations when using in clinical settings.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the documentation files
3. Test with `simple_rag_demo.py` for basic functionality
4. Verify Milvus connection with `test_milvus_connection.py`

---

## Author

HoHuuAn
