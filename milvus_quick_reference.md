# Milvus Docker Quick Reference

## 🚀 Quick Start Commands

### Start Milvus
```powershell
.\standalone.bat start
```

### Stop Milvus
```powershell
.\standalone.bat stop
```

### Delete Milvus (removes container and data)
```powershell
.\standalone.bat delete
```

### Using Docker Compose (Alternative)
```powershell
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f
```

## 🔍 Check Status

### Check Docker containers
```powershell
docker ps
```

### Check Milvus logs
```powershell
docker logs milvus-standalone
```

### Test connection
```powershell
python test_milvus_connection.py
```

## 📊 Connection Details

- **Milvus Server**: `localhost:19530`
- **Etcd**: `localhost:2379`
- **Minio** (if using Docker Compose): `localhost:9090` and `localhost:9091`

## 📁 Data Storage

- **Standalone script**: Data stored in `volumes/milvus/`
- **Docker Compose**: 
  - Milvus data: `volumes/milvus/`
  - Etcd data: `volumes/etcd/`
  - Minio data: `volumes/minio/`

## 🛠 Troubleshooting

### Docker Engine Issues
1. Ensure Docker Desktop is running as administrator
2. Check virtualization is enabled in BIOS
3. Restart Docker service: `net start com.docker.service`

### Port Conflicts
```powershell
# Check what's using port 19530
netstat -an | findstr 19530
```

### Reset Everything
```powershell
# Stop and remove everything
.\standalone.bat delete
docker system prune -f

# Start fresh
.\standalone.bat start
```

## 🐍 Python Usage

### Install dependencies
```powershell
pip install pymilvus numpy
```

### Basic connection test
```python
from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
```

## 📚 Next Steps

1. **Learn Milvus basics**: Check the [official documentation](https://milvus.io/docs)
2. **Build a RAG system**: Use the `simple_rag_demo.py` as a starting point
3. **Production setup**: Consider using proper embedding models like sentence-transformers
4. **Scale up**: Explore Milvus distributed mode for larger datasets

## 🔗 Useful Links

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus API Reference](https://milvus.io/api-reference/pymilvus/v2.4.x/About.md)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
- [Community Discord](https://milvus.io/discord)
