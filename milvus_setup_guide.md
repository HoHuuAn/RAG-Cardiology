# Milvus Docker Setup Guide for Windows

## Prerequisites
Before setting up Milvus, ensure you have:
1. **Docker Desktop** installed and running in administrator mode
2. **Windows Subsystem for Linux 2 (WSL 2)** installed (optional but recommended)
3. **Python 3.8+** installed

## Setup Methods

### Method 1: Using Standalone Script (Recommended for beginners)

1. **Start Docker Desktop** in administrator mode (right-click → "Run as administrator")

2. **Run the standalone script:**
   ```powershell
   .\standalone.bat start
   ```

3. **Verify installation:**
   - Milvus will be available at `http://localhost:19530`
   - An embedded etcd will be available at port 2379
   - Data will be stored in `volumes/milvus` folder

4. **Manage the container:**
   ```powershell
   # Stop Milvus
   .\standalone.bat stop
   
   # Delete Milvus container and data
   .\standalone.bat delete
   ```

### Method 2: Using Docker Compose (Recommended for production)

1. **Start Docker Desktop** in administrator mode

2. **Start Milvus with Docker Compose:**
   ```powershell
   docker compose up -d
   ```

3. **Verify installation:**
   - Milvus standalone: `http://localhost:19530`
   - Minio (object storage): `http://localhost:9090` and `http://localhost:9091`
   - Data stored in respective `volumes/` folders

4. **Stop the services:**
   ```powershell
   docker compose down
   ```

## Installing Python Client

Install the Milvus Python client:
```powershell
pip install pymilvus
```

## Testing Your Installation

Use the provided `test_milvus_connection.py` script to test your Milvus installation.

## Troubleshooting

### Docker Engine Stopped Error
1. Check if virtualization is enabled in Task Manager → Performance tab
2. Start Docker Desktop Service: `net start com.docker.service`
3. Update WSL: `wsl --update`
4. Ensure Docker Desktop is running as administrator

### WSL-Related Issues
1. In Docker Desktop Settings → General: Check "Use the WSL 2 based engine"
2. In Settings → Resources → WSL Integration: Enable integration with your WSL distribution

### Volume-Related Errors
If you see "Read config failed" errors, check that volumes are properly mounted and accessible.

## Next Steps
- Check the [Milvus Quickstart Guide](https://milvus.io/docs/quickstart.md)
- Learn about [Managing Collections](https://milvus.io/docs/manage-collections.md)
- Explore [Vector Search](https://milvus.io/docs/single-vector-search.md)
