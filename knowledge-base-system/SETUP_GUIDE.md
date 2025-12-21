# Knowledge Base System - Setup Guide

Complete guide to setting up and deploying the Knowledge Base System.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Setup](#docker-setup)
4. [Configuration](#configuration)
5. [Adding Companies](#adding-companies)
6. [Testing](#testing)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB for models and indexes
- **OS**: Windows, macOS, or Linux

### Optional

- **Docker**: For containerized deployment
- **GPU**: NVIDIA GPU with CUDA (optional, for faster embeddings)
- **Redis**: For distributed caching (optional)

## Local Development Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd knowledge-base-system
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn
- sentence-transformers
- FAISS
- NLTK
- And all other dependencies

### Step 4: Download Required Models

The embedding model will be downloaded automatically on first run, but you can pre-download it:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Download NLTK data:

```bash
python -c "import nltk; nltk.download('punkt')"
```

### Step 5: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings:

```env
DEBUG=True
LOG_LEVEL=INFO
PORT=8000
DEFAULT_COMPANY_ID=techstore
```

### Step 6: Verify Installation

```bash
python -c "from app.config import settings; print('Configuration loaded successfully')"
```

### Step 7: Run the Server

```bash
python run.py
```

Visit http://localhost:8000/docs to see the API documentation.

## Docker Setup

### Quick Start with Docker

1. **Build the image:**
```bash
docker build -t knowledge-base-system .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 -v $(pwd)/indexes:/app/indexes knowledge-base-system
```

### Using Docker Compose (Recommended)

1. **Start all services:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f knowledge-base
```

3. **Stop services:**
```bash
docker-compose down
```

4. **Rebuild after changes:**
```bash
docker-compose up -d --build
```

### Docker with GPU Support

Edit `docker-compose.yml` to add GPU support:

```yaml
services:
  knowledge-base:
    # ... other config ...
    environment:
      - GPU_ENABLED=True
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Configuration

### Environment Variables

Create a `.env` file with these key settings:

```env
# Application
APP_NAME=Knowledge Base System
DEBUG=False
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Embedding
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
GPU_ENABLED=False

# Chunking
CHUNK_SIZE_MIN=200
CHUNK_SIZE_TARGET=450
CHUNK_SIZE_MAX=600
CHUNK_OVERLAP=75

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
RERANK_ENABLED=True

# Caching
CACHE_ENABLED=True
CACHE_TTL=3600

# Paths
KNOWLEDGE_BASE_DIR=./knowledge_bases
INDEX_DIR=./indexes
LOG_DIR=./logs

# Default Company
DEFAULT_COMPANY_ID=techstore
```

### Performance Tuning

**For production:**
```env
DEBUG=False
WORKERS=4
MAX_CONCURRENT_REQUESTS=100
GPU_ENABLED=True  # If GPU available
```

**For development:**
```env
DEBUG=True
WORKERS=1
LOG_LEVEL=DEBUG
```

## Adding Companies

### Step 1: Create Directory Structure

```bash
mkdir -p knowledge_bases/mycompany
```

### Step 2: Add Metadata

Create `knowledge_bases/mycompany/metadata.json`:

```json
{
  "company_id": "mycompany",
  "company_name": "My Company Name",
  "description": "Brief description of the company",
  "active": true,
  "categories": ["support", "billing", "products"],
  "contact": {
    "email": "support@mycompany.com",
    "phone": "1-800-123-4567",
    "hours": "Mon-Fri 9AM-5PM"
  }
}
```

### Step 3: Add Documents

**FAQs** (`faqs.json`):
```json
{
  "faqs": [
    {
      "question": "How do I contact support?",
      "answer": "You can reach us at support@mycompany.com or call 1-800-123-4567.",
      "category": "support"
    }
  ]
}
```

**Products** (`products.json`):
```json
{
  "products": [
    {
      "name": "Product Name",
      "category": "products",
      "description": "Detailed product description..."
    }
  ]
}
```

**Policies** (`policies.txt`):
Plain text file with company policies.

### Step 4: Register Company

Add to `knowledge_bases/config.json`:

```json
{
  "companies": [
    {
      "company_id": "mycompany",
      "company_name": "My Company Name",
      "description": "...",
      "active": true,
      "categories": ["support", "billing", "products"]
    }
  ]
}
```

### Step 5: Load Company

```bash
curl -X POST "http://localhost:8000/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{"company_id": "mycompany", "force_reindex": true}'
```

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_chunking.py -v
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
# View coverage report: open htmlcov/index.html
```

### Test API Manually

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List companies
curl http://localhost:8000/api/v1/companies

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"company_id": "techstore", "query": "return policy", "top_k": 3}'
```

## Production Deployment

### Option 1: Systemd Service (Linux)

Create `/etc/systemd/system/knowledge-base.service`:

```ini
[Unit]
Description=Knowledge Base System
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/knowledge-base-system
Environment="PATH=/opt/knowledge-base-system/venv/bin"
ExecStart=/opt/knowledge-base-system/venv/bin/python run.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable knowledge-base
sudo systemctl start knowledge-base
sudo systemctl status knowledge-base
```

### Option 2: Docker in Production

```bash
docker-compose -f docker-compose.yml up -d
```

### Option 3: Cloud Deployment

**AWS EC2:**
1. Launch EC2 instance (t3.medium or larger)
2. Install Docker
3. Clone repository
4. Run with Docker Compose

**Google Cloud Run:**
1. Build Docker image
2. Push to Google Container Registry
3. Deploy to Cloud Run

**Azure Container Instances:**
1. Build Docker image
2. Push to Azure Container Registry
3. Create container instance

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL with Let's Encrypt

```bash
sudo certbot --nginx -d api.example.com
```

## Troubleshooting

### Common Issues

**1. Model download fails**
```bash
# Manually download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**2. Port already in use**
```bash
# Find process using port 8000
lsof -i :8000
# Kill it or change PORT in .env
```

**3. Out of memory**
```env
# Reduce batch size
EMBEDDING_BATCH_SIZE=16
# Disable caching
CACHE_ENABLED=False
```

**4. Slow queries**
```env
# Enable GPU if available
GPU_ENABLED=True
# Reduce top_k
TOP_K_RESULTS=3
```

**5. NLTK data not found**
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Logs

Check logs for debugging:
```bash
# Application logs
tail -f logs/app_$(date +%Y-%m-%d).log

# Error logs
tail -f logs/error_$(date +%Y-%m-%d).log

# Docker logs
docker-compose logs -f
```

### Performance Monitoring

Monitor the `/api/v1/stats` endpoint:
```bash
curl http://localhost:8000/api/v1/stats
```

Watch for:
- High average query time (>500ms)
- Low cache hit rate (<50%)
- High memory usage (>80% of available)

## Next Steps

1. **Load your companies**: Follow the "Adding Companies" section
2. **Test the API**: Use the interactive docs at `/docs`
3. **Integrate with your LLM**: Use the `/context` endpoint
4. **Monitor performance**: Check `/health` and `/stats` regularly
5. **Set up backups**: Backup `indexes/` directory regularly

## Support

- Review API documentation: http://localhost:8000/docs
- Check logs in `logs/` directory
- Run health check: http://localhost:8000/api/v1/health

For issues, check existing issues or create a new one.
