# Knowledge Base System for AI Voice Agents

A production-ready RAG (Retrieval Augmented Generation) knowledge base management system built with FastAPI, designed for AI voice agents handling customer service calls.

## Features

- **Multi-Company Support**: Manage knowledge bases for multiple companies with easy switching
- **Intelligent Chunking**: Hybrid sentence-based chunking (200-600 words) with semantic awareness
- **Fast Retrieval**: FAISS-powered vector search with <200ms query times
- **Smart Re-ranking**: Priority-based and keyword-overlap re-ranking for better results
- **LLM-Ready Context**: Formatted context assembly for seamless LLM integration
- **Caching**: Built-in query and embedding caching for improved performance
- **REST API**: Comprehensive FastAPI endpoints with automatic documentation
- **Scalable**: Handles 50+ concurrent requests with efficient batch processing

## Architecture

```
Knowledge Base System
├── Document Ingestion
│   ├── File parsing (JSON, TXT)
│   ├── Intelligent chunking
│   └── Embedding generation
├── Vector Storage (FAISS)
│   ├── Similarity search
│   └── Metadata filtering
├── Retrieval Pipeline
│   ├── Query preprocessing
│   ├── Embedding search
│   ├── Re-ranking
│   └── Context assembly
└── FastAPI REST API
    ├── Company management
    ├── Query endpoints
    └── Statistics
```

## Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (easily migrateable to ChromaDB)
- **NLP**: NLTK for sentence tokenization
- **Logging**: Loguru
- **Testing**: pytest

## Quick Start

### Installation

1. **Clone the repository**:
```bash
cd knowledge-base-system
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (automatic on first run):
```python
python -c "import nltk; nltk.download('punkt')"
```

5. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the Server

**Option 1: Using run script**
```bash
python run.py
```

**Option 2: Using uvicorn directly**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## Usage Examples

### 1. Load a Company

```bash
curl -X POST "http://localhost:8000/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "techstore",
    "force_reindex": false
  }'
```

### 2. Query Knowledge Base

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "techstore",
    "query": "What is your return policy?",
    "top_k": 5
  }'
```

### 3. Get LLM Context

```bash
curl -X POST "http://localhost:8000/api/v1/context" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "techstore",
    "query": "How can I track my order?",
    "top_k": 3,
    "include_metadata": true
  }'
```

### 4. List Available Companies

```bash
curl "http://localhost:8000/api/v1/companies"
```

### 5. Health Check

```bash
curl "http://localhost:8000/api/v1/health"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/company/load` | POST | Load and index company documents |
| `/api/v1/company/switch` | POST | Switch active company |
| `/api/v1/query` | POST | Query knowledge base |
| `/api/v1/context` | POST | Get formatted LLM context |
| `/api/v1/companies` | GET | List all companies |
| `/api/v1/health` | GET | System health status |
| `/api/v1/stats` | GET | Usage statistics |
| `/api/v1/ingest` | POST | Re-ingest documents |
| `/api/v1/company/{id}` | DELETE | Delete company data |
| `/api/v1/cache/clear` | POST | Clear caches |

## Adding Your Own Company

1. **Create company directory**:
```bash
mkdir knowledge_bases/mycompany
```

2. **Add metadata.json**:
```json
{
  "company_id": "mycompany",
  "company_name": "My Company",
  "description": "Company description",
  "active": true,
  "categories": ["support", "billing", "products"]
}
```

3. **Add documents**:
- `faqs.json` - FAQ entries
- `products.json` - Product descriptions
- `policies.txt` - Policy documents

4. **Update config.json**:
Add your company to `knowledge_bases/config.json`

5. **Load company**:
```bash
curl -X POST "http://localhost:8000/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{"company_id": "mycompany"}'
```

## Configuration

Key settings in `.env`:

```env
# Embedding Model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Chunking Strategy
CHUNK_SIZE_MIN=200
CHUNK_SIZE_TARGET=450
CHUNK_SIZE_MAX=600
CHUNK_OVERLAP=75

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.3
RERANK_ENABLED=True

# Performance
GPU_ENABLED=False  # Set to True if GPU available
MAX_CONCURRENT_REQUESTS=50
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chunking.py -v
```

## Performance Benchmarks

- **Query latency**: <200ms average
- **Throughput**: 50+ concurrent requests
- **Memory usage**: <500MB per company
- **Startup time**: <30 seconds
- **Retrieval accuracy**: >80% on test set

## Project Structure

```
knowledge-base-system/
├── app/
│   ├── core/               # Core RAG components
│   │   ├── chunking.py     # Document chunking
│   │   ├── embeddings.py   # Embedding generation
│   │   ├── retrieval.py    # RAG retrieval
│   │   └── vector_store.py # FAISS vector store
│   ├── services/           # Business logic
│   │   ├── document_processor.py
│   │   └── knowledge_service.py
│   ├── api/                # FastAPI endpoints
│   │   ├── endpoints.py
│   │   └── dependencies.py
│   ├── models/             # Pydantic schemas
│   ├── utils/              # Utilities
│   ├── config.py           # Configuration
│   └── main.py             # FastAPI app
├── knowledge_bases/        # Company data
│   ├── techstore/
│   ├── healthplus/
│   └── edulearn/
├── tests/                  # Test suite
├── indexes/                # Saved vector indexes
├── logs/                   # Application logs
├── requirements.txt
├── .env
└── run.py
```

## Sample Companies

The system includes 3 sample companies with realistic data:

1. **TechStore** - Electronics retailer (25 FAQs, 15 products, policies)
2. **HealthPlus** - Healthcare service (30 FAQs, 10 services, policies)
3. **EduLearn** - Online education (20 FAQs, 12 courses, policies)

## Troubleshooting

**Issue**: "Model not found"
```bash
# Manually download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Issue**: "Port already in use"
```bash
# Change port in .env
PORT=8001
```

**Issue**: "NLTK punkt not found"
```bash
python -c "import nltk; nltk.download('punkt')"
```

## Production Deployment

1. **Disable debug mode**:
```env
DEBUG=False
```

2. **Use production server**:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

3. **Enable GPU** (if available):
```env
GPU_ENABLED=True
```

4. **Set up monitoring**:
- Monitor `/api/v1/health` endpoint
- Track `/api/v1/stats` for metrics

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check the `/docs` endpoint for API documentation
- Review logs in the `logs/` directory

## Roadmap

- [ ] ChromaDB integration
- [ ] Redis caching
- [ ] Async processing
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Webhook support for updates

## Authors

Built for AI voice agent customer service applications.
