# Quick Start Guide

Get the Knowledge Base System running in 5 minutes!

## Prerequisites

- Python 3.10+
- 4GB RAM minimum

## Installation (3 steps)

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Download Models

This happens automatically on first run, but you can pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "import nltk; nltk.download('punkt')"
```

### 3. Start Server

```bash
python run.py
```

That's it! Server is running at http://localhost:8000

## Quick Test

Open another terminal and try these commands:

```bash
# 1. Check health
curl http://localhost:8000/api/v1/health

# 2. See available companies
curl http://localhost:8000/api/v1/companies

# 3. Load TechStore (sample company)
curl -X POST http://localhost:8000/api/v1/company/load \
  -H "Content-Type: application/json" \
  -d '{"company_id": "techstore", "force_reindex": false}'

# 4. Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "techstore",
    "query": "What is your return policy?",
    "top_k": 3
  }'
```

## Interactive API Documentation

Visit http://localhost:8000/docs to try the API interactively!

## Next Steps

1. **Try other sample companies:**
   - `healthplus` - Healthcare company
   - `edulearn` - Education company

2. **Add your own company:**
   - See SETUP_GUIDE.md for details

3. **Integrate with your application:**
   - Use `/api/v1/context` endpoint for LLM-ready context

## Common Commands

```bash
# Load a company
curl -X POST http://localhost:8000/api/v1/company/load \
  -H "Content-Type: application/json" \
  -d '{"company_id": "COMPANY_ID", "force_reindex": false}'

# Query knowledge base
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"company_id": "COMPANY_ID", "query": "YOUR_QUESTION", "top_k": 5}'

# Get LLM context
curl -X POST http://localhost:8000/api/v1/context \
  -H "Content-Type: application/json" \
  -d '{"company_id": "COMPANY_ID", "query": "YOUR_QUESTION", "top_k": 3}'

# Get statistics
curl http://localhost:8000/api/v1/stats
```

## Docker Quick Start

If you prefer Docker:

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Troubleshooting

**Port 8000 already in use?**
```bash
# Change port in .env
PORT=8001
```

**Out of memory?**
```bash
# Start with one company only
# Reduce EMBEDDING_BATCH_SIZE in .env
```

**Slow queries?**
```bash
# Reduce TOP_K_RESULTS in .env
# Enable GPU if available (GPU_ENABLED=True)
```

## Getting Help

- Full docs: README.md
- Setup guide: SETUP_GUIDE.md
- API docs: http://localhost:8000/docs
- Logs: Check `logs/` directory

Happy building!
