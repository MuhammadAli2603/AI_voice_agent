# Knowledge Base Integration Guide

The AI Voice Agent now integrates with the Knowledge Base RAG system to provide context-aware, accurate responses based on company-specific information.

## Overview

When a `company_id` is provided in the request, the LLM will:
1. Query the Knowledge Base for relevant context
2. Inject the retrieved information into the prompt
3. Generate responses based on actual company data
4. Fall back to general responses if KB is unavailable

## Setup

### 1. Start Both Services

**Terminal 1: Knowledge Base Service**
```bash
cd knowledge-base-system
python run.py
# Runs on http://localhost:8001
```

**Terminal 2: Voice Agent Service**
```bash
python -m app.main
# Runs on http://localhost:8000
```

### 2. Load a Company Knowledge Base

```bash
curl -X POST "http://localhost:8001/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{
    "company_id": "techstore",
    "force_reindex": false
  }'
```

Available companies:
- `techstore` - Electronics retailer (products, FAQs, policies)
- `healthplus` - Healthcare service (services, appointments, FAQs)
- `edulearn` - Online education (courses, enrollment, FAQs)

### 3. Configure Knowledge Base URL (Optional)

If KB service runs on a different host/port:

```bash
# In .env file
KB_SERVICE_URL=http://kb-service:8001
```

## Usage Examples

### Example 1: Chat Endpoint with KB Context

**Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is your return policy?",
    "company_id": "techstore"
  }'
```

**Response:**
```json
{
  "response": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with all packaging and accessories. Refunds are processed within 5-7 business days.",
  "processing_time": 3.2,
  "kb_enabled": true
}
```

**Without KB (no company_id):**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is your return policy?"
  }'
```

**Response:**
```json
{
  "response": "I'd be happy to help with return information. Could you please specify which company or product you're asking about?",
  "processing_time": 2.1,
  "kb_enabled": false
}
```

### Example 2: Complete Voice Agent Pipeline

**Request:**
```bash
curl -X POST "http://localhost:8000/voice-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "<base64_encoded_audio>",
    "language": "en",
    "company_id": "healthplus"
  }'
```

**Response:**
```json
{
  "transcription": "How do I schedule an appointment?",
  "llm_response": "You can schedule an appointment by calling our booking line at 1-800-HEALTH or through our online portal at healthplus.com/book. Appointments are available Monday-Friday 8am-6pm.",
  "audio_base64": "<synthesized_response_audio>",
  "total_processing_time": 8.5,
  "breakdown": {
    "stt_time": 2.3,
    "llm_time": 4.8,
    "tts_time": 1.4
  },
  "kb_enabled": true
}
```

### Example 3: WebSocket with KB Context

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  // Send audio with company_id
  ws.send(JSON.stringify({
    type: 'audio',
    data: 'base64_audio_data',
    language: 'en',
    company_id: 'edulearn'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'transcription') {
    console.log('User said:', data.text);
  }

  if (data.type === 'llm_response') {
    console.log('Agent response:', data.text);
  }

  if (data.type === 'audio_response') {
    // Play synthesized audio
    playAudio(data.data);
  }
};
```

### Example 4: File Upload with KB

```bash
curl -X POST "http://localhost:8000/voice-agent/file" \
  -F "file=@question.wav" \
  -F "language=en" \
  -F "company_id=techstore"
```

## How It Works

### Data Flow with KB Integration

```
1. User Request
   ↓
2. [STT] Audio → Text
   ↓
3. [KB Query] "What is your return policy?" + company_id="techstore"
   ↓
4. Knowledge Base Service
   - Generate query embedding
   - Search FAISS vector store
   - Retrieve top 3 relevant chunks
   - Return formatted context
   ↓
5. [LLM] Enhanced Prompt
   System: "You are a friendly receptionist..."

   Relevant Information from Knowledge Base:
   [Chunk 1] TechStore Return Policy: Returns accepted within 30 days...
   [Chunk 2] Refund Processing: Refunds processed in 5-7 business days...
   [Chunk 3] Return Conditions: Items must be unopened and in original packaging...

   Use the above information to answer accurately.

   User: What is your return policy?
   Assistant:
   ↓
6. [LLM API] Generate response based on KB context
   ↓
7. [TTS] Text → Audio
   ↓
8. Return Response
```

### KB Query Details

**Request to KB Service:**
```json
POST http://localhost:8001/api/v1/context
{
  "company_id": "techstore",
  "query": "What is your return policy?",
  "top_k": 3,
  "include_metadata": false
}
```

**Response from KB Service:**
```json
{
  "context": "You are a customer service agent for TechStore.\n\nRelevant Information:\n\n[Source 1]\nOur return policy allows returns within 30 days of purchase...\n\n[Source 2]\nRefunds are processed within 5-7 business days...\n\n[Source 3]\nAll items must be in original condition...",
  "confidence_score": 0.87,
  "sources": ["faqs.json", "policies.txt"],
  "total_results": 3,
  "retrieval_time": 0.045
}
```

## Configuration

### LLM Module Configuration

```python
# app/modules/llm/receptionist_llm.py
class ReceptionistLLM:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        api_key: str = "",
        kb_service_url: str = "http://localhost:8001"  # KB service URL
    ):
        self.kb_service_url = kb_service_url
```

### KB Query Parameters

```python
def _query_knowledge_base(
    self,
    query: str,
    company_id: str,
    top_k: int = 3  # Number of context chunks to retrieve
):
    # Query KB service
    # Timeout: 5 seconds
    # Confidence threshold: 0.3 (30%)
```

## Testing the Integration

### Test 1: KB Available

```bash
# Start KB service
cd knowledge-base-system && python run.py

# Load company data
curl -X POST "http://localhost:8001/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{"company_id": "techstore"}'

# Test chat with KB
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Do you sell laptops?",
    "company_id": "techstore"
  }'

# Expected: Response about TechStore laptop products
```

### Test 2: KB Unavailable (Graceful Fallback)

```bash
# Stop KB service (Ctrl+C in KB terminal)

# Test chat without KB
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What products do you sell?",
    "company_id": "techstore"
  }'

# Expected: Generic response (KB query fails gracefully)
```

### Test 3: No Company ID

```bash
# Test without company_id
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how can you help?"
  }'

# Expected: Generic receptionist response (no KB query)
```

## Monitoring

### KB Query Logs

```
# Voice Agent logs (when KB is queried)
2026-01-03 15:23:45 | INFO | Querying knowledge base for company: techstore
2026-01-03 15:23:45 | INFO | KB context retrieved (confidence: 0.87)
2026-01-03 15:23:48 | INFO | ✓ LLM Response: 'Our return policy...' (took 2.8s)
```

### KB Service Logs

```
# Knowledge Base service logs
2026-01-03 15:23:45 | INFO | Request: POST /api/v1/context Status: 200 Time: 0.045s
2026-01-03 15:23:45 | INFO | Retrieved 3 chunks for company techstore in 0.045s - Confidence: 0.87
```

## Troubleshooting

### Issue: KB queries always fail

**Symptoms:**
- Logs show "Knowledge base unavailable"
- Responses are generic despite providing company_id

**Solutions:**
1. Verify KB service is running: `curl http://localhost:8001/api/v1/health`
2. Check KB_SERVICE_URL in .env matches actual KB service URL
3. Ensure company is loaded: `curl http://localhost:8001/api/v1/companies`
4. Check network connectivity between services

### Issue: Low confidence scores

**Symptoms:**
- KB returns context but confidence < 0.3
- Responses don't use KB context

**Solutions:**
1. Improve query quality (more specific questions)
2. Lower confidence threshold in `_query_knowledge_base()` (line 117)
3. Add more relevant documents to knowledge base
4. Re-index knowledge base with force_reindex=true

### Issue: Wrong company context

**Symptoms:**
- Getting TechStore responses when asking about HealthPlus

**Solutions:**
1. Verify correct company_id in request
2. Check company is loaded: `curl http://localhost:8001/api/v1/companies`
3. Clear and reload company:
   ```bash
   curl -X DELETE "http://localhost:8001/api/v1/company/techstore"
   curl -X POST "http://localhost:8001/api/v1/company/load" -d '{"company_id":"techstore"}'
   ```

## Performance Considerations

### Latency Impact

**Without KB:**
- STT: 1-3s
- LLM: 2-5s
- TTS: 1-2s
- **Total: 4-10s**

**With KB:**
- STT: 1-3s
- KB Query: 0.05-0.2s (negligible)
- LLM: 2-5s
- TTS: 1-2s
- **Total: 4-10s** (KB adds <200ms)

### Optimization Tips

1. **Run KB service locally** for minimum latency
2. **Use GPU for KB embeddings** (10x faster): `GPU_ENABLED=True` in KB .env
3. **Cache frequently asked questions** (already implemented in KB)
4. **Reduce top_k** if responses are too long (default: 3)
5. **Increase KB timeout** for slower networks (default: 5s)

## Adding Custom Companies

### 1. Create Company Directory

```bash
mkdir knowledge-base-system/knowledge_bases/mycompany
```

### 2. Add Metadata

```json
// knowledge_bases/mycompany/metadata.json
{
  "company_id": "mycompany",
  "company_name": "My Company",
  "description": "Company description",
  "active": true,
  "categories": ["support", "sales", "products"]
}
```

### 3. Add Knowledge Files

```json
// knowledge_bases/mycompany/faqs.json
[
  {
    "question": "What are your business hours?",
    "answer": "We're open Monday-Friday 9am-5pm EST.",
    "category": "general",
    "priority": "high"
  }
]
```

### 4. Register Company

```json
// knowledge_bases/config.json
{
  "companies": [
    ...,
    {
      "company_id": "mycompany",
      "company_name": "My Company",
      "description": "Company description",
      "active": true,
      "categories": ["support", "sales"]
    }
  ]
}
```

### 5. Load Company

```bash
curl -X POST "http://localhost:8001/api/v1/company/load" \
  -H "Content-Type: application/json" \
  -d '{"company_id": "mycompany"}'
```

### 6. Test

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your business hours?",
    "company_id": "mycompany"
  }'
```

## Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  knowledge-base:
    build: ./knowledge-base-system
    container_name: kb-service
    ports:
      - "8001:8000"
    environment:
      - GPU_ENABLED=False
    volumes:
      - ./knowledge-base-system/knowledge_bases:/app/knowledge_bases
      - kb-indexes:/app/indexes
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  voice-agent:
    build: .
    container_name: voice-agent
    ports:
      - "8000:8000"
    environment:
      - KB_SERVICE_URL=http://knowledge-base:8000
      - HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY}
    depends_on:
      - knowledge-base
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  kb-indexes:
```

### Usage

```bash
# Build and start
docker-compose up -d

# Load company (after services start)
sleep 30  # Wait for services to be ready
curl -X POST "http://localhost:8001/api/v1/company/load" \
  -d '{"company_id": "techstore"}'

# Test
curl -X POST "http://localhost:8000/chat" \
  -d '{"message": "What products do you sell?", "company_id": "techstore"}'
```

## API Reference

### POST /chat
Generate chat response with optional KB context

**Request:**
```json
{
  "message": "string (required)",
  "conversation_history": [],  // optional
  "company_id": "string"  // optional
}
```

**Response:**
```json
{
  "response": "string",
  "processing_time": float,
  "kb_enabled": boolean
}
```

### POST /voice-agent
Complete pipeline with optional KB context

**Request:**
```json
{
  "audio_base64": "string (required)",
  "language": "en",
  "conversation_history": [],
  "company_id": "string"  // optional
}
```

**Response:**
```json
{
  "transcription": "string",
  "llm_response": "string",
  "audio_base64": "string",
  "total_processing_time": float,
  "breakdown": {
    "stt_time": float,
    "llm_time": float,
    "tts_time": float
  },
  "kb_enabled": boolean
}
```

### WebSocket /ws
Real-time voice streaming with KB

**Send:**
```json
{
  "type": "audio",
  "data": "base64_audio",
  "language": "en",
  "company_id": "techstore"  // optional
}
```

**Receive:**
```json
{
  "type": "audio_response",
  "data": "base64_audio",
  "timing": {...}
}
```

---

**Integration Complete!** Your AI Voice Agent now uses the Knowledge Base for accurate, context-aware responses.
