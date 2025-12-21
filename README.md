# 🎤 AI Voice Agent - Complete Open Source Solution

A fully modular, open-source AI voice agent built with Hugging Face models. Features Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) capabilities with a friendly receptionist personality.

## 🌟 Features

- **Modular Architecture**: Independent STT, LLM, and TTS modules
- **100% Open Source**: All models from Hugging Face
- **Friendly Receptionist**: Warm, professional, and helpful AI personality
- **FastAPI Backend**: High-performance async API
- **LangChain Integration**: Advanced conversation management
- **WebSocket Support**: Real-time voice streaming
- **Multiple Interfaces**: REST API and WebSocket

## 🏗️ Architecture

```
User Speech → STT (Whisper) → LLM (Receptionist) → TTS (Coqui) → AI Voice Response
```

### Modules

1. **STT Module** - Speech-to-Text using OpenAI Whisper
2. **LLM Module** - Conversational AI with receptionist personality
3. **TTS Module** - Text-to-Speech using Coqui TTS

## 📋 Prerequisites

- Python 3.14.0
- CUDA-capable GPU (optional, for faster processing)
- 8GB+ RAM recommended

## 🚀 Installation

### 1. Clone or Create Project

```bash
# Create project directory
mkdir ai-voice-agent
cd ai-voice-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## ⚙️ Configuration

Edit `.env` file:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Models (all from Hugging Face)
STT_MODEL=openai/whisper-base  # Options: whisper-tiny, base, small, medium, large
LLM_MODEL=microsoft/DialoGPT-medium  # Or any conversational model
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC

# Device (cpu, cuda, mps)
DEVICE=cpu

# Audio
SAMPLE_RATE=16000

# LLM Settings
LLM_MAX_LENGTH=512
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
```

## 🎯 Usage

### Start the Server

```bash
# From project root
python -m app.main

# Or with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Health Check

```bash
GET /health
```

### 2. Complete Voice Agent Pipeline

```bash
POST /voice-agent
Content-Type: application/json

{
  "audio_base64": "base64_encoded_audio",
  "language": "en",
  "conversation_history": []
}
```

### 3. Transcription Only (STT)

```bash
POST /transcribe
Content-Type: application/json

{
  "audio_base64": "base64_encoded_audio",
  "language": "en"
}
```

Or upload file:

```bash
POST /transcribe/file
Content-Type: multipart/form-data

file: audio_file.wav
language: en
```

### 4. Chat Only (LLM)

```bash
POST /chat
Content-Type: application/json

{
  "message": "Hello, I need help",
  "conversation_history": []
}
```

### 5. Text-to-Speech Only (TTS)

```bash
POST /synthesize
Content-Type: application/json

{
  "text": "Hello! How can I help you today?",
  "language": "en",
  "speed": 1.0
}
```

### 6. Reset Conversation

```bash
POST /reset
```

## 🔌 WebSocket Usage

Connect to `ws://localhost:8000/ws` for real-time streaming:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

// Send audio
ws.send(JSON.stringify({
  type: 'audio',
  data: 'base64_audio_data',
  language: 'en'
}));

// Receive responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.type, data);
};
```

## 🐍 Python Usage Example

```python
import requests
import base64

# Read audio file
with open('audio.wav', 'rb') as f:
    audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()

# Complete pipeline
response = requests.post(
    'http://localhost:8000/voice-agent',
    json={
        'audio_base64': audio_base64,
        'language': 'en'
    }
)

result = response.json()
print(f"Transcription: {result['transcription']}")
print(f"LLM Response: {result['llm_response']}")
print(f"Processing Time: {result['total_processing_time']:.2f}s")

# Decode audio response
audio_response = base64.b64decode(result['audio_base64'])
with open('response.wav', 'wb') as f:
    f.write(audio_response)
```

## 🧪 Testing Individual Modules

```python
from app.modules.stt.whisper_stt import WhisperSTT
from app.modules.llm.receptionist_llm import ReceptionistLLM
from app.modules.tts.coqui_tts import CoquiTTSModel

# Test STT
stt = WhisperSTT()
result = stt.transcribe("audio.wav")
print(result['text'])

# Test LLM
llm = ReceptionistLLM()
response = llm.generate_response("Hello!")
print(response)

# Test TTS
tts = CoquiTTSModel()
audio = tts.synthesize("Hello, how can I help you?")
```

## 🎨 Customization

### Change LLM Personality

Edit `app/modules/llm/receptionist_llm.py`:

```python
RECEPTIONIST_PROMPT = """Your custom personality here..."""
```

### Use Different Models

1. **STT Models**: Any Whisper variant from HuggingFace
   - `openai/whisper-tiny` (fastest)
   - `openai/whisper-base`
   - `openai/whisper-small`
   - `openai/whisper-medium`
   - `openai/whisper-large-v3` (best quality)

2. **LLM Models**: Any conversational model
   - `microsoft/DialoGPT-medium`
   - `facebook/blenderbot-400M-distill`
   - `google/flan-t5-base`

3. **TTS Models**: List available Coqui TTS models:
   ```python
   from TTS.api import TTS
   print(TTS().list_models())
   ```

## 📊 Performance Tips

1. **Use GPU**: Set `DEVICE=cuda` for 5-10x speedup
2. **Smaller Models**: Use whisper-tiny and smaller LLMs for faster response
3. **Batch Processing**: Process multiple requests together
4. **Caching**: Enable model caching in production

## 🐳 Docker Support

```dockerfile
# Create Dockerfile
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "app.main"]
```

```bash
# Build and run
docker build -t ai-voice-agent .
docker run -p 8000:8000 ai-voice-agent
```

## 🔧 Troubleshooting

### Issue: Models not downloading

```bash
# Set Hugging Face cache
export HF_HOME=/path/to/cache
```

### Issue: CUDA out of memory

```bash
# Use smaller models or CPU
DEVICE=cpu
STT_MODEL=openai/whisper-tiny
```

### Issue: Audio quality problems

```bash
# Adjust sample rate in .env
SAMPLE_RATE=22050  # Higher = better quality
```

## 📝 Project Structure

```
ai-voice-agent/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Configuration
│   ├── api/
│   │   ├── routes.py        # REST endpoints
│   │   └── websocket.py     # WebSocket handler
│   ├── modules/
│   │   ├── stt/            # Speech-to-Text
│   │   ├── llm/            # Language Model
│   │   └── tts/            # Text-to-Speech
│   ├── pipeline/
│   │   └── voice_pipeline.py  # Main orchestrator
│   ├── utils/
│   │   ├── logger.py
│   │   └── audio_utils.py
│   └── models/
│       └── schemas.py       # Pydantic models
├── tests/
├── requirements.txt
├── .env
└── README.md
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

- OpenAI Whisper for STT
- Hugging Face for model hosting
- Coqui TTS for speech synthesis
- FastAPI for the web framework
- LangChain for conversation management

## 📞 Support

For issues or questions:
- Create an issue on GitHub
- Check documentation at `/docs` endpoint

## 🎓 Learn More

- [Whisper Documentation](https://github.com/openai/whisper)
- [Hugging Face Models](https://huggingface.co/models)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)

---

Made with ❤️ for the open-source community