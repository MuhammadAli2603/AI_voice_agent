# AI Voice Agent - Telephony Integration Setup Guide

Complete guide to setting up the AI voice agent with Asterisk telephony integration.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: 8GB minimum (16GB recommended)
- **CPU**: 4+ cores
- **Disk**: 20GB available
- **Network**: Static IP for SIP trunking

### Software Dependencies
- Python 3.9+
- Asterisk 18+ or 20+
- Redis 6+
- FFmpeg
- libsndfile

---

## 🚀 Quick Start (15 Minutes)

### Step 1: Install Asterisk

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y asterisk asterisk-core-sounds-en asterisk-moh-opsound-wav

# Verify installation
asterisk -V
# Should show: Asterisk 18.x or higher

# Start Asterisk
sudo systemctl start asterisk
sudo systemctl enable asterisk
```

### Step 2: Install Python Dependencies

```bash
cd /path/to/AI_voice_agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install additional telephony dependencies
pip install pyst2==0.5.1 \
            webrtcvad==2.0.10 \
            silero-vad==4.0.0 \
            torch==2.1.0 \
            pydub==0.25.1 \
            redis==5.0.1 \
            prometheus-client==0.19.0
```

### Step 3: Install Redis

```bash
# Ubuntu/Debian
sudo apt-get install -y redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Test
redis-cli ping
# Should return: PONG
```

### Step 4: Configure Asterisk

```bash
# Copy configuration files
sudo cp telephony/config/extensions.conf /etc/asterisk/extensions.conf

# Edit paths in configuration
sudo nano /etc/asterisk/extensions.conf
# Update VOICE_AGENT_PATH and PYTHON_BIN to your actual paths

# Reload Asterisk
sudo asterisk -rx "dialplan reload"
```

### Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env
nano .env
```

Add your Hugging Face API key:
```bash
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxx
KB_SERVICE_URL=http://localhost:8001
```

### Step 6: Start Services

```bash
# Terminal 1: Knowledge Base Service
cd knowledge-base-system
python run.py

# Terminal 2: Voice Agent API
cd ..
python -m app.main

# Terminal 3: Metrics Server
python -m app.monitoring.metrics

# Terminal 4: Test call
# (Connect a SIP phone to extension 100)
```

---

## 📞 Testing Your Setup

### Test 1: Echo Test

```bash
# Connect SIP phone and dial 200
# You should hear your own voice echoed back
```

### Test 2: AI Voice Agent Test

```bash
# Dial extension 100
# Expected flow:
# 1. Call answered
# 2. Welcome message
# 3. AI asks how it can help
# 4. Speak your question
# 5. AI responds with knowledge base context
# 6. Conversation continues
# 7. Say "goodbye" to end call
```

### Test 3: Company-Specific Routing

```bash
# Test TechStore (extension 18005551234)
# Should load TechStore knowledge base

# Test HealthPlus (extension 18005555678)
# Should load HealthPlus knowledge base

# Test EduLearn (extension 18005559999)
# Should load EduLearn knowledge base
```

---

## 🔧 Detailed Configuration

### Asterisk SIP Configuration

Create `/etc/asterisk/pjsip.conf`:

```ini
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0:5060

[transport-tcp]
type=transport
protocol=tcp
bind=0.0.0.0:5060

; Internal SIP phone example
[6001]
type=endpoint
context=testing
disallow=all
allow=ulaw
allow=alaw
auth=6001
aors=6001

[6001]
type=auth
auth_type=userpass
password=secret123
username=6001

[6001]
type=aor
max_contacts=1

; Repeat for 6002, 6003, etc.
```

### Audio Prompts

Create custom audio prompts in `/var/lib/asterisk/sounds/en/`:

```bash
# Welcome message
sox -n -r 8000 -c 1 welcome.wav synth 2 sine 1000

# Or use TTS to generate:
# "Welcome to TechStore. How can I help you today?"
```

### Firewall Configuration

```bash
# Allow SIP and RTP
sudo ufw allow 5060/udp
sudo ufw allow 5060/tcp
sudo ufw allow 10000:20000/udp  # RTP media
```

---

## 📊 Monitoring

### Prometheus Metrics

Access metrics at: `http://localhost:9090/metrics`

Key metrics:
- `voice_agent_active_calls` - Current active calls
- `voice_agent_total_calls` - Total calls by status
- `voice_agent_stt_latency_seconds` - STT processing time
- `voice_agent_llm_latency_seconds` - LLM response time
- `voice_agent_kb_confidence` - Knowledge base confidence

### Grafana Dashboard (Optional)

1. Install Grafana:
```bash
sudo apt-get install -y grafana
sudo systemctl start grafana-server
```

2. Access Grafana: `http://localhost:3000` (admin/admin)

3. Add Prometheus data source: `http://localhost:9090`

4. Import dashboard from `app/monitoring/dashboard.json`

### Log Files

```bash
# Voice agent logs
tail -f logs/voice_agent.log

# Asterisk logs
tail -f /var/log/asterisk/full

# Call session logs
tail -f logs/calls_$(date +%Y-%m-%d).jsonl

# AGI errors
tail -f /var/log/asterisk/agi_error.log
```

---

## 🔍 Troubleshooting

### Issue: Call not connecting

**Check:**
```bash
# Verify Asterisk is running
sudo asterisk -rx "core show version"

# Check SIP registrations
sudo asterisk -rx "pjsip show endpoints"

# Check active channels
sudo asterisk -rx "core show channels"
```

**Solution:**
- Ensure SIP phone registered
- Check firewall rules
- Verify extensions.conf syntax

### Issue: No audio from AI

**Check:**
```bash
# Test Python path
which python
/opt/ai-voice-agent/venv/bin/python

# Test AGI script manually
cd /opt/ai-voice-agent
./venv/bin/python telephony/asterisk_agi.py
# Should not error immediately
```

**Solution:**
- Verify Python virtual environment activated
- Check file permissions
- Test audio bridge conversion

### Issue: High latency (>10s)

**Check:**
```bash
# Monitor metrics
curl http://localhost:9090/metrics | grep latency

# Check API key
echo $HUGGINGFACE_API_KEY

# Test KB service
curl http://localhost:8001/api/v1/health
```

**Solution:**
- Verify fast internet connection
- Check Hugging Face API status
- Consider local model deployment
- Optimize knowledge base queries

### Issue: Barge-in not working

**Current Status:** Basic barge-in detection implemented but not fully integrated with AGI streaming.

**Workaround:**
- User can press `#` to end recording early
- Future: Implement full duplex audio streaming with ARI

---

## 🔐 Security Hardening

### 1. Restrict SIP Access

```bash
# /etc/asterisk/pjsip.conf
[transport-udp]
type=transport
protocol=udp
bind=YOUR_SERVER_IP:5060
local_net=192.168.1.0/24  # Your internal network
external_media_address=YOUR_PUBLIC_IP
external_signaling_address=YOUR_PUBLIC_IP
```

### 2. Enable Fail2Ban

```bash
sudo apt-get install fail2ban

# Create /etc/fail2ban/jail.d/asterisk.conf
[asterisk]
enabled = true
port = 5060,5061
filter = asterisk
logpath = /var/log/asterisk/full
maxretry = 3
bantime = 86400
```

### 3. Use TLS/SRTP

```bash
# Generate certificates
sudo mkdir -p /etc/asterisk/keys
sudo openssl req -x509 -newkey rsa:4096 -keyout /etc/asterisk/keys/asterisk.key -out /etc/asterisk/keys/asterisk.crt -days 365 -nodes

# Enable in pjsip.conf
[transport-tls]
type=transport
protocol=tls
bind=0.0.0.0:5061
cert_file=/etc/asterisk/keys/asterisk.crt
priv_key_file=/etc/asterisk/keys/asterisk.key
```

---

## 📈 Performance Tuning

### Increase Concurrent Calls

```python
# telephony/call_session.py
session_manager = CallSessionManager(
    max_concurrent_calls=50  # Increase from 20
)
```

### Optimize Audio Processing

```bash
# Use faster whisper model
STT_MODEL=openai/whisper-base  # Instead of large-v3

# Reduce LLM tokens
LLM_MAX_NEW_TOKENS=100  # Instead of 150
```

### Redis Performance

```bash
# /etc/redis/redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

---

## 🐳 Docker Deployment (Advanced)

```bash
# Build and start all services
docker-compose -f docker/docker-compose-telephony.yml up -d

# View logs
docker-compose logs -f voice-agent

# Stop services
docker-compose down
```

---

## 📚 Additional Resources

- Asterisk Documentation: https://wiki.asterisk.org
- PYST2 Library: https://github.com/rdegges/pyst2
- Silero VAD: https://github.com/snakers4/silero-vad
- Prometheus: https://prometheus.io/docs

---

## 🆘 Getting Help

**Logs to include when reporting issues:**
```bash
# Collect diagnostic info
sudo asterisk -rx "core show version" > diagnostics.txt
sudo asterisk -rx "pjsip show endpoints" >> diagnostics.txt
tail -100 /var/log/asterisk/full >> diagnostics.txt
tail -100 logs/voice_agent.log >> diagnostics.txt
```

**Common Questions:**
- Q: Can I use FreeSWITCH instead?
  - A: Yes, create `telephony/freeswitch_esl.py` following similar pattern

- Q: How do I add more companies?
  - A: Add knowledge bases in `knowledge-base-system/knowledge_bases/` and update DID routing

- Q: Can I use local models instead of API?
  - A: Yes, replace HF API calls with local Whisper/LLaMA inference

---

## ✅ Production Checklist

Before going live:

- [ ] Configure SIP trunk with provider
- [ ] Set up DID number routing
- [ ] Create company-specific knowledge bases
- [ ] Configure call recording (with consent)
- [ ] Set up monitoring and alerts
- [ ] Test failover scenarios
- [ ] Document call flows
- [ ] Train support team
- [ ] Perform load testing
- [ ] Review security settings
- [ ] Set up backup/disaster recovery

---

**You're now ready to handle real phone calls with AI! 🎉📞**
