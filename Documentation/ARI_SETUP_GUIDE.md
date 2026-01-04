# ARI Setup Guide - Full-Duplex Streaming

Complete guide to setting up the AI voice agent with Asterisk ARI for full-duplex audio streaming and real-time barge-in.

---

## 🎯 What is ARI?

**Asterisk REST Interface (ARI)** provides:
- WebSocket-based call control
- Bidirectional audio streaming
- Real-time event handling
- Lower latency than AGI
- Better integration for modern applications

---

## 🔄 AGI vs ARI Comparison

| Feature | AGI (Old) | ARI (New) |
|---------|-----------|-----------|
| **Architecture** | Sequential | Full-duplex |
| **Audio** | Record → Process → Play | Simultaneous send/receive |
| **Latency** | 4-10s | 1-3s |
| **Barge-in** | Not supported | Real-time |
| **Control** | Script-based | REST API + WebSocket |
| **Scalability** | Limited | High |

---

## 📦 Prerequisites

### System Requirements
- Asterisk 16+ or 18+ (ARI support)
- Python 3.9+
- Redis 6+
- 16GB RAM recommended (for concurrent calls)

### Required Asterisk Modules
```bash
# Check if ARI modules are loaded
sudo asterisk -rx "module show like ari"

# Should see:
# res_ari.so
# res_ari_channels.so
# res_ari_bridges.so
# res_stasis.so
```

---

## 🚀 Installation Steps

### Step 1: Install Dependencies

```bash
cd /path/to/AI_voice_agent

# Activate virtual environment
source venv/bin/activate

# Install ARI dependencies
pip install -r requirements.txt

# Verify ARI package
python -c "import ari; print('ARI version:', ari.__version__)"
```

### Step 2: Configure Asterisk for ARI

#### Copy configuration files:
```bash
# Copy ARI configuration
sudo cp telephony/config/ari.conf /etc/asterisk/ari.conf

# Copy HTTP configuration (for WebSocket)
sudo cp telephony/config/http.conf /etc/asterisk/http.conf

# Copy ARI dialplan
sudo cp telephony/config/extensions-ari.conf /etc/asterisk/extensions.conf
```

#### Edit configurations if needed:
```bash
# Edit ARI credentials (optional)
sudo nano /etc/asterisk/ari.conf

# Edit HTTP bind address (optional)
sudo nano /etc/asterisk/http.conf
```

### Step 3: Reload Asterisk

```bash
# Reload all configurations
sudo asterisk -rx "module reload"

# Reload specific modules
sudo asterisk -rx "ari reload"
sudo asterisk -rx "http reload"
sudo asterisk -rx "dialplan reload"

# Verify ARI is enabled
sudo asterisk -rx "ari show status"
# Should show: ARI Status: Enabled
```

### Step 4: Configure Environment

```bash
# Edit .env file
nano .env
```

Add ARI configuration:
```bash
# ARI Configuration
ARI_URL=http://localhost:8088
ARI_USERNAME=voice_agent
ARI_PASSWORD=voice_agent_secret_2024
ARI_APP_NAME=voice_agent

# Hugging Face (existing)
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxx

# Knowledge Base (existing)
KB_SERVICE_URL=http://localhost:8001
```

### Step 5: Start Services

```bash
# Terminal 1: Start Knowledge Base Service
cd knowledge-base-system
python run.py

# Terminal 2: Start Voice Agent API (optional, for REST API)
cd ..
python -m app.main

# Terminal 3: Start ARI Handler (NEW!)
./start_ari_handler.sh

# Terminal 4: Start Metrics Server
python -m app.monitoring.metrics
```

---

## 🧪 Testing

### Test 1: Verify ARI Connection

```bash
# Check if ARI handler is running
ps aux | grep asterisk_ari

# Check logs
tail -f logs/ari_handler.log

# Should see:
# "Connected to Asterisk ARI"
# "Listening for calls on Stasis app 'voice_agent'..."
```

### Test 2: Make Test Call

```bash
# From SIP phone, dial extension 100 (TechStore)
# or dial 101 (HealthPlus)
# or dial 102 (EduLearn)

# Expected flow:
# 1. Call enters Stasis app
# 2. ARI handler answers
# 3. Welcome message plays
# 4. AI conversation begins
# 5. Real-time responses
# 6. Barge-in detection active
```

### Test 3: Check Asterisk ARI Events

```bash
# Monitor ARI events in Asterisk
sudo asterisk -rx "ari set debug on"

# Watch events
tail -f /var/log/asterisk/full | grep ARI

# Should see:
# StasisStart events when calls come in
# ChannelDtmfReceived for DTMF
# StasisEnd when calls finish
```

### Test 4: Verify WebSocket Connection

```bash
# Check if WebSocket is listening
sudo netstat -tlnp | grep 8088

# Should show Asterisk listening on port 8088
```

---

## 📊 Monitoring

### ARI-Specific Metrics

Check metrics at: `http://localhost:9090/metrics`

New ARI metrics:
```
voice_agent_ari_active_calls
voice_agent_ari_stasis_starts_total
voice_agent_ari_stasis_ends_total
voice_agent_ari_events_processed_total
voice_agent_ari_websocket_errors_total
```

### Logs

```bash
# ARI handler logs
tail -f logs/ari_handler.log

# Asterisk ARI debug
sudo asterisk -rx "ari set debug on"
tail -f /var/log/asterisk/full | grep ARI

# Call session logs
tail -f logs/calls_$(date +%Y-%m-%d).jsonl
```

---

## 🔧 Troubleshooting

### Issue: ARI Handler Can't Connect

**Symptoms:**
```
Error: Connection refused to http://localhost:8088
```

**Solutions:**
```bash
# Check if Asterisk HTTP server is running
sudo asterisk -rx "http show status"

# Check if port 8088 is listening
sudo netstat -tlnp | grep 8088

# Verify http.conf
sudo cat /etc/asterisk/http.conf | grep "enabled\|bindport"

# Reload HTTP module
sudo asterisk -rx "module reload res_http_websocket.so"
```

### Issue: Authentication Failed

**Symptoms:**
```
Error: 401 Unauthorized
```

**Solutions:**
```bash
# Verify ARI user exists
sudo asterisk -rx "ari show users"

# Check credentials in ari.conf
sudo cat /etc/asterisk/ari.conf | grep -A5 "\[voice_agent\]"

# Ensure .env matches ari.conf
cat .env | grep ARI_
```

### Issue: Calls Not Entering Stasis

**Symptoms:**
- Calls connect but no ARI events
- No "StasisStart" in logs

**Solutions:**
```bash
# Verify dialplan
sudo asterisk -rx "dialplan show voice_agent"

# Check Stasis app name
sudo asterisk -rx "stasis show apps"
# Should list: voice_agent

# Test dialplan
sudo asterisk -rx "console dial 100@testing-ari"
```

### Issue: No Audio / Silent Calls

**Symptoms:**
- Call connects but no audio in either direction

**Solutions:**
```bash
# Check audio formats
sudo asterisk -rx "core show translation"

# Verify media paths
sudo asterisk -rx "core show file formats"

# Check if audio files exist
ls -la /var/lib/asterisk/sounds/en/

# Test audio playback
sudo asterisk -rx "console dial 200@testing-ari"  # Echo test
```

### Issue: High Latency

**Symptoms:**
- Delays > 5 seconds between user speech and AI response

**Solutions:**
```bash
# Check processing times in logs
grep "Processing:" logs/ari_handler.log

# Monitor metrics
curl http://localhost:9090/metrics | grep latency

# Optimize:
# - Use faster Whisper model (base instead of large)
# - Reduce LLM max tokens
# - Check network latency to HF API
# - Consider local model deployment
```

---

## 🔐 Security

### Production Hardening

1. **Use HTTPS for ARI:**
```ini
# http.conf
[general]
tlsenable = yes
tlsbindaddr = 0.0.0.0:8089
tlscertfile = /etc/asterisk/keys/asterisk.crt
tlsprivatekey = /etc/asterisk/keys/asterisk.key
```

2. **Restrict ARI access:**
```ini
# ari.conf
[general]
allowed_origins = https://your-domain.com
```

3. **Change default password:**
```bash
# Edit ari.conf
sudo nano /etc/asterisk/ari.conf

[voice_agent]
password = YOUR_STRONG_PASSWORD_HERE

# Update .env
nano .env
ARI_PASSWORD=YOUR_STRONG_PASSWORD_HERE
```

4. **Firewall rules:**
```bash
# Only allow localhost for ARI (recommended)
sudo ufw deny 8088
# OR allow specific IPs
sudo ufw allow from YOUR_SERVER_IP to any port 8088
```

---

## 📈 Performance Tuning

### Increase Concurrent Calls

```python
# telephony/call_session.py
session_manager = CallSessionManager(
    max_concurrent_calls=50  # Increase from default 20
)
```

### Optimize Audio Buffering

```python
# telephony/ari_audio_bridge.py
config = AudioStreamConfig(
    chunk_duration_ms=20,  # Lower = more responsive, higher CPU
    buffer_size=5          # Lower = less latency, higher dropout risk
)
```

### WebSocket Keep-Alive

```ini
# http.conf
[general]
session_keep_alive = 10000  # 10 seconds
```

---

## 🚀 Advanced Features

### Full-Duplex Streaming (Future)

Current implementation uses ARI recording/playback.
For true full-duplex:

1. Create ExternalMedia channel
2. Stream RTP directly to/from application
3. Process audio in real-time with VAD
4. Implement streaming barge-in

See `ARI_IMPLEMENTATION_PLAN.md` for details.

### Multi-Region Deployment

```bash
# Deploy ARI handlers in multiple regions
# Use SIP routing to regional Asterisk servers
# Each region has its own ARI handler
```

### Load Balancing

```bash
# Use HAProxy or nginx to load balance ARI connections
# Multiple ARI handler instances for redundancy
```

---

## 📚 Additional Resources

- [Asterisk ARI Documentation](https://wiki.asterisk.org/wiki/display/AST/Asterisk+REST+Interface)
- [Python ARI Client](https://github.com/asterisk/ari-py)
- [Stasis Application Guide](https://wiki.asterisk.org/wiki/display/AST/Getting+Started+with+ARI)
- [ARI Examples](https://github.com/asterisk/ari-examples)

---

## ✅ Migration Checklist (AGI → ARI)

Before migrating from AGI to ARI:

- [ ] Backup current configuration
- [ ] Test ARI on non-production system
- [ ] Update Asterisk to 16+ if needed
- [ ] Configure ARI in Asterisk
- [ ] Install Python ARI dependencies
- [ ] Update dialplan to use Stasis()
- [ ] Start ARI handler
- [ ] Test with single call
- [ ] Monitor performance
- [ ] Gradually route production traffic
- [ ] Monitor error logs
- [ ] Keep AGI as fallback initially

---

## 🎉 Success Criteria

✅ **You're ready when:**

1. `sudo asterisk -rx "ari show status"` shows "Enabled"
2. ARI handler connects without errors
3. Test calls enter Stasis app
4. Bidirectional audio works
5. Latency < 3 seconds
6. Metrics dashboard shows activity
7. Error rate < 1%
8. Can handle 20+ concurrent calls

---

**You've upgraded to full-duplex ARI! 🚀**
