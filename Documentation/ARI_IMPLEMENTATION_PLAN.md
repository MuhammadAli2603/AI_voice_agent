# Asterisk ARI Implementation Plan

## Overview
Upgrade from AGI (sequential) to ARI (full-duplex streaming) for real-time bidirectional audio processing.

## Current Limitations (AGI-based)
1. **Sequential Processing**: Record → Process → Playback (no overlap)
2. **No True Barge-in**: Can't interrupt TTS playback in real-time
3. **High Latency**: 4-10s due to sequential operations
4. **Limited Control**: AGI provides basic call control only

## ARI Advantages
1. **Full-Duplex Streaming**: Simultaneous audio send/receive
2. **Real-Time Barge-in**: Detect interrupts while speaking
3. **Lower Latency**: Streaming reduces wait time
4. **Better Control**: WebSocket-based with rich API
5. **Modern Architecture**: RESTful + WebSocket communication

---

## Architecture Comparison

### Current (AGI)
```
Call → AGI Script → Record (wait) → STT → LLM → TTS → Playback (wait) → Loop
       ↓
    Sequential
    No overlap
```

### New (ARI)
```
Call → ARI WebSocket → ┌─ Incoming Audio Stream → STT → LLM → Response
       ↓                │
    Full-Duplex        └─ Outgoing Audio Stream ← TTS ← (can be interrupted)
    Simultaneous
```

---

## Implementation Steps

### Phase 1: Setup & Infrastructure
1. Install ARI dependencies (`ari-py`, `websockets`)
2. Configure Asterisk for ARI
3. Create ARI application in `ari.conf`
4. Update dialplan for ARI routing

### Phase 2: Core ARI Handler
1. Create `telephony/asterisk_ari.py`
2. Implement WebSocket connection to Asterisk
3. Handle ARI events (StasisStart, ChannelDtmfReceived, etc.)
4. Implement bidirectional audio bridge

### Phase 3: Audio Streaming
1. Create external media channel for audio streaming
2. Implement RTP audio reception (caller → AI)
3. Implement RTP audio transmission (AI → caller)
4. Audio format conversion (8kHz µ-law ↔ 16kHz PCM)

### Phase 4: Barge-in Integration
1. Integrate existing `barge_in.py` with ARI
2. Monitor incoming audio while TTS is playing
3. Cancel TTS stream when interrupt detected
4. Buffer and process interrupt audio

### Phase 5: Session Management
1. Update `call_session.py` for ARI compatibility
2. Track streaming metrics
3. Handle concurrent streams per call
4. Manage WebSocket lifecycle

### Phase 6: Testing & Optimization
1. Test with single call
2. Load test with multiple concurrent calls
3. Measure latency improvements
4. Tune VAD sensitivity
5. Optimize buffer sizes

---

## Technical Details

### Required Packages
```bash
pip install ari==0.1.3
pip install websockets==12.0
pip install aiortc==1.6.0  # For WebRTC/RTP handling
```

### Asterisk Configuration

#### ari.conf
```ini
[general]
enabled = yes
pretty = no

[voice_agent]
type = user
read_only = no
password = voice_agent_secret
```

#### http.conf
```ini
[general]
enabled = yes
bindaddr = 127.0.0.1
bindport = 8088
```

#### extensions.conf
```ini
[voice-agent-ari]
exten => 100,1,NoOp(AI Voice Agent - ARI)
 same => n,Stasis(voice_agent,techstore)
 same => n,Hangup()

exten => 18005551234,1,NoOp(TechStore)
 same => n,Stasis(voice_agent,techstore)
 same => n,Hangup()

exten => 18005555678,1,NoOp(HealthPlus)
 same => n,Stasis(voice_agent,healthplus)
 same => n,Hangup()

exten => 18005559999,1,NoOp(EduLearn)
 same => n,Stasis(voice_agent,edulearn)
 same => n,Hangup()
```

### ARI Event Flow

```
1. StasisStart → Call enters Stasis app
   ↓
2. Answer channel
   ↓
3. Create external media channel
   ↓
4. Connect to WebSocket audio stream
   ↓
5. Start bidirectional audio processing:
   - Incoming: Caller → STT → LLM
   - Outgoing: TTS → Caller
   - Monitor: VAD for barge-in
   ↓
6. StasisEnd → Call leaves, cleanup
```

---

## File Structure

```
telephony/
├── asterisk_agi.py          # OLD: Sequential AGI (keep for reference)
├── asterisk_ari.py          # NEW: Full-duplex ARI handler
├── ari_audio_bridge.py      # NEW: Bidirectional audio streaming
├── ari_session_manager.py   # NEW: ARI-specific session management
├── call_session.py          # Updated for ARI compatibility
├── barge_in.py              # Integrated with streaming
├── audio_bridge.py          # Audio format conversion (reused)
└── config/
    ├── extensions.conf      # Updated for ARI
    ├── ari.conf             # NEW: ARI configuration
    └── http.conf            # Updated for WebSocket
```

---

## Performance Targets

| Metric | Current (AGI) | Target (ARI) | Improvement |
|--------|---------------|--------------|-------------|
| Total Latency | 4-10s | 1-3s | 70-80% |
| Barge-in Delay | N/A (sequential) | <300ms | Real-time |
| Concurrent Calls | 20 | 50+ | 2.5x |
| Audio Quality | 8kHz µ-law | 16kHz PCM | Higher |

---

## Risk Mitigation

### Risks
1. **Complexity**: ARI more complex than AGI
2. **Debugging**: Harder to troubleshoot streaming issues
3. **Resource Usage**: More CPU/memory for concurrent streams
4. **Network Latency**: WebSocket adds network hop

### Mitigations
1. Incremental development with thorough testing
2. Comprehensive logging and metrics
3. Resource monitoring and limits
4. Local WebSocket (low latency)
5. Keep AGI implementation as fallback

---

## Testing Strategy

### Unit Tests
- Audio bridge conversion accuracy
- VAD detection sensitivity
- Session state management
- WebSocket reconnection

### Integration Tests
- ARI event handling
- Full conversation flow
- Barge-in detection
- Error recovery

### Load Tests
- 10 concurrent calls
- 25 concurrent calls
- 50 concurrent calls
- Measure latency, CPU, memory

### User Acceptance Tests
- Natural conversation flow
- Barge-in responsiveness
- Audio quality
- Error handling (hang-ups, network issues)

---

## Rollout Plan

### Week 1: Setup & Core Handler
- Day 1-2: Install dependencies, configure Asterisk
- Day 3-5: Build core ARI handler, WebSocket connection

### Week 2: Audio Streaming
- Day 1-3: Implement bidirectional audio
- Day 4-5: Audio format conversion, testing

### Week 3: Barge-in & Integration
- Day 1-2: Integrate barge-in with streaming
- Day 3-5: Session management, testing

### Week 4: Testing & Optimization
- Day 1-2: Load testing
- Day 3-4: Performance tuning
- Day 5: Documentation, deployment

---

## Success Criteria

✅ **Complete** when:
1. Calls answered via ARI (not AGI)
2. Full-duplex audio streaming working
3. Real-time barge-in detection (<300ms)
4. Latency reduced to <3s
5. 50+ concurrent calls supported
6. All tests passing
7. Documentation updated

---

## Next Steps

1. ✅ Create this implementation plan
2. ⏭️ Install ARI dependencies
3. ⏭️ Configure Asterisk for ARI
4. ⏭️ Build core ARI handler
5. ⏭️ Implement audio streaming
6. ⏭️ Integrate barge-in
7. ⏭️ Test and optimize

---

**Ready to achieve 100% completion! 🚀**
