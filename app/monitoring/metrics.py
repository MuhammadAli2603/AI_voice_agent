"""
Prometheus metrics for AI Voice Agent telephony system.
Tracks call volume, latency, quality, and errors.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Optional


#==============================================
# CALL METRICS
#==============================================

# Total calls
total_calls = Counter(
    'voice_agent_total_calls',
    'Total number of calls handled',
    ['company_id', 'status']  # status: completed, error, hangup, timeout
)

# Active calls
active_calls = Gauge(
    'voice_agent_active_calls',
    'Number of currently active calls'
)

# Call duration
call_duration = Histogram(
    'voice_agent_call_duration_seconds',
    'Call duration in seconds',
    ['company_id'],
    buckets=[10, 30, 60, 120, 300, 600, 1200, 1800]  # 10s to 30min
)

# Conversation turns per call
conversation_turns = Histogram(
    'voice_agent_conversation_turns',
    'Number of conversation turns per call',
    ['company_id'],
    buckets=[1, 2, 3, 5, 10, 15, 20, 30, 50]
)

#==============================================
# LATENCY METRICS
#==============================================

# STT latency
stt_latency = Histogram(
    'voice_agent_stt_latency_seconds',
    'Speech-to-text processing latency',
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
)

# LLM latency
llm_latency = Histogram(
    'voice_agent_llm_latency_seconds',
    'LLM response generation latency',
    ['company_id'],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0]
)

# TTS latency
tts_latency = Histogram(
    'voice_agent_tts_latency_seconds',
    'Text-to-speech synthesis latency',
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
)

# Total turn latency (STT + LLM + TTS)
total_turn_latency = Histogram(
    'voice_agent_total_turn_latency_seconds',
    'Total latency for one conversation turn',
    ['company_id'],
    buckets=[1, 2, 3, 5, 7, 10, 15, 20]
)

#==============================================
# QUALITY METRICS
#==============================================

# Knowledge base confidence
kb_confidence = Histogram(
    'voice_agent_kb_confidence',
    'Knowledge base retrieval confidence scores',
    ['company_id'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Interruptions (barge-in)
interruptions = Counter(
    'voice_agent_interruptions_total',
    'Total number of caller interruptions',
    ['company_id']
)

# Audio quality issues
audio_quality_issues = Counter(
    'voice_agent_audio_quality_issues',
    'Audio quality problems detected',
    ['issue_type']  # dropout, distortion, noise, etc.
)

#==============================================
# ERROR METRICS
#==============================================

# Errors by type
errors = Counter(
    'voice_agent_errors_total',
    'Total errors encountered',
    ['error_type', 'component']  # component: stt, llm, tts, telephony, kb
)

# API failures
api_failures = Counter(
    'voice_agent_api_failures',
    'API call failures',
    ['api_name', 'status_code']  # api_name: huggingface, kb_service
)

# Session management errors
session_errors = Counter(
    'voice_agent_session_errors',
    'Session management errors',
    ['error_type']
)

#==============================================
# CAPACITY METRICS
#==============================================

# Capacity usage
capacity_usage = Gauge(
    'voice_agent_capacity_usage_percent',
    'Percentage of max concurrent calls in use'
)

# Queue depth (if queuing implemented)
call_queue_depth = Gauge(
    'voice_agent_call_queue_depth',
    'Number of calls waiting in queue'
)

#==============================================
# SYSTEM INFO
#==============================================

system_info = Info(
    'voice_agent_system',
    'System information'
)

system_info.info({
    'version': '1.0.0',
    'stt_model': 'openai/whisper-large-v3',
    'llm_model': 'microsoft/DialoGPT-medium',
    'tts_model': 'facebook/mms-tts-eng',
    'telephony': 'asterisk-agi'
})


#==============================================
# HELPER CLASSES
#==============================================

class CallMetrics:
    """
    Convenience class for tracking metrics for a single call.
    Automatically records metrics to Prometheus.
    """

    def __init__(self, session_id: str, company_id: str):
        """
        Initialize call metrics tracker.

        Args:
            session_id: Unique session identifier
            company_id: Company context
        """
        self.session_id = session_id
        self.company_id = company_id
        self.start_time = time.time()

        # Increment active calls
        active_calls.inc()

        # Increment total calls (status will be updated on end)
        # (Tracked separately on call end)

    def record_turn(
        self,
        stt_time: float,
        llm_time: float,
        tts_time: float,
        kb_confidence: float = 0.0,
        interrupted: bool = False
    ):
        """
        Record metrics for a conversation turn.

        Args:
            stt_time: STT processing time in seconds
            llm_time: LLM processing time in seconds
            tts_time: TTS processing time in seconds
            kb_confidence: KB confidence score (0-1)
            interrupted: Whether turn was interrupted
        """
        # Record component latencies
        stt_latency.observe(stt_time)
        llm_latency.labels(company_id=self.company_id).observe(llm_time)
        tts_latency.observe(tts_time)

        # Record total turn latency
        total_time = stt_time + llm_time + tts_time
        total_turn_latency.labels(company_id=self.company_id).observe(total_time)

        # Record KB confidence
        if kb_confidence > 0:
            kb_confidence.labels(company_id=self.company_id).observe(kb_confidence)

        # Record interruption
        if interrupted:
            interruptions.labels(company_id=self.company_id).inc()

    def record_error(self, error_type: str, component: str):
        """
        Record an error.

        Args:
            error_type: Type of error (timeout, api_failure, etc.)
            component: Which component failed
        """
        errors.labels(
            error_type=error_type,
            component=component
        ).inc()

    def record_api_failure(self, api_name: str, status_code: int):
        """
        Record API failure.

        Args:
            api_name: API that failed (huggingface, kb_service)
            status_code: HTTP status code
        """
        api_failures.labels(
            api_name=api_name,
            status_code=str(status_code)
        ).inc()

    def end_call(self, status: str, num_turns: int):
        """
        End call and record final metrics.

        Args:
            status: Call end status (completed, error, hangup, timeout)
            num_turns: Number of conversation turns
        """
        # Decrement active calls
        active_calls.dec()

        # Increment total calls with status
        total_calls.labels(
            company_id=self.company_id,
            status=status
        ).inc()

        # Record call duration
        duration = time.time() - self.start_time
        call_duration.labels(company_id=self.company_id).observe(duration)

        # Record conversation turns
        conversation_turns.labels(company_id=self.company_id).observe(num_turns)


class MetricsServer:
    """
    Prometheus metrics HTTP server.
    Exposes metrics at /metrics endpoint.
    """

    def __init__(self, port: int = 9090):
        """
        Initialize metrics server.

        Args:
            port: Port to expose metrics on
        """
        from prometheus_client import start_http_server

        self.port = port
        start_http_server(port)

        print(f"✓ Metrics server started on http://localhost:{port}/metrics")

    def update_capacity(self, active_count: int, max_count: int):
        """
        Update capacity usage metric.

        Args:
            active_count: Current active calls
            max_count: Maximum allowed calls
        """
        usage_percent = (active_count / max_count) * 100 if max_count > 0 else 0
        capacity_usage.set(usage_percent)


# Singleton metrics server
_metrics_server: Optional[MetricsServer] = None


def get_metrics_server(port: int = 9090) -> MetricsServer:
    """Get or create metrics server"""
    global _metrics_server
    if _metrics_server is None:
        _metrics_server = MetricsServer(port=port)
    return _metrics_server


# Example usage
if __name__ == "__main__":
    import time

    # Start metrics server
    server = get_metrics_server(port=9090)

    # Simulate some calls
    for i in range(5):
        metrics = CallMetrics(f"call-{i}", "techstore")

        # Simulate conversation turns
        for turn in range(3):
            metrics.record_turn(
                stt_time=0.5,
                llm_time=2.0,
                tts_time=0.8,
                kb_confidence=0.85
            )
            time.sleep(0.1)

        metrics.end_call(status="completed", num_turns=3)

    print("Metrics available at http://localhost:9090/metrics")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped")
