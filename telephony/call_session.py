"""
Multi-call session management for concurrent telephony handling.
Each call maintains isolated state, conversation context, and resources.
"""
import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass, field, asdict
import redis
from app.utils.logger import log


class CallState(Enum):
    """Call lifecycle states"""
    RINGING = "ringing"
    ANSWERED = "answered"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    TRANSFERRING = "transferring"
    ENDING = "ending"
    ENDED = "ended"
    FAILED = "failed"


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    timestamp: str
    user_text: str
    agent_text: str
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    kb_chunks_used: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    interrupted: bool = False


@dataclass
class CallSession:
    """
    Complete state for an active call session.
    Maintains conversation history, performance metrics, and call metadata.
    """
    # Identity
    session_id: str
    call_id: str  # Asterisk/FreeSWITCH channel ID

    # Call metadata
    caller_number: str
    called_number: str
    company_id: str

    # State
    state: CallState
    started_at: datetime
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Conversation
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    current_turn: int = 0

    # Performance tracking
    total_interrupts: int = 0
    total_errors: int = 0
    avg_stt_latency: float = 0.0
    avg_llm_latency: float = 0.0
    avg_tts_latency: float = 0.0

    # Resource management
    audio_buffer_size: int = 0
    kb_cache: Dict = field(default_factory=dict)

    # Error tracking
    last_error: Optional[str] = None
    error_log: List[str] = field(default_factory=list)

    def duration_seconds(self) -> float:
        """Calculate call duration"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return (datetime.now() - self.started_at).total_seconds()

    def add_turn(
        self,
        user_text: str,
        agent_text: str,
        stt_latency: float,
        llm_latency: float,
        tts_latency: float,
        kb_chunks: List[str] = None,
        confidence: float = 0.0,
        interrupted: bool = False
    ):
        """Add conversation turn and update metrics"""
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_text=user_text,
            agent_text=agent_text,
            stt_latency_ms=stt_latency * 1000,
            llm_latency_ms=llm_latency * 1000,
            tts_latency_ms=tts_latency * 1000,
            kb_chunks_used=kb_chunks or [],
            confidence_score=confidence,
            interrupted=interrupted
        )

        self.conversation_history.append(turn)
        self.current_turn += 1

        if interrupted:
            self.total_interrupts += 1

        # Update averages
        self._update_latency_averages()

    def _update_latency_averages(self):
        """Recalculate average latencies"""
        if not self.conversation_history:
            return

        total_turns = len(self.conversation_history)
        self.avg_stt_latency = sum(t.stt_latency_ms for t in self.conversation_history) / total_turns
        self.avg_llm_latency = sum(t.llm_latency_ms for t in self.conversation_history) / total_turns
        self.avg_tts_latency = sum(t.tts_latency_ms for t in self.conversation_history) / total_turns

    def add_error(self, error: str):
        """Log error for this session"""
        self.total_errors += 1
        self.last_error = error
        self.error_log.append(f"{datetime.now().isoformat()}: {error}")

    def to_dict(self) -> dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['started_at'] = self.started_at.isoformat()
        data['answered_at'] = self.answered_at.isoformat() if self.answered_at else None
        data['ended_at'] = self.ended_at.isoformat() if self.ended_at else None
        data['state'] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create session from dictionary"""
        # Convert ISO strings back to datetime
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('answered_at'):
            data['answered_at'] = datetime.fromisoformat(data['answered_at'])
        if data.get('ended_at'):
            data['ended_at'] = datetime.fromisoformat(data['ended_at'])
        data['state'] = CallState(data['state'])

        # Reconstruct ConversationTurn objects
        turns = []
        for turn_data in data.get('conversation_history', []):
            turns.append(ConversationTurn(**turn_data))
        data['conversation_history'] = turns

        return cls(**data)


class CallSessionManager:
    """
    Manages multiple concurrent call sessions with Redis persistence.

    Features:
    - Thread-safe session management
    - Automatic cleanup of idle sessions
    - Performance monitoring per session
    - Conversation history preservation
    """

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        session_timeout: int = 3600,  # 1 hour
        max_concurrent_calls: int = 20
    ):
        """
        Initialize session manager.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            session_timeout: Session TTL in seconds
            max_concurrent_calls: Maximum concurrent calls allowed
        """
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=False  # We'll handle JSON encoding
        )
        self.session_timeout = session_timeout
        self.max_concurrent_calls = max_concurrent_calls

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, CallSession] = {}

        # Locks for thread safety
        self.locks: Dict[str, asyncio.Lock] = {}

        log.info(f"CallSessionManager initialized (max_concurrent={max_concurrent_calls})")

    async def create_session(
        self,
        call_id: str,
        caller_number: str,
        called_number: str,
        company_id: str
    ) -> CallSession:
        """
        Create new call session.

        Args:
            call_id: Unique call identifier from telephony system
            caller_number: Caller phone number
            called_number: Called phone number (DID)
            company_id: Company context for knowledge base

        Returns:
            CallSession object

        Raises:
            Exception: If max concurrent calls reached
        """
        # Check capacity
        active_count = await self.get_active_count()
        if active_count >= self.max_concurrent_calls:
            raise Exception(f"Maximum concurrent calls reached ({self.max_concurrent_calls})")

        # Generate session ID
        session_id = f"session-{uuid.uuid4().hex[:16]}"

        # Create session
        session = CallSession(
            session_id=session_id,
            call_id=call_id,
            caller_number=caller_number,
            called_number=called_number,
            company_id=company_id,
            state=CallState.ANSWERED,
            started_at=datetime.now()
        )

        # Store in memory and Redis
        self.active_sessions[session_id] = session
        self.locks[session_id] = asyncio.Lock()

        await self._save_session(session)

        log.info(f"Session created: {session_id} for {caller_number} → {company_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[CallSession]:
        """
        Retrieve session by ID.

        Args:
            session_id: Session identifier

        Returns:
            CallSession if found, None otherwise
        """
        # Check memory cache first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Try Redis
        session_data = self.redis.get(f"session:{session_id}")
        if session_data:
            session = CallSession.from_dict(json.loads(session_data))
            self.active_sessions[session_id] = session
            return session

        return None

    async def update_session(self, session: CallSession):
        """
        Update session state in memory and Redis.

        Args:
            session: Updated session object
        """
        async with self.locks.get(session.session_id, asyncio.Lock()):
            self.active_sessions[session.session_id] = session
            await self._save_session(session)

    async def end_session(self, session_id: str, reason: str = "completed"):
        """
        End call session and clean up.

        Args:
            session_id: Session to end
            reason: Ending reason (completed, hangup, timeout, error)
        """
        session = await self.get_session(session_id)
        if not session:
            log.warning(f"Cannot end session {session_id}: not found")
            return

        # Update state
        session.state = CallState.ENDED
        session.ended_at = datetime.now()

        # Save final state
        await self._save_final_state(session, reason)

        # Remove from active sessions
        self.active_sessions.pop(session_id, None)
        self.locks.pop(session_id, None)

        # Remove from Redis (already saved to long-term storage)
        self.redis.delete(f"session:{session_id}")

        log.info(
            f"Session ended: {session_id} "
            f"(duration={session.duration_seconds():.1f}s, turns={session.current_turn}, reason={reason})"
        )

    async def get_active_count(self) -> int:
        """Get count of active sessions"""
        # Count from Redis (source of truth)
        pattern = "session:*"
        keys = self.redis.keys(pattern)
        return len(keys)

    async def cleanup_idle_sessions(self, max_idle_seconds: int = 300):
        """
        Clean up sessions idle for too long.

        Args:
            max_idle_seconds: Maximum idle time before cleanup
        """
        now = datetime.now()
        to_remove = []

        for session_id, session in self.active_sessions.items():
            idle_time = (now - session.started_at).total_seconds()

            # Check if idle too long
            if session.state == CallState.IN_PROGRESS and idle_time > max_idle_seconds:
                log.warning(f"Session {session_id} idle for {idle_time:.1f}s, cleaning up")
                to_remove.append(session_id)

        # Remove idle sessions
        for session_id in to_remove:
            await self.end_session(session_id, reason="timeout")

    async def _save_session(self, session: CallSession):
        """Save session to Redis with TTL"""
        session_data = json.dumps(session.to_dict(), default=str)
        self.redis.setex(
            f"session:{session.session_id}",
            self.session_timeout,
            session_data
        )

    async def _save_final_state(self, session: CallSession, reason: str):
        """
        Save final session state to long-term storage.

        This would typically write to:
        - Database (PostgreSQL, MongoDB)
        - Data warehouse (BigQuery, Redshift)
        - Analytics platform (Elasticsearch)

        For now, we'll write to a JSON log file.
        """
        log_entry = {
            **session.to_dict(),
            'end_reason': reason,
            'final_duration_seconds': session.duration_seconds()
        }

        # Write to daily log file
        log_file = f"logs/calls_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
        except Exception as e:
            log.error(f"Failed to write call log: {e}")

    async def get_session_stats(self) -> dict:
        """
        Get aggregate statistics across all sessions.

        Returns:
            Dict with stats like avg latency, total calls, etc.
        """
        active_count = await self.get_active_count()

        total_turns = sum(s.current_turn for s in self.active_sessions.values())
        total_interrupts = sum(s.total_interrupts for s in self.active_sessions.values())

        avg_stt = sum(s.avg_stt_latency for s in self.active_sessions.values()) / max(active_count, 1)
        avg_llm = sum(s.avg_llm_latency for s in self.active_sessions.values()) / max(active_count, 1)
        avg_tts = sum(s.avg_tts_latency for s in self.active_sessions.values()) / max(active_count, 1)

        return {
            'active_calls': active_count,
            'total_turns': total_turns,
            'total_interrupts': total_interrupts,
            'avg_stt_latency_ms': avg_stt,
            'avg_llm_latency_ms': avg_llm,
            'avg_tts_latency_ms': avg_tts,
            'capacity_used_percent': (active_count / self.max_concurrent_calls) * 100
        }


# Singleton instance
_session_manager: Optional[CallSessionManager] = None


def get_session_manager() -> CallSessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = CallSessionManager()
    return _session_manager
