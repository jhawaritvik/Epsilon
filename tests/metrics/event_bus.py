"""
Event Bus for Epsilon Pipeline.

Decouples agents from metrics collection. Agents emit events; MetricsCollector
subscribes.

This allows agents to remain pure and deterministic, which is critical for
reproducible research.
"""

from typing import Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class Event:
    """Represents a single event in the system."""
    
    name: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""


class EventBus:
    """
    A simple, thread-safe publish/subscribe event bus.
    
    Usage:
        bus = get_event_bus()
        bus.subscribe("ASSUMPTION_FAILED", handler_fn)
        bus.emit("ASSUMPTION_FAILED", {"type": "normality", ...})
    """
    
    _instance: 'EventBus' = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
                    cls._instance._event_log: List[Event] = []
        return cls._instance
    
    def subscribe(self, event_name: str, handler: Callable[[Event], None]):
        """
        Subscribe to an event.
        
        Args:
            event_name: Name of the event to listen for.
            handler: Callback function that receives the Event.
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(handler)
    
    def unsubscribe(self, event_name: str, handler: Callable[[Event], None]):
        """Unsubscribe a handler from an event."""
        if event_name in self._subscribers:
            self._subscribers[event_name] = [
                h for h in self._subscribers[event_name] if h != handler
            ]
    
    def emit(self, event_name: str, data: Dict[str, Any], source: str = ""):
        """
        Emit an event to all subscribers.
        
        Args:
            event_name: Name of the event.
            data: Event data payload.
            source: Optional source identifier (e.g., agent name).
        """
        event = Event(name=event_name, data=data, source=source)
        self._event_log.append(event)
        
        handlers = self._subscribers.get(event_name, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log but don't crash
                print(f"[EventBus] Error in handler for {event_name}: {e}")
    
    def get_event_log(self) -> List[Event]:
        """Returns a copy of the event log."""
        return list(self._event_log)
    
    def clear_log(self):
        """Clears the event log. Useful for resetting between tests."""
        self._event_log.clear()
    
    def reset(self):
        """Resets all subscribers and the event log."""
        self._subscribers.clear()
        self._event_log.clear()


# Global accessor
_event_bus: EventBus = None


def get_event_bus() -> EventBus:
    """Returns the global EventBus singleton."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def emit_event(event_name: str, data: Dict[str, Any], source: str = ""):
    """
    Convenience function to emit an event.
    
    Agents should call this instead of directly using the bus.
    
    Example:
        emit_event("ASSUMPTION_FAILED", {"type": "normality", "agent": "evaluation"})
    """
    get_event_bus().emit(event_name, data, source)


# =============================================================================
# Event Names (Constants)
# =============================================================================

# Iteration Events
ITERATION_STARTED = "ITERATION_STARTED"
ITERATION_COMPLETED = "ITERATION_COMPLETED"
ITERATION_FAILED = "ITERATION_FAILED"

# Scientific Validity Events
ASSUMPTION_FAILED = "ASSUMPTION_FAILED"
ASSUMPTION_PASSED = "ASSUMPTION_PASSED"
CRYSTALLIZATION_ATTEMPTED = "CRYSTALLIZATION_ATTEMPTED"

# Memory Events
MEMORY_QUERIED = "MEMORY_QUERIED"
MEMORY_HIT = "MEMORY_HIT"
EVIDENCE_ADDED = "EVIDENCE_ADDED"
KNOWLEDGE_REUSED = "KNOWLEDGE_REUSED"
FAILURE_RECALLED = "FAILURE_RECALLED"

# Contract Integrity Events
AUTHORITY_VIOLATION = "AUTHORITY_VIOLATION"
MODALITY_VIOLATION = "MODALITY_VIOLATION"
TOOL_MISUSE = "TOOL_MISUSE"
AMBIGUITY_HALT = "AMBIGUITY_HALT"

# Cost Events
TOKENS_USED = "TOKENS_USED"
TIME_RECORDED = "TIME_RECORDED"

# Run Events
RUN_STARTED = "RUN_STARTED"
RUN_COMPLETED = "RUN_COMPLETED"
RUN_FAILED = "RUN_FAILED"
