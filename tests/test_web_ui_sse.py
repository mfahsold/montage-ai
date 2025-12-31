"""
Tests for Server-Sent Events (SSE) Logic
"""

import pytest
import queue
from src.montage_ai.web_ui.sse import MessageAnnouncer, format_sse
from src.montage_ai.web_ui.app import app

def test_format_sse():
    """Test SSE message formatting."""
    assert format_sse("hello") == "data: hello\n\n"
    assert format_sse("hello", event="greeting") == "event: greeting\ndata: hello\n\n"
    assert format_sse('{"status": "ok"}') == 'data: {"status": "ok"}\n\n'

def test_announcer_broadcast():
    """Test that announcer broadcasts to listeners."""
    announcer = MessageAnnouncer()
    q1 = announcer.listen()
    q2 = announcer.listen()
    
    announcer.announce("test message")
    
    assert q1.get() == "test message"
    assert q2.get() == "test message"

def test_announcer_dead_listener_cleanup():
    """Test that full queues (dead listeners) are removed."""
    announcer = MessageAnnouncer()
    q = announcer.listen()
    
    # Fill the queue
    for i in range(5):
        q.put(f"msg {i}")
        
    # This should trigger the queue.Full exception and removal
    announcer.announce("overflow")
    
    assert len(announcer.listeners) == 0

def test_stream_route_exists():
    """Test that the stream endpoint is registered."""
    from src.montage_ai.web_ui.app import app
    # Check if the route is in the url_map
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    assert '/api/stream' in rules
