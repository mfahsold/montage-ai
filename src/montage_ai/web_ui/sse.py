"""
Server-Sent Events (SSE) Utility

Provides a thread-safe message announcer for broadcasting events to connected clients.
Used by the Web UI for real-time job status updates without polling overhead.
"""

import queue
from typing import List, Optional

class MessageAnnouncer:
    """
    Thread-safe message broadcaster for Server-Sent Events.
    Maintains a list of listener queues and broadcasts messages to all of them.
    """
    def __init__(self):
        self.listeners: List[queue.Queue] = []

    def listen(self) -> queue.Queue:
        """
        Register a new listener and return a queue for receiving messages.
        The queue has a small maxsize to prevent memory leaks if a client disconnects
        but the server keeps pushing.
        """
        q = queue.Queue(maxsize=5)
        self.listeners.append(q)
        return q

    def announce(self, msg: str) -> None:
        """
        Broadcast a message to all active listeners.
        Iterates backwards to safely remove dead listeners (full queues).
        """
        # We iterate backwards to allow deleting dead listeners
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                # Listener is not consuming messages (disconnected or slow)
                del self.listeners[i]

def format_sse(data: str, event: Optional[str] = None) -> str:
    """
    Format a data string as a Server-Sent Event message.
    
    Args:
        data: The message payload (usually JSON).
        event: Optional event type name.
        
    Returns:
        Formatted SSE string (e.g., "event: update\ndata: {...}\n\n").
    """
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg
