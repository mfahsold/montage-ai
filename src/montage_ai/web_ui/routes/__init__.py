"""Web UI routes package.

This module only defines shared blueprint objects for optional route modules.
Route modules are imported explicitly by the app to avoid circular imports.
"""

from flask import Blueprint

# Create blueprints
main_bp = Blueprint("main", __name__)
jobs_bp = Blueprint("jobs", __name__, url_prefix="/api")
files_bp = Blueprint("files", __name__, url_prefix="/api")
sessions_bp = Blueprint("sessions", __name__, url_prefix="/api/session")
shorts_bp = Blueprint("shorts", __name__, url_prefix="/api/shorts")
transcripts_bp = Blueprint("transcripts", __name__, url_prefix="/api/transcript")
cgpu_bp = Blueprint("cgpu", __name__, url_prefix="/api/cgpu")
status_bp = Blueprint("status", __name__, url_prefix="/api")

__all__ = [
    "main_bp",
    "jobs_bp",
    "files_bp",
    "sessions_bp",
    "shorts_bp",
    "transcripts_bp",
    "cgpu_bp",
    "status_bp",
]
