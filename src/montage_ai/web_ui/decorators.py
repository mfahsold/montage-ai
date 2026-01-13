"""
API Decorators for DRY error handling and validation.

Usage:
    from .decorators import api_endpoint, require_file, require_json

    @app.route('/api/example', methods=['POST'])
    @api_endpoint
    @require_json('filename', 'style')
    def api_example():
        # No need for try/except or validation boilerplate
        data = request.json
        ...
"""

import os
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union, List

from flask import request, jsonify

from ..logger import logger


def api_endpoint(f: Callable) -> Callable:
    """
    Decorator that wraps API endpoints with standardized error handling.

    Catches common exceptions and returns appropriate JSON error responses:
    - FileNotFoundError -> 404
    - ValueError, KeyError -> 400
    - Exception -> 500

    Example:
        @app.route('/api/data')
        @api_endpoint
        def get_data():
            # If this raises, decorator handles the error response
            return jsonify({"data": process_data()})
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except FileNotFoundError as e:
            logger.warning(f"[{f.__name__}] File not found: {e}")
            return jsonify({"error": str(e) or "File not found"}), 404
        except (ValueError, KeyError) as e:
            logger.warning(f"[{f.__name__}] Validation error: {e}")
            return jsonify({"error": str(e) or "Invalid request"}), 400
        except Exception as e:
            logger.error(f"[{f.__name__}] Internal error: {e}", exc_info=True)
            return jsonify({"error": str(e) or "Internal server error"}), 500
    return wrapper


def require_json(*fields: str, allow_empty: bool = False) -> Callable:
    """
    Decorator that validates required JSON fields in request body.

    Args:
        *fields: Required field names
        allow_empty: If False (default), empty strings are rejected

    Example:
        @app.route('/api/upload', methods=['POST'])
        @require_json('filename', 'type')
        def upload():
            data = request.json  # Guaranteed to have 'filename' and 'type'
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.json or {}

            missing = []
            for field in fields:
                value = data.get(field)
                if value is None:
                    missing.append(field)
                elif not allow_empty and isinstance(value, str) and not value.strip():
                    missing.append(field)

            if missing:
                return jsonify({
                    "error": f"Missing required field(s): {', '.join(missing)}"
                }), 400

            return f(*args, **kwargs)
        return wrapper
    return decorator


def require_file(
    path_param: str = 'filename',
    base_dirs: Optional[List[Path]] = None,
    extensions: Optional[List[str]] = None
) -> Callable:
    """
    Decorator that validates file existence and optionally extension.

    Args:
        path_param: Name of the JSON field or URL param containing the filename
        base_dirs: List of directories to search for the file
        extensions: Allowed file extensions (e.g., ['.mp4', '.mov'])

    Example:
        @app.route('/api/analyze', methods=['POST'])
        @require_file('filename', base_dirs=[INPUT_DIR], extensions=['.mp4'])
        def analyze():
            # File is guaranteed to exist
            ...
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get filename from JSON body or URL params
            data = request.json or {}
            filename = data.get(path_param) or request.args.get(path_param) or kwargs.get(path_param)

            if not filename:
                return jsonify({"error": f"Missing {path_param}"}), 400

            # Security: prevent path traversal
            safe_filename = os.path.basename(filename)
            if safe_filename != filename and not filename.startswith('/'):
                # Allow absolute paths but block ../
                if '..' in filename:
                    return jsonify({"error": "Invalid filename"}), 400

            # Check extension if specified
            if extensions:
                ext = Path(filename).suffix.lower()
                if ext not in extensions:
                    return jsonify({
                        "error": f"Invalid file format. Allowed: {', '.join(extensions)}"
                    }), 400

            # Find file in base directories
            file_path = None
            if base_dirs:
                for base_dir in base_dirs:
                    candidate = base_dir / safe_filename
                    if candidate.exists():
                        file_path = candidate
                        break

                if not file_path:
                    return jsonify({"error": f"File not found: {safe_filename}"}), 404
            else:
                # Absolute path mode
                file_path = Path(filename)
                if not file_path.exists():
                    return jsonify({"error": f"File not found: {filename}"}), 404

            # Inject resolved path into kwargs
            kwargs['_resolved_file_path'] = file_path

            return f(*args, **kwargs)
        return wrapper
    return decorator


def validate_range(
    field: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    default: Optional[Union[int, float]] = None
) -> Callable:
    """
    Decorator that validates numeric field is within range.

    Args:
        field: Name of the JSON field to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        default: Default value if field is missing

    Example:
        @app.route('/api/render', methods=['POST'])
        @validate_range('duration', min_val=5, max_val=300, default=60)
        def render():
            data = request.json
            duration = data.get('duration', 60)  # Guaranteed valid
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            data = request.json or {}
            value = data.get(field, default)

            if value is None:
                return jsonify({"error": f"Missing required field: {field}"}), 400

            try:
                value = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid {field}: must be a number"}), 400

            if min_val is not None and value < min_val:
                return jsonify({
                    "error": f"Invalid {field}: must be >= {min_val}"
                }), 400

            if max_val is not None and value > max_val:
                return jsonify({
                    "error": f"Invalid {field}: must be <= {max_val}"
                }), 400

            return f(*args, **kwargs)
        return wrapper
    return decorator


# Convenience aliases
json_required = require_json
file_required = require_file


__all__ = [
    'api_endpoint',
    'require_json',
    'require_file',
    'validate_range',
    'json_required',
    'file_required',
]
