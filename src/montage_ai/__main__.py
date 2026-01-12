"""
Montage AI - Entry point for python -m montage_ai

This avoids the RuntimeWarning about 'montage_ai.editor' being in sys.modules
before execution by NOT importing editor through __init__.py first.
"""

if __name__ == "__main__":
    # Direct import to avoid circular import through __init__.py
    # Wrap in BrokenPipeError handler to avoid noisy tracebacks when stdout
    # is closed by a pipe consumer (e.g., user presses Ctrl+C while tee is attached).
    import sys
    try:
        from montage_ai.editor import main
        main()
    except BrokenPipeError:
        # Exit quietly when the output pipe is closed
        try:
            sys.exit(0)
        except Exception:
            pass
