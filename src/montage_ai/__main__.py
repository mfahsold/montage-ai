"""
Montage AI - Entry point for python -m montage_ai

This avoids the RuntimeWarning about 'montage_ai.editor' being in sys.modules
before execution by NOT importing editor through __init__.py first.
"""

if __name__ == "__main__":
    # Direct import to avoid circular import through __init__.py
    # Install a sane SIGPIPE handler and suppress logging internal errors so
    # we don't spew '--- Logging error ---' when stdout/stderr are closed by
    # a pipe consumer (e.g., `tee`, Ctrl+C, or CI pipelines).
    import sys
    import signal
    import logging

    # Default SIGPIPE behavior (terminate) so writes to closed pipes don't
    # raise expensive exceptions later on.
    try:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception:
        # Not all platforms support SIGPIPE (e.g., Windows); ignore failures
        pass

    # Avoid printing internal logging exceptions to stderr in production runs
    logging.raiseExceptions = False

    try:
        from montage_ai.editor import main
        main()
    except BrokenPipeError:
        # Exit quietly when the output pipe is closed
        try:
            sys.exit(0)
        except Exception:
            pass
