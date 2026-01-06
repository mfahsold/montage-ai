"""
Montage AI - Entry point for python -m montage_ai

This avoids the RuntimeWarning about 'montage_ai.editor' being in sys.modules
before execution by NOT importing editor through __init__.py first.
"""

if __name__ == "__main__":
    # Direct import to avoid circular import through __init__.py
    from montage_ai.editor import main
    main()
