
import sys
import flask
import pytest
import numpy as np

def test_debug_env():
    print("\n--- DEBUG INFO ---")
    print(f"sys.path: {sys.path}")
    print(f"flask: {flask}")
    print(f"flask file: {getattr(flask, '__file__', 'no file')}")
    print(f"flask path: {getattr(flask, '__path__', 'no path')}")
    print(f"flask is package: {getattr(flask, '__package__', 'no package')}")
    
    try:
        from flask import testing
        print(f"flask.testing: {testing}")
    except ImportError as e:
        print(f"Failed to import flask.testing: {e}")

    try:
        print(f"np.bool_: {np.bool_}")
        print(f"type(np.bool_): {type(np.bool_)}")
        print(f"isinstance(np.bool_, type): {isinstance(np.bool_, type)}")
    except Exception as e:
        print(f"Numpy check failed: {e}")
        
    print("------------------\n")
