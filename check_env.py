import flask
import numpy as np
import sys
print(f"Python: {sys.version}")
print(f"Flask: {flask.__version__}")
try:
    from flask import testing
    print(f"flask.testing imported: {testing}")
except ImportError as e:
    print(f"flask.testing failed: {e}")

print(f"Numpy: {np.__version__}")
try:
    print(f"Numpy bool_: {np.bool_}")
except AttributeError:
    print("Numpy bool_ not found")
