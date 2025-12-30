
import numpy as np
from typing import Any

def _sanitize_metadata(data: Any) -> Any:
    """Recursively convert numpy types to python types for JSON/OTIO serialization."""
    if isinstance(data, dict):
        return {k: _sanitize_metadata(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_metadata(v) for v in data]
    elif hasattr(data, 'item'):  # numpy scalar
        try:
            return data.item()
        except ValueError:
            # In case it's an array that has .item() but isn't a scalar
            pass
    
    if hasattr(data, 'tolist'):  # numpy array
        return data.tolist()
    
    return data

# Test cases
f32 = np.float32(1.5)
i64 = np.int64(10)
arr = np.array([1, 2, 3], dtype=np.float32)
nested = {
    "score": np.float32(0.9),
    "vector": np.array([0.1, 0.2], dtype=np.float32),
    "list": [np.int64(5)]
}

print(f"f32: {type(f32)} -> {type(_sanitize_metadata(f32))}")
print(f"i64: {type(i64)} -> {type(_sanitize_metadata(i64))}")
print(f"arr: {type(arr)} -> {type(_sanitize_metadata(arr))}")
print(f"nested: {_sanitize_metadata(nested)}")

sanitized = _sanitize_metadata(nested)
print(f"nested['score']: {type(sanitized['score'])}")
print(f"nested['vector']: {type(sanitized['vector'])}")
print(f"nested['list'][0]: {type(sanitized['list'][0])}")
