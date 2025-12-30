import json
import numpy as np

from montage_ai.timeline_exporter import TimelineExporter


def main() -> None:
    exporter = TimelineExporter(output_dir="/tmp")

    sample = {
        "score": np.float32(0.9),
        "count": np.int64(7),
        "vector": np.array([0.1, 0.2], dtype=np.float32),
        "nested": {"value": np.int64(3)},
        "items": [np.float32(1.25), np.int64(2)],
    }

    sanitized = exporter._sanitize_metadata(sample)
    json.dumps(sanitized)

    print("score", type(sanitized["score"]))
    print("count", type(sanitized["count"]))
    print("vector", type(sanitized["vector"]))
    print("nested", type(sanitized["nested"]["value"]))
    print("items", type(sanitized["items"][0]), type(sanitized["items"][1]))


if __name__ == "__main__":
    main()
