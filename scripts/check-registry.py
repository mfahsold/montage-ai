#!/usr/bin/env python3
"""CLI wrapper that checks registry health using the Python registry helper."""
import sys
import os
from montage_ai.ops.registry import check_registry


def main():
    # Prefer explicit CLI arg, then environment, then localhost
    host = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get('REGISTRY_HOST') if os.environ.get('REGISTRY_HOST') else 'localhost')
    res = check_registry(host)
    for port, info in res.items():
        print(f"Port {port}: TCP={info['tcp']}, HTTP={info['http']} ({info['http_note']}), HTTPS={info['https']} ({info['https_note']})")


if __name__ == "__main__":
    main()
