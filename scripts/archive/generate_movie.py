#!/usr/bin/env python3
"""Deprecated CLI: Open-Sora movie generation was removed from Montage AI."""

import sys
from textwrap import dedent


DEPRECATION_MESSAGE = dedent(
    """
    ❌ Movie generation is no longer available.

    The Open-Sora integration (and other raw footage generators) was removed as
    part of the pivot to pure post-production. This script now only exists to
    provide a clear message instead of an ImportError.

    What to do instead:
    - Use existing footage with the main CLI: ./montage-ai.sh run
    - Or open the Web UI: make web
    """
)


def main() -> int:
    print(DEPRECATION_MESSAGE.strip())
    return 1


if __name__ == "__main__":
    sys.exit(main())
