"""Registry helper utilities for checking registry reachability and endpoints."""
from __future__ import annotations

import requests
import socket
from typing import Iterable, Tuple


def check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_registry_http(host: str, port: int, use_https: bool = False, timeout: float = 3.0) -> Tuple[bool, str]:
    scheme = "https" if use_https else "http"
    url = f"{scheme}://{host}:{port}/v2/"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code in (200, 401):
            return True, f"OK ({r.status_code})"
        else:
            return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)


def check_registry(host: str, ports: Iterable[int] = (30500, 5000)) -> dict:
    """Return dict with port -> (tcp_ok, http_ok, https_ok, notes)"""
    result = {}
    for p in ports:
        tcp_ok = check_port(host, p)
        http_ok, http_note = check_registry_http(host, p, use_https=False)
        https_ok, https_note = check_registry_http(host, p, use_https=True)
        result[p] = {
            "tcp": tcp_ok,
            "http": http_ok,
            "https": https_ok,
            "http_note": http_note,
            "https_note": https_note,
        }
    return result
