import pytest
import socket
import types

from montage_ai.ops import registry

class DummyResponse:
    def __init__(self, status_code):
        self.status_code = status_code

class DummyConn:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_check_port_success(monkeypatch):
    called = {}
    def fake_create_conn(addr, timeout):
        called['addr'] = addr
        return DummyConn()
    monkeypatch.setattr(socket, 'create_connection', fake_create_conn)
    assert registry.check_port('example.com', 5000, timeout=0.1) is True
    assert called['addr'] == ('example.com', 5000)


def test_check_port_failure(monkeypatch):
    def fake_create_conn(addr, timeout):
        raise OSError("no route")
    monkeypatch.setattr(socket, 'create_connection', fake_create_conn)
    assert registry.check_port('example.com', 5000, timeout=0.1) is False


def test_check_registry_http_ok(monkeypatch):
    def fake_get(url, timeout):
        return DummyResponse(200)
    monkeypatch.setattr('montage_ai.ops.registry.requests.get', fake_get)
    ok, note = registry.check_registry_http('example.com', 5000, use_https=False, timeout=0.1)
    assert ok is True
    assert 'OK' in note


def test_check_registry_http_auth(monkeypatch):
    def fake_get(url, timeout):
        return DummyResponse(401)
    monkeypatch.setattr('montage_ai.ops.registry.requests.get', fake_get)
    ok, note = registry.check_registry_http('example.com', 5000, use_https=False, timeout=0.1)
    assert ok is True


def test_check_registry_http_failure(monkeypatch):
    def fake_get(url, timeout):
        raise Exception('timeout')
    monkeypatch.setattr('montage_ai.ops.registry.requests.get', fake_get)
    ok, note = registry.check_registry_http('example.com', 5000, use_https=False, timeout=0.1)
    assert ok is False
    assert 'timeout' in note


def test_check_registry_ports(monkeypatch):
    # Make check_port return True for 30500, False for others
    def fake_create_conn(addr, timeout):
        if addr[1] == 30500:
            return DummyConn()
        raise OSError()
    monkeypatch.setattr(socket, 'create_connection', fake_create_conn)

    def fake_get(url, timeout):
        if ":30500" in url:
            return DummyResponse(200)
        raise Exception('conn')
    monkeypatch.setattr('montage_ai.ops.registry.requests.get', fake_get)

    res = registry.check_registry('example.com', ports=(30500,5000))
    assert res[30500]['tcp'] is True
    assert res[30500]['http'] is True
    assert res[5000]['tcp'] is False
    assert res[5000]['http'] is False
