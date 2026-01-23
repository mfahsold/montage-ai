import os
import time
from pathlib import Path

from montage_ai.proxy_generator import ProxyGenerator


def test_proxy_reuse_when_fresh(tmp_path):
    src = tmp_path / "in.mp4"
    proxy_dir = tmp_path / "proxies"
    src.write_bytes(b"0" * 1024)

    # create an existing proxy with newer mtime
    proxy_dir.mkdir()
    proxy = proxy_dir / "proxy_in.mp4"
    proxy.write_bytes(b"x")

    # ensure proxy mtime is newer than source
    now = time.time()
    os.utime(src, (now - 10, now - 10))
    os.utime(proxy, (now, now))

    gen = ProxyGenerator(proxy_dir)
    p = gen.get_proxy_path(src)
    assert p.exists()
    # ensure ensure_proxy returns existing proxy (does not raise)
    res = gen.ensure_proxy(str(src))
    assert res is not None
    assert res == p


def test_proxy_regenerated_when_stale(tmp_path, monkeypatch):
    src = tmp_path / "in2.mp4"
    proxy_dir = tmp_path / "proxies"
    src.write_bytes(b"0" * 1024)

    proxy_dir.mkdir()
    proxy = proxy_dir / "proxy_in2.mp4"
    proxy.write_bytes(b"x")

    # make proxy older than source
    now = time.time()
    os.utime(src, (now, now))
    os.utime(proxy, (now - 1000, now - 1000))

    gen = ProxyGenerator(proxy_dir)

    # Monkeypatch actual generation to simulate creation
    called = {}
    def fake_generate(self, s, out, fmt="h264"):
        called['gen'] = True
        out.write_bytes(b"proxy")
        return out

    monkeypatch.setattr(ProxyGenerator, "_generate", fake_generate)
    res = gen.ensure_proxy(str(src))
    assert called.get('gen') is True
    assert res.exists() and res.read_bytes() == b"proxy"


def test_analysis_proxy_ttl_and_reuse(tmp_path, monkeypatch):
    src = tmp_path / "long.mp4"
    src.write_bytes(b"0" * 1024)
    proxy_dir = tmp_path / "proxies"
    proxy_dir.mkdir()

    gen = ProxyGenerator(proxy_dir)
    out = proxy_dir / "long_analysis_proxy_240p.mp4"
    out.write_bytes(b"x")

    # Make the proxy fresh -> ensure_analysis_proxy should reuse
    now = time.time()
    os.utime(out, (now, now))
    res = gen.ensure_analysis_proxy(str(src), height=240)
    assert res is not None and res.exists()

    # Make proxy old (beyond small TTL) and ensure regeneration is attempted
    os.utime(out, (now - 100000, now - 100000))
    called = {}

    def fake_generate_analysis(self, s, o, height=360):
        from pathlib import Path
        o = Path(o)
        called['created'] = True
        o.write_bytes(b"new")
        return True

    # Patch the instance method (ensure bound method signature is correct in test env)
    monkeypatch.setattr(gen, "generate_analysis_proxy", fake_generate_analysis.__get__(gen, ProxyGenerator))
    res2 = gen.ensure_analysis_proxy(str(src), height=240, force=True)
    assert called.get('created') is True
    assert res2.exists() and res2.read_bytes() == b"new"


def test_cache_eviction_under_limit(tmp_path, monkeypatch):
    proxy_dir = tmp_path / "proxies"
    proxy_dir.mkdir()

    # Create several proxy files to exceed a tiny max_bytes threshold
    total = 0
    for i in range(5):
        p = proxy_dir / f"proxy_file_{i}.mp4"
        p.write_bytes(b"0" * 1024 * 100)  # 100KB each
        total += p.stat().st_size
        time.sleep(0.01)

    gen = ProxyGenerator(proxy_dir)
    # Evict until under 250KB
    gen._enforce_cache_limits(max_bytes=250 * 1024, min_age_seconds=0)
    remaining = sum(p.stat().st_size for p in proxy_dir.iterdir() if p.is_file())
    assert remaining <= 250 * 1024


def test_generate_analysis_proxy_instance_method_invocation(tmp_path, monkeypatch):
    """Regression: ensure instance `generate_analysis_proxy` is callable and wired
    correctly (previous bug passed `self` into source_path). We monkeypatch
    subprocess.run so the test is fast and deterministic.
    """
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"0" * 1024)
    proxy_dir = tmp_path / "proxies"
    proxy_dir.mkdir()
    out = proxy_dir / "clip_analysis_proxy.mp4"

    gen = ProxyGenerator(proxy_dir)

    # Capture the command and simulate successful ffmpeg
    called = {}
    class FakeCompleted:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, capture_output, text, timeout):
        called['cmd'] = cmd
        called['timeout'] = timeout
        # create the temp output file that ffmpeg would write
        tmp_out = Path(cmd[-1])
        tmp_out.write_bytes(b"proxy")
        return FakeCompleted()

    monkeypatch.setattr('subprocess.run', fake_run)

    # Should not raise and should call subprocess.run with a scale filter containing 240
    res = gen.ensure_analysis_proxy(str(src), height=240, force=True)
    assert res is not None
    assert any('scale=-2:240' in str(x) for x in called['cmd'])
    assert isinstance(called['timeout'], int) and called['timeout'] > 0
