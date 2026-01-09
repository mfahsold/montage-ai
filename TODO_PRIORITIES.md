# 🎯 HIGH-IMPACT TODOs für Montage AI (Priorisiert)

## TIER 1: CRITICAL (Breaking Bugs)

### 1. **Bare `except:` clauses → Specific Exception Handling**
- **Files**: `subprocess_manager.py` (lines 79, 179)
- **Impact**: Silent failures, hard to debug, PEP 8 violation
- **Fix**: Replace `except:` with specific exception types
- **Effort**: 30min
- **Value**: High (catches all errors properly)

```python
# BEFORE (subprocess_manager.py:79)
except:
    pass

# AFTER
except (OSError, ProcessError) as e:
    logger.warning(f"Could not terminate process: {e}")
```

---

## TIER 2: MEDIUM (DX & Reliability)

### 2. **Upload Path Sanitization (Security)**
- **File**: `web_ui/routes/session.py` (line 57)
- **Comment**: "TODO: Use a proper upload manager/path sanitizer"
- **Impact**: Path traversal vulnerability, arbitrary file write
- **Fix**: Add `werkzeug.utils.secure_filename()` + path validation
- **Effort**: 45min
- **Value**: High (security critical)

```python
# BEFORE (session.py:57)
# TODO: Use a proper upload manager/path sanitizer
file.save(upload_path)

# AFTER
from werkzeug.utils import secure_filename
from pathlib import Path

filename = secure_filename(file.filename)
upload_path = Path(UPLOAD_DIR) / filename
if not str(upload_path.resolve()).startswith(str(UPLOAD_DIR.resolve())):
    raise ValueError("Invalid upload path")
```

---

### 3. **Redis Error Handling → Custom Exceptions**
- **Files**: `redis_resilience.py` (lines 82, 109, 133)
- **Impact**: Generic `Exception` handling, poor error messages
- **Fix**: Use custom `RedisConnectionError`, `RedisTimeoutError`
- **Effort**: 1h
- **Value**: Medium (improves resilience pattern)

```python
# BEFORE (redis_resilience.py:82)
except Exception as e:
    logger.error(f"Redis error: {e}")

# AFTER
except redis.TimeoutError as e:
    raise RedisTimeoutError(f"Redis operation timed out: {e}")
except redis.ConnectionError as e:
    raise RedisConnectionError(f"Redis unavailable: {e}")
```

---

## TIER 3: TECH DEBT (Code Quality)

### 4. **Subprocess Manager → Use `subprocess.run()` properly**
- **File**: `subprocess_manager.py` (entire file)
- **Issue**: Custom wrapper around `Popen`, error-prone
- **Fix**: Simplify to use `subprocess.run()` with proper timeouts
- **Effort**: 1.5h
- **Value**: Low (refactor, not urgent)

### 5. **Type Hints → Full Coverage**
- **Files**: `creative_director.py`, `telemetry.py`, `proxy_generator.py`
- **Impact**: Runtime errors caught earlier, better IDE support
- **Effort**: 2-3h per file
- **Value**: Medium (improves robustness)

---

## RECOMMENDED PRIORITY FOR THIS SESSION

**Given deployment is running, I recommend:**

### **QUICK WIN** (30min)
1. Fix bare `except:` clauses in `subprocess_manager.py` → Better error capture

### **SECURITY** (45min)
2. Fix path traversal in `web_ui/routes/session.py` → Prevent arbitrary writes

### **OPTIONAL** (1h)
3. Add `RedisError` custom exceptions → Better resilience

**Total: ~2h for 2-3 critical fixes + 1 security fix**

All changes are low-risk (no breaking changes, pure improvements).
