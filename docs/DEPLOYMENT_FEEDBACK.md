# Deployment Process Findings & Recommendations

## Overview
During the multi-architecture build implementation for `montage-ai` (Fluxibri Core), several networking and configuration friction points were identified. This document summarizes the findings and the applied fixes.

## Issues Identified

### 1. Build Instability on ARM64 Nodes
**Symptom:** The `pip install` step frequently failed on ARM-based build nodes with `ReadTimeoutError`.
**Root Cause:** Network latency or limited bandwidth on some distributed build nodes caused standard timeouts to trigger during heavy package downloads.
**Resolution:** Increased the default pip timeout configuration in the `Dockerfile`.
```dockerfile
# Added --default-timeout=100
RUN pip install --default-timeout=100 --no-cache-dir --prefer-binary -r requirements.txt
```

### 2. Private Registry Trust (x509 Errors)
**Symptom:** `docker buildx build --push` failed during the final manifest export phase with `x509: certificate signed by unknown authority`.
**Root Cause:** The `docker buildx` driver handles the multi-arch manifest merge on the *client* (host) machine, not exclusively within the BuildKit container. While BuildKit was configured with `insecure = true` via `buildkitd.toml`, the host's Docker client rejected the self-signed certificate of the registry (`192.168.1.12:5000`).
**Resolution:** Correctly identified that the host machine's trust store needed the registry's certificate.
**Action Taken:** 
- Fetched registry certificate: `openssl s_client -showcerts -connect 192.168.1.12:5000 </dev/null`
- Installed to host: `cp registry.crt /usr/local/share/ca-certificates/registry.crt && update-ca-certificates`

### 3. Registry Configuration
**Symptom:** Build scripts were defaulting to outdated IP addresses.
**Resolution:** Updated `scripts/build_multiarch.sh` to point to the active registry `192.168.1.12:5000`.

## Recommendations for CI/CD

1. **Automate Certificate Trust:**
   For CI runners, ensure the registry CA certificate is injected into the runner's trust store during initialization, or switch to a public/Let's Encrypt certificate for the registry to avoid x509 issues entirely.

2. **BuildKit Config Management:**
   Keep `buildkitd.toml` in the repository to allow developers to easily replicate the insecure registry setup in local dev environments if they cannot install system certificates.

3. **Pip Caching:**
   Consider using a local PyPI mirror or persistent caching volume for `pip` if build timeouts persist, though the timeout bump has resolved immediately observable issues.
