# Cluster Workflow - Technical Reference

> **Note:** For most users, see [DX Guide](../docs/DX.md) - use `make cluster`

This document is for cluster administrators and CI/CD setup only.

---

## Prerequisites

```bash
# Verify cluster
kubectl get nodes  # Expected: 6 Ready nodes

# Check registry
kubectl get svc -n montage-ai montage-ai-registry
```

---

## Registry Cache Issue

**Symptom:** `ERROR: failed to configure registry cache importer`

**Status:** Build works despite error - BuildKit caches locally.

**Fix (Optional):**
```bash
# Configure buildx for HTTP registry
cat > /tmp/buildkitd.toml <<EOF
[registry."10.43.17.166:5000"]
  http = true
  insecure = true
EOF

docker buildx create --use --name multiarch-insecure \
  --driver docker-container \
  --driver-opt network=host \
  --config /tmp/buildkitd.toml
```

---

## Architecture Notes

- **ARM64 nodes:** Real-ESRGAN skipped (x86_64 only)
- **AMD64 nodes:** Full feature set (VAAPI, Vulkan)
- **Registry:** HTTP only (ClusterIP 10.43.17.166:5000)

---

**Most users:** Just run `make cluster` - this handles everything automatically.
