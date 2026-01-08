# Montage-AI Kubernetes - Failure Recovery & Runbook

## Quick Status Check

```bash
# Full health check
kubectl get all -n montage-ai
kubectl get pvc -n montage-ai
kubectl describe deployment montage-ai-web -n montage-ai
kubectl logs -f -n montage-ai -l app=montage-ai,component=web
```

---

## Common Failure Scenarios & Fixes

### 1. Pod Stuck in CrashLoopBackOff

**Symptoms:**
```
montage-ai-web-xxx   0/1   CrashLoopBackOff   5   2m
```

**Diagnosis:**
```bash
# Check pod logs
kubectl logs -n montage-ai <pod-name>

# Check events
kubectl describe pod -n montage-ai <pod-name>

# Check startup probe details
kubectl get pod -n montage-ai <pod-name> -o yaml | grep -A 20 "startupProbe"
```

**Common Causes & Fixes:**

| Cause | Error Log | Fix |
|-------|-----------|-----|
| Image pull failure | `ImagePullBackOff` | `kubectl describe pod -n montage-ai <pod-name>` → check registry URL |
| Invalid mount | `no such file or directory /data/input` | `kubectl get pvc -n montage-ai` → all must be Bound |
| Memory OOM | `OOMKilled` | Increase `.spec.containers[0].resources.limits.memory` |
| Port conflict | `Address already in use` | Check if port 8080 is available on node |

**Recovery:**
```bash
# Force restart pod
kubectl rollout restart deployment montage-ai-web -n montage-ai

# Or delete and recreate
kubectl delete pod -n montage-ai -l app=montage-ai,component=web
```

---

### 2. Pod Pending (Not Scheduling)

**Symptoms:**
```
montage-ai-web-xxx   0/1   Pending   0   5m
```

**Diagnosis:**
```bash
kubectl describe pod -n montage-ai <pod-name> | grep -A 10 "Events"
```

**Common Causes:**

| Cause | Fix |
|-------|-----|
| PVC not Bound | `kubectl get pvc -n montage-ai` → check all STATUS=Bound |
| No node with matching nodeSelector | `kubectl get nodes --show-labels \| grep amd64` |
| Insufficient resources | `kubectl top nodes` → check CPU/memory available |
| Node disk pressure | SSH to node: `df -h` → free up space if <20GB available |

**Recovery:**
```bash
# If nodeSelector problem:
kubectl patch deployment montage-ai-web -n montage-ai -p \
  '{"spec":{"template":{"spec":{"nodeSelector":null}}}}'

# Check PVC binding:
kubectl patch pvc montage-input -n montage-ai -p '{"spec":{"volumeName":"montage-pv-input"}}'
```

---

### 3. Web UI Returns 503 / Health Check Failing

**Symptoms:**
```bash
curl http://montage-ai.fluxibri.lan/health
# → Connection refused or timeout
```

**Diagnosis:**
```bash
# Check if pod is running
kubectl get pods -n montage-ai -l app=montage-ai,component=web

# Check service endpoints
kubectl get endpoints -n montage-ai

# Check ingress routing
kubectl get ingress -n montage-ai
kubectl describe ingress montage-ai -n montage-ai

# Test direct pod access
POD_IP=$(kubectl get pods -n montage-ai -l app=montage-ai,component=web -o jsonpath='{.items[0].status.podIP}')
curl http://$POD_IP:8080/health
```

**Common Causes & Fixes:**

| Cause | Fix |
|-------|-----|
| Pod not running | Check logs: `kubectl logs -n montage-ai <pod-name>` |
| Service has no endpoints | `kubectl get endpoints montage-ai-web -n montage-ai` should have IPs |
| Service selector wrong | Verify pod labels: `kubectl get pods -n montage-ai --show-labels` |
| Ingress not routing | Check Traefik: `kubectl logs -n kube-system -l app=traefik` |

**Recovery:**
```bash
# Recreate service (fix selector)
kubectl delete svc montage-ai-web -n montage-ai
kubectl apply -f deploy/k3s/app/service.yaml

# Recreate ingress
kubectl delete ingress montage-ai -n montage-ai
kubectl apply -f deploy/k3s/app/ingress.yaml
```

---

### 4. Storage Full / PVC Usage High

**Symptoms:**
```bash
# Pod running but /data/output is unreachable or slow
kubectl exec -n montage-ai <pod> -- df -h /data/
# → 100% usage
```

**Diagnosis:**
```bash
# Check PVC usage
kubectl exec -n montage-ai <pod> -- du -sh /data/*/

# Check node disk
kubectl exec -n montage-ai <pod> -- df -h /

# Monitor in real-time
watch 'kubectl exec -n montage-ai <pod> -- du -sh /data/*/'
```

**Recovery:**
```bash
# Option 1: Delete old rendered videos
kubectl exec -n montage-ai <pod> -- bash -c 'rm -rf /data/output/*.mp4'

# Option 2: Create backup and cleanup
./scripts/backup-montage-data.sh /tmp/montage-backup
kubectl exec -n montage-ai <pod> -- rm -rf /data/output /data/cache

# Option 3: Expand PVC (requires provisioner support)
kubectl patch pvc montage-output -n montage-ai -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

---

### 5. Node Goes Down (Hardware Failure)

**Symptoms:**
```
codeai-worker-amd64   NotReady   <none>   18h   v1.34.3+k3s1
```

**Impact:**
- Pod is stuck on failed node
- Data is inaccessible (HostPath on that node)
- Pod cannot reschedule (due to nodeAffinity)

**Recovery:**

**If node is recoverable:**
```bash
# Wait for node to come back online
kubectl get nodes -w | grep codeai-worker-amd64

# Pod should restart automatically
kubectl get pods -n montage-ai -w
```

**If node is permanently lost:**
```bash
# 1. Remove node from cluster
kubectl delete node codeai-worker-amd64

# 2. Restore from backup
scp user@nas:/backups/montage-ai-backup_YYYYMMDD_HHMMSS.tar.gz /tmp/
tar -xzf /tmp/montage-ai-backup_YYYYMMDD_HHMMSS.tar.gz

# 3. Create data on new node (once available)
# Copy backup to new node's /var/montage-ai/
scp /tmp/montage-ai-backup_YYYYMMDD_HHMMSS/output/* \
  root@<new-worker>:/var/montage-ai/output/

# 4. Recreate PV nodeAffinity to point to new node
kubectl patch pv montage-pv-output -p '{"spec":{"nodeAffinity":{"required":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"In","values":["<new-node-name>"]}]}]}}}}'

# 5. Restart pod
kubectl rollout restart deployment montage-ai-web -n montage-ai
```

---

### 6. Memory Leak / Growing Memory Usage

**Symptoms:**
```
# Memory keeps increasing over time
watch 'kubectl top pod -n montage-ai'
# Memory: 2GB → 3GB → 4GB → 5GB → OOMKilled
```

**Diagnosis:**
```bash
# Check top processes in pod
kubectl exec -n montage-ai <pod> -- ps aux | sort -k6 -nr | head -5

# Check for hanging FFmpeg processes
kubectl exec -n montage-ai <pod> -- ps aux | grep ffmpeg

# Check disk cache
kubectl exec -n montage-ai <pod> -- free -h
```

**Recovery:**
```bash
# Kill hanging processes
kubectl exec -n montage-ai <pod> -- killall ffmpeg

# Restart pod
kubectl delete pod -n montage-ai -l app=montage-ai,component=web

# If recurring, increase memory limits:
kubectl set resources deployment montage-ai-web -n montage-ai \
  --limits=memory=8Gi --requests=memory=4Gi
```

---

### 7. Network Connectivity Issues

**Symptoms:**
```
# Pod can't reach external APIs (Gemini, cgpu)
kubectl logs -n montage-ai <pod>
# → Connection refused, timeout, DNS errors
```

**Diagnosis:**
```bash
# Check pod DNS
kubectl exec -n montage-ai <pod> -- cat /etc/resolv.conf

# Test DNS resolution
kubectl exec -n montage-ai <pod> -- nslookup api.gemini.google.com

# Test connectivity
kubectl exec -n montage-ai <pod> -- curl -v https://api.gemini.google.com

# Check network policy
kubectl get networkpolicy -n montage-ai
```

**Recovery:**
```bash
# Check if NetworkPolicy is too restrictive
kubectl describe networkpolicy montage-ai-network-policy -n montage-ai

# Temporarily disable NetworkPolicy
kubectl delete networkpolicy montage-ai-network-policy -n montage-ai

# Then reapply with fixes
kubectl apply -f deploy/k3s/app/resilience.yaml
```

---

## Preventive Maintenance

### Daily
```bash
# Monitor pod health
kubectl get pods -n montage-ai -w

# Check resource usage
kubectl top pods -n montage-ai
kubectl top nodes
```

### Weekly
```bash
# Check PVC usage
kubectl exec -n montage-ai <pod> -- du -sh /data/*/

# Review logs for errors
kubectl logs -n montage-ai -l app=montage-ai,component=web --tail=1000 | grep ERROR

# Test backup
./scripts/backup-montage-data.sh /tmp/test-backup
ls -lh /tmp/test-backup
```

### Monthly
```bash
# Full backup
./scripts/backup-montage-data.sh /backups/monthly

# Node disk cleanup
# Remove old backups, temporary files, unused images
kubectl debug node/<node-name> -it --image=busybox -- \
  df -h /var/montage-ai && \
  du -sh /var/montage-ai/*

# Test disaster recovery procedure (offline)
# Simulate pod loss, restore from backup, verify data
```

---

## Emergency Contacts & Escalation

| Issue | Owner | Slack Channel |
|-------|-------|---------------|
| Pod/Deployment | @montage-devops | #montage-ai |
| Storage | @storage-team | #infrastructure |
| Network/Traefik | @network-team | #infrastructure |
| Cluster Health | @k3s-admin | #cluster-ops |

---

## Useful Commands Reference

```bash
# View all montage-ai resources
kubectl get all -n montage-ai

# Describe everything
kubectl describe all -n montage-ai

# Follow logs in real-time
kubectl logs -f -n montage-ai -l app=montage-ai,component=web

# Interactive shell in pod
kubectl exec -it -n montage-ai <pod-name> -- /bin/bash

# Copy files from pod
kubectl cp montage-ai/<pod-name>:/data/output /local/path

# Port forward for local testing
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:8080

# Check resource utilization
kubectl top pod -n montage-ai
kubectl top nodes

# Get pod events (latest first)
kubectl get events -n montage-ai --sort-by='.lastTimestamp' | tail -20

# Debug networking
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  /bin/sh -c 'nslookup montage-ai.fluxibri.lan'
```

---

**Last Updated:** 2026-01-08  
**Runbook Version:** 1.0  
**Maintained By:** DevOps Team
