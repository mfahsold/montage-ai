# Montage AI - K3s Local Cluster Deployment Checklist

## Overview

This checklist documents the verification process for Kubernetes (K3s) deployment readiness.  
While Docker local deployment has been fully tested, K3s cluster deployment requires an existing cluster.

---

## Prerequisites for K3s Deployment Testing

### Cluster Availability

To fully test K3s deployment, you need:

- [ ] K3s or Kubernetes cluster running
- [ ] kubectl configured and connected
- [ ] Cluster has minimum 16 GB RAM available
- [ ] At least 4 CPUs available
- [ ] StorageClass available (local-path or NFS)

### Alternative: Quick Local Setup Options

If no cluster is available, you can test K3s deployment with:

**Option A: k3d (Docker-based K3s)**
```bash
# Install k3d
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash

# Create local cluster
k3d cluster create montage-ai --agents 2 -p "8080:80@loadbalancer" -p "8443:443@loadbalancer"

# Get kubeconfig
k3d kubeconfig get montage-ai

# Verify
kubectl cluster-info
kubectl get nodes
```

**Option B: minikube**
```bash
# Install minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster (with NFS storage)
minikube start --cpus=4 --memory=16384 --addons storage-provisioner

# Enable NFS provisioner
minikube addons enable nfs-provisioner
```

**Option C: Kind (Kubernetes in Docker)**
```bash
# Install kind
go install sigs.k8s.io/kind@latest

# Create cluster
kind create cluster --name montage-ai

# Verify
kubectl cluster-info
kubectl get nodes
```

---

## Pre-Deployment Configuration Checklist

### Step 1: Copy Configuration Template
- [ ] `cp deploy/k3s/config-global.yaml.example deploy/k3s/config-global.yaml`
- [ ] File exists at: `deploy/k3s/config-global.yaml`

### Step 2: Edit Configuration (Replace <...> placeholders)

Fill in the following sections in `config-global.yaml`:

#### Registry Configuration
```yaml
registry:
  host: "<REGISTRY_HOST>"          # e.g., localhost, ghcr.io, 192.168.1.10
  port: "<REGISTRY_PORT>"          # e.g., 5000, 30500, blank for ghcr.io
  url: "<REGISTRY_URL>"            # e.g., localhost:5000, ghcr.io/myorg
```
- [ ] Registry host configured
- [ ] Registry port configured (or blank for default)
- [ ] Registry URL follows format

#### Cluster Configuration
```yaml
cluster:
  namespace: "<CLUSTER_NAMESPACE>"    # e.g., montage-ai, default
  clusterDomain: "<CLUSTER_DOMAIN>"   # e.g., cluster.local
```
- [ ] Cluster namespace set
- [ ] Cluster domain set (usually cluster.local)

#### Node Configuration
```yaml
cluster:
  nodes:
    - name: "control-plane"
      ip: "<CONTROL_PLANE_IP>"        # e.g., 192.168.1.100
      arch: "amd64"                   # or arm64 - verify with: kubectl get node <name> -o jsonpath='{.status.nodeInfo.architecture}'
```
- [ ] Control plane node IP set
- [ ] Node architecture verified with kubectl
- [ ] Worker nodes configured (if applicable)

**Verify architecture:**
```bash
kubectl get nodes -o wide
# or
kubectl get node <node-name> -o jsonpath='{.status.nodeInfo.architecture}'
```

#### Storage Configuration
```yaml
storage:
  classes:
    default: "local-path"           # or nfs-client, custom-class
  nfs:
    server: "<NFS_SERVER_IP>"       # e.g., 192.168.1.10 (if using NFS)
    path: "/mnt/nfs-montage"        # NFS export path
```
- [ ] Default storage class set
- [ ] Storage class exists in cluster
- [ ] NFS server configured (if using NFS)
- [ ] NFS path is writable

**Check available storage classes:**
```bash
kubectl get storageclass
```

#### Image Configuration
```yaml
images:
  montage_ai:
    name: "montage-ai"
    tag: "latest"
```
- [ ] Image name set
- [ ] Image tag set
- [ ] Image accessible from all nodes (test: `docker pull <registry>/<name>:<tag>`)

### Step 3: Validate Configuration (No <...> Placeholders)

Verify no unreplaced placeholders remain:
```bash
grep '<[A-Z_]*>' deploy/k3s/config-global.yaml
# Should return: (empty - no matches)
```
- [ ] No unreplaced placeholders found

---

## Pre-Flight Checks

### Step 4: Run Pre-Flight Validation

```bash
make -C deploy/k3s config
# Generates: deploy/k3s/base/cluster-config.env

make -C deploy/k3s pre-flight
```

Expected output:
```
[OK] kubectl found
[OK] kustomize found
[OK] make found
[OK] config-global.yaml exists
[OK] No unreplaced placeholders
[OK] kubectl connected to cluster
[OK] StorageClass available
[OK] Cluster node architecture(s): amd64, arm64
[OK] kustomize build (base) succeeds
All checks passed. Ready to deploy.
```

Checklist:
- [ ] `make -C deploy/k3s config` succeeds
- [ ] `make -C deploy/k3s pre-flight` shows all [OK]
- [ ] No blockers identified
- [ ] cluster-config.env generated successfully

---

## Deployment Steps

### Step 5: Build and Push Image

```bash
cd deploy/k3s
BUILD_MULTIARCH=true ./build-and-push.sh
```

Or single-arch:
```bash
./build-and-push.sh
```

Verification:
- [ ] Image builds successfully
- [ ] Image pushes to registry
- [ ] Image is multi-arch (if requested)
- [ ] All nodes can pull image (test: `docker pull <image>`)

### Step 6: Deploy Cluster

```bash
./deploy.sh cluster
```

Expected output:
```
Deploying Montage AI to cluster (namespace: montage-ai)...
✓ Creating namespace
✓ Creating secrets
✓ Creating storage (PVCs)
✓ Deploying services
✓ Deploying workers
✓ Deploying web UI
Deployment complete!
```

Verification:
- [ ] Deployment script exits with code 0
- [ ] Namespace created: `kubectl get namespace montage-ai`
- [ ] PVCs created: `kubectl get pvc -n montage-ai`
- [ ] Pods starting: `kubectl get pods -n montage-ai`

### Step 7: Bootstrap Storage

```bash
./bootstrap.sh
```

Expected output:
```
Initializing storage for Montage AI...
✓ PVCs are healthy
✓ .ready markers created
✓ Directories initialized
✓ Web UI accessible at: https://montage-ai.local
```

Verification:
- [ ] Bootstrap exits with code 0
- [ ] All PVCs are bound: `kubectl get pvc -n montage-ai`
- [ ] Web UI pods are running: `kubectl get pod -n montage-ai -l app=montage-ai-web`
- [ ] Data directories accessible

---

## Deployment Verification

### Step 8: Check Deployment Status

```bash
# Get pod status
kubectl get pods -n montage-ai

# Watch pod startup
kubectl get pods -n montage-ai -w

# Check logs
kubectl logs -n montage-ai -l app=montage-ai-web --tail=50

# Check for errors
kubectl get events -n montage-ai
```

Verification:
- [ ] All pods in Running state
- [ ] No CrashLoopBackOff or ImagePullBackOff
- [ ] No pending pods after 5 minutes
- [ ] Logs show successful initialization

### Step 9: Access Web UI

**Method A: Ingress (if configured)**
```bash
curl https://montage-ai.local/
# or open in browser: https://montage-ai.local
```

**Method B: Port Forward**
```bash
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:8080
# Then open: http://localhost:8080
```

**Method C: NodePort (if configured)**
```bash
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
NODE_PORT=$(kubectl get svc -n montage-ai montage-ai-web -o jsonpath='{.spec.ports[0].nodePort}')
curl http://$NODE_IP:$NODE_PORT/
```

Verification:
- [ ] HTTP/HTTPS connection successful
- [ ] Web UI loads without errors
- [ ] Upload page accessible
- [ ] Dashboard shows 0 jobs

### Step 10: Test Rendering Pipeline

```bash
# Upload test video via Web UI
# OR use API
curl -X POST http://localhost:8080/api/jobs \
  -F "file=@test_video.mp4" \
  -F "style=dynamic" \
  -F "quality=preview"

# Check job status
JOB_ID="<job-id-from-response>"
curl http://localhost:8080/api/jobs/$JOB_ID

# Watch pod logs
kubectl logs -n montage-ai -f job.batch/montage-render-$JOB_ID
```

Verification:
- [ ] Job created successfully
- [ ] Render pod starts within 30 seconds
- [ ] Processing shows beat detection → scene analysis → rendering
- [ ] Output video appears in /data/output PVC
- [ ] Job status transitions to "completed"

---

## Post-Deployment Verification

### Step 11: Verify Cluster Health

```bash
# Check resource usage
kubectl top nodes -n montage-ai
kubectl top pods -n montage-ai

# Check storage
kubectl get pvc -n montage-ai
df /data/input /data/output /data/music  # From storage node

# Check network
kubectl get svc -n montage-ai
kubectl get ingress -n montage-ai  # if using ingress
```

Verification:
- [ ] CPU usage reasonable (no spikes)
- [ ] Memory usage within limits
- [ ] PVCs bound and healthy
- [ ] Services/endpoints active
- [ ] Network connectivity working

### Step 12: Test Rollback & Recovery

```bash
# Trigger a pod restart
kubectl rollout restart deployment/montage-ai-web -n montage-ai

# Verify recovery
kubectl get pods -n montage-ai
kubectl logs -n montage-ai -l app=montage-ai-web --tail=20

# Verify data persistence
curl http://localhost:8080/api/jobs  # Should still show previous jobs
```

Verification:
- [ ] Pods restart successfully
- [ ] Services come back online within 30 seconds
- [ ] Data persisted across restarts
- [ ] No data loss after recovery

---

## Troubleshooting Checklist

If deployment fails, check:

### Configuration Issues
- [ ] No unreplaced placeholders in config-global.yaml
- [ ] Correct cluster namespace and domain
- [ ] Node architectures match config
- [ ] Storage class exists and is writable

### Image Issues
- [ ] Image built successfully
- [ ] Image pushed to correct registry
- [ ] All nodes can pull image
- [ ] Image supports correct architecture

### Storage Issues
- [ ] PVCs exist and are bound
- [ ] Storage provisioner running
- [ ] NFS server accessible (if using NFS)
- [ ] Directories writable and have sufficient space

### Network Issues
- [ ] Pods can reach each other
- [ ] DNS resolution works (nslookup montage-ai-web.montage-ai.svc.cluster.local)
- [ ] Ingress/LoadBalancer configured correctly
- [ ] Firewall rules allow necessary ports

### Resource Issues
- [ ] Cluster has sufficient free resources
- [ ] Pod resource requests not exceeding node resources
- [ ] No memory pressure or CPU throttling
- [ ] Eviction thresholds not triggered

### Logs & Debugging
```bash
# Get all events
kubectl get events -n montage-ai --sort-by='.lastTimestamp'

# Describe problem pod
kubectl describe pod <pod-name> -n montage-ai

# Get full logs
kubectl logs <pod-name> -n montage-ai --previous  # if crashed

# Debug with shell
kubectl exec -it <pod-name> -n montage-ai -- /bin/bash
```

---

## Final Acceptance Criteria

- [ ] Pre-flight checks all pass
- [ ] Image builds and pushes successfully
- [ ] Deployment script completes without errors
- [ ] Bootstrap script initializes storage
- [ ] All pods in Running state
- [ ] Web UI accessible and responsive
- [ ] Test render job completes successfully
- [ ] Output video is valid and plays
- [ ] Data persists across pod restarts
- [ ] No error logs in system pods
- [ ] Resource usage is within limits

---

## Next Steps

After successful K3s deployment:

1. **Configure monitoring** (Prometheus/Grafana)
2. **Set up logging** (ELK/Loki)
3. **Enable autoscaling** (KEDA)
4. **Configure backup** (Velero)
5. **Document runbooks** for operations team

---

## References

- [K3s Official Docs](https://docs.k3s.io/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Troubleshooting K3s](https://docs.k3s.io/advanced#troubleshooting)
- Montage AI Docs: [cluster-deploy.md](docs/cluster-deploy.md)

---

**Checklist Version:** 1.0  
**Last Updated:** February 9, 2026  
**Status:** Ready for K3s Deployment Testing
