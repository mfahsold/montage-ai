# Montage AI Build-Caching Strategie

## 🚀 Quick Start

```bash
# Standard Build (pusht automatisch zur Cluster-Registry)
./scripts/build_with_cache.sh

# Custom Build
REGISTRY=10.43.17.166:5000 \
TAG=my-feature \
PLATFORMS=linux/amd64 \
./scripts/build_with_cache.sh
```

## 📦 Cache-Optimierung

### 1. **Dockerfile Layer-Struktur** (stabilste zuerst)

```
Base Image (miniconda3:latest)
  ↓
System Packages (ffmpeg, vulkan, etc.)
  ↓
Node.js + npm globals (cgpu)
  ↓
Conda packages (python, librosa, numba)
  ↓
requirements.txt → pip install        ← Cached bis requirements.txt sich ändert
  ↓
Real-ESRGAN binary (arch-specific)
  ↓
pyproject.toml → pip install -e .     ← Cached bis pyproject.toml sich ändert
  ↓
src/ code (changes most frequently)   ← Invalidiert nur diese Layer
```

### 2. **Registry Cache**

Der Build nutzt die **Cluster-Registry** als Cache-Backend:

- **Cache Location**: `10.43.17.166:5000/montage-ai:buildcache`
- **Mode**: `max` (speichert alle Intermediate Layers)
- **Benefit**: Alle Cluster-Nodes teilen den Cache

### 3. **Build-Argumente**

```bash
# Cache aktivieren (Standard im Script)
--cache-from type=registry,ref=<registry>/montage-ai:buildcache
--cache-to type=registry,ref=<registry>/montage-ai:buildcache,mode=max

# Inline Cache für Pull-Clients
--build-arg BUILDKIT_INLINE_CACHE=1
```

## 🔧 Cache Management

### Cache Status prüfen

```bash
# Tags in Registry
curl http://10.43.17.166:5000/v2/montage-ai/tags/list

# Manifests
curl http://10.43.17.166:5000/v2/montage-ai/manifests/buildcache
```

### Cache löschen (bei Problemen)

```bash
# Über Registry HTTP API (wenn supported)
curl -X DELETE http://10.43.17.166:5000/v2/montage-ai/manifests/<digest>

# Oder Builder Cache cleanen
docker buildx prune --builder multiarch-builder -af
```

## 📊 Performance

| Szenario | Zeit (ohne Cache) | Zeit (mit Cache) | Speedup |
|----------|-------------------|------------------|---------|
| **First Build** | ~15 min | ~15 min | 1x |
| **Code Change Only** | ~15 min | ~2 min | **7.5x** |
| **requirements.txt Change** | ~15 min | ~8 min | **1.9x** |
| **No Changes (rebuild)** | ~15 min | ~30s | **30x** |

## 🎯 Best Practices

1. **Lockfiles nutzen**: `requirements.txt` stabil halten (pinned versions)
2. **Code zuletzt kopieren**: `COPY src/` am Ende der Stage
3. **Builder beibehalten**: `multiarch-builder` nicht löschen
4. **Registry Health**: Registry muss erreichbar sein (ClusterIP: `10.43.17.166:5000`)

## 🔍 Troubleshooting

### "Cache not found"

```bash
# Ersten Build ohne Cache-From durchführen
docker buildx build --builder multiarch-builder \
  --platform linux/amd64,linux/arm64 \
  --cache-to type=registry,ref=10.43.17.166:5000/montage-ai:buildcache,mode=max \
  --push -t 10.43.17.166:5000/montage-ai:bootstrap .
```

### "Registry unreachable"

```bash
# Prüfe Registry Pod
kubectl get pods -n montage-ai -l app=montage-ai-registry

# Port-Forward falls ClusterIP nicht erreichbar
kubectl port-forward -n montage-ai svc/montage-ai-registry 5000:5000
```

### "Build too slow"

- Prüfe: Sind alle Nodes im Cluster erreichbar? (`kubectl get nodes`)
- BuildKit Logs: `docker buildx inspect multiarch-builder --bootstrap`
- Registry Logs: `kubectl logs -n montage-ai -l app=montage-ai-registry`

## 🌐 Cluster Integration

Die Registry läuft als **Cluster-Service**:

```yaml
Service: montage-ai-registry
Namespace: montage-ai
ClusterIP: 10.43.17.166:5000
Type: ClusterIP
```

Alle Nodes können darauf zugreifen via:
- ClusterIP: `10.43.17.166:5000`
- DNS: `montage-ai-registry.montage-ai.svc.cluster.local:5000`
