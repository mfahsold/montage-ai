# Montage AI Documentation

Quick navigation for users, developers, and operators.

---

## 👤 For Users (Getting Started)

| Guide | Purpose |
|-------|---------|
| [Getting Started](getting-started.md) | Installation, first steps, CLI/Web UI |
| [Features](features.md) | What Montage AI can do |
| [Configuration](configuration.md) | Environment variables and tuning |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |
| [Performance Tuning](performance-tuning.md) | Optimization for your hardware |
| [Optional Dependencies](OPTIONAL_DEPENDENCIES.md) | Installation options: core, [ai], [web], [cloud] |

---

## 👨‍💻 For Developers

| Guide | Purpose |
|-------|---------|
| [Architecture](architecture.md) | System design and pipeline overview |
| [Algorithms](algorithms.md) | Technical details: beat detection, motion analysis, scene detection |
| [Models](models.md) | AI/ML libraries and versions |
| [LLM Agents Guide](llm-agents.md) | Coding principles for AI contributions |

---

## 🚀 For DevOps / Deployment

| Guide | Purpose |
|-------|---------|
| [CI/CD Setup](ci.md) | Vendor-agnostic CI pipeline (local, Jenkins, Drone) |
| [Cloud GPU Setup](cgpu-setup.md) | Optional: Google Cloud GPU integration |
| [Preview SLO (canonical)](operations/preview-slo.md) | Preview SLOs, metrics, and benchmark steps |
| [Preview Worker Sizing](operations/montage.yaml) | Sample CPU/memory/affinity for preview workers |

---

## 🔒 Security & Quality

| Guide | Purpose |
|-------|---------|
| [Dependency Management](DEPENDENCY_MANAGEMENT.md) | Dependency policies and update workflow |
| [Security Policy](../SECURITY.md) | Vulnerability reporting (responsible disclosure) |
| [Responsible AI](responsible-ai.md) | AI ethics and limitations |
| [Privacy](privacy.md) | Data handling and privacy |

---

## 📋 Quick References

- [QUICK_START.md](../QUICK_START.md) — Common commands and scenarios
- [README.md](../README.md) — Project overview
- [LICENSE](../LICENSE) — Licensing terms

---

## Repository Structure

```
montage-ai/
├── src/montage_ai/          # Main application code
├── tests/                   # Test suite (586 tests)
├── docs/                    # This documentation
├── deploy/k3s/              # Kubernetes deployment
├── scripts/ci.sh            # Vendor-agnostic CI runner
├── QUICK_START.md           # Common operations
├── SECURITY.md              # Vulnerability policy
└── README.md                # Project overview
```

---

## Installation Methods

```bash
# Core only (minimal)
pip install montage-ai

# With AI features (smart reframing, face detection)
pip install montage-ai[ai]

# With Web UI (Flask + job queue)
pip install montage-ai[web]

# With Cloud GPU (optional upscaling)
pip install montage-ai[cloud]

# Everything (development)
pip install montage-ai[all]
```

See [Optional Dependencies](OPTIONAL_DEPENDENCIES.md) for details.

---

## Common Tasks

**First time using Montage AI?**  
→ Read [Getting Started](getting-started.md)

**Setting up for production?**  
→ Read [Configuration](configuration.md) and [CI/CD Setup](ci.md)

**Slow performance?**  
→ Read [Performance Tuning](performance-tuning.md)

**Want to contribute?**  
→ Read [LLM Agents Guide](llm-agents.md) and check [Architecture](architecture.md)

**Security question?**  
→ Read [Security Policy](../SECURITY.md) and [Responsible AI](responsible-ai.md)

---

## Feedback

- 🐛 [Report Issues](https://github.com/mfahsold/montage-ai/issues)
- 💬 [Discussions](https://github.com/mfahsold/montage-ai/discussions)
- 🔒 [Security](../SECURITY.md)
