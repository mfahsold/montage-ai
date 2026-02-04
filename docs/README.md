# Montage AI Documentation

> **We polish pixels, we don't generate them.**

Free, open-source, privacy-first AI rough-cut tool for creators and professionals.

**Live site:** [mfahsold.github.io/montage-ai](https://mfahsold.github.io/montage-ai)

---

## Quick Start

Start here:
- [Getting Started](getting-started.md) — full walkthrough
- [`QUICK_START.md`](../QUICK_START.md) — command cheat sheet
- [K8s Deployment](../deploy/k3s/README.md) — cluster setup

---

## Documentation Index

### Getting Started

| Document | Description |
| ------ | ------------- |
| [Getting Started](getting-started.md) | Installation & first montage |
| [Features](features.md) | What Montage AI can do |
| [Configuration](configuration.md) | Environment variables |
| [Config Defaults (code-aligned)](config-defaults.md) | Audited default values |
| [CLI Reference](CLI_REFERENCE.md) | Command-line usage |

### Deployment

| Document | Description |
| ------ | ------------- |
| [Deployment Guide](../deploy/README.md) | All deployment options |
| [K8s Deployment](../deploy/k3s/README.md) | Kubernetes/K3s deployment |
| [Kubernetes Runbook](KUBERNETES_RUNBOOK.md) | Operations (public stub; internal on request) |
| [Preview SLO (canonical)](operations/preview-slo.md) | Preview SLOs and benchmark steps |
| [Preview Worker Sizing](operations/montage.yaml) | Sample CPU/memory/affinity |
| [Operations Hub](operations/README.md) | Operational runbooks index |
| [Troubleshooting](troubleshooting.md) | Common issues & fixes |

### Architecture

| Document | Description |
| ------ | ------------- |
| [Architecture](architecture.md) | System design |
| [Algorithms](algorithms.md) | Beat detection, scene analysis |
| [Models](models.md) | AI/ML libraries used |
| [LLM Agent Guidelines](llm-agents.md) | AI coding principles |

### Cloud & Infrastructure

| Document | Description |
| ------ | ------------- |
| [cgpu Setup](cgpu-setup.md) | Google Cloud GPU credentials |
| [Optional Dependencies](OPTIONAL_DEPENDENCIES.md) | AI/ML library options |

### Reference

| Document | Description |
| ------ | ------------- |
| [Parameter Reference](PARAMETER_REFERENCE.md) | All configurable parameters |
| [Style Quick Reference](STYLE_QUICK_REFERENCE.md) | Editing styles |
| [AI Director Tuning](AI_DIRECTOR_PARAMETER_TUNING.md) | LLM prompt tuning |

### Strategy

| Document | Description |
| ------ | ------------- |
| [Roadmap](roadmap/README.md) | Public roadmap overview |

---

## Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| Get started | [Getting Started](getting-started.md) |
| Deploy to K8s | [K8s Deployment](../deploy/k3s/README.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |
| Understand internals | [Architecture](architecture.md) |
| See all features | [Features](features.md) |

---

**Archive:** Historical/internal notes live in the private docs set.
**Contributing:** [CONTRIBUTING.md](../CONTRIBUTING.md)

See `PUBLIC_DOCS_POLICY.md` for the public vs private documentation rules.
