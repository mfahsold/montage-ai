# Montage AI Documentation

> **We polish pixels, we don't generate them.**

Free, open-source, privacy-first AI rough-cut tool for creators and professionals.

**Live site:** [mfahsold.github.io/montage-ai](https://mfahsold.github.io/montage-ai) — SEO landing, docs, and quickstarts.

---

## 🚀 Start Here

**[DX Guide](DX.md)** - The Golden Path  
Two workflows. Four commands. 90% of your work.

```bash
make dev       # Local development (5 sec feedback)
make cluster   # Cluster deployment (2-15 min)
```

---

## 📚 Documentation Index

### For Users
- **[Getting Started](getting-started.md)** - Installation & first montage
- **[Features](features.md)** - What Montage AI can do
- **[Configuration](configuration.md)** - Environment variables

### For Developers
- **[DX Guide](DX.md)** - Development workflow ⭐ Start here!
- **[Architecture](architecture.md)** - System design
- **[LLM Agents](llm-agents.md)** - AI coding guidelines
- **[Descript Alternative (Offline)](blog/descript-alternative.md)** - SEO landing & positioning

### For Operators
- **[Cluster Workflow](../deploy/CLUSTER_WORKFLOW.md)** - Kubernetes
- **[Troubleshooting](troubleshooting.md)** - Common issues

### Strategy
- **[Competitive Analysis](COMPETITIVE_ANALYSIS.md)** - Market position
- **[Strategy](STRATEGY.md)** - Product vision
- **[Marketing Playbook](MARKETING_PLAYBOOK.md)** - GTM strategy

---

## 🎯 Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| **Get started** | [Getting Started](getting-started.md) |
| **Develop locally** | [DX Guide](DX.md) |
| **Deploy to cluster** | [DX Guide](DX.md) |
| **Understand internals** | [Architecture](architecture.md) |
| **See all features** | [Features](features.md) |
| **Fix an error** | [Troubleshooting](troubleshooting.md) |

---

**Archive:** [archive/](archive/) - Historical docs

**Principles:**
1. DX.md is the starting point
2. Each doc has one clear purpose
3. Examples over theory
4. Archive, don't delete
| [Algorithms](algorithms.md) | Beat detection, scene analysis, clip selection |
| [AI Director](AI_DIRECTOR.md) | LLM prompt interpretation and fallback chain |
| [Models](models.md) | AI/ML libraries used and why |
| [LLM Agent Guidelines](llm-agents.md) | Coding principles for AI assistants |
| [RQ Migration](RQ_MIGRATION.md) | Redis Queue infrastructure (async jobs) |
| [Contributing](../CONTRIBUTING.md) | How to contribute code |

---

## Cloud & Infrastructure

| Doc | Description |
|-----|-------------|
| [Hybrid Workflow](hybrid-workflow.md) | Local + Cloud GPU setup for limited hardware |
| [cgpu Setup](cgpu-setup.md) | Google Cloud credentials for cgpu |
| [Cloud Offloading](cloud_offloading_implementation.md) | cgpu job types and implementation |
| [Stability Report](stability_report.md) | Memory protection and job admission |
| [Deployment Scenarios](deployment_scenarios.md) | Docker, Kubernetes, cloud hosting options |

---

## Roadmap & Strategy

| Doc | Description |
|-----|-------------|
| [Q1 2026 Strategy](STRATEGY.md) | Current focus: Transcript Editor, Shorts Studio 2.0, Pro Handoff |
| [Business Plan](BUSINESS_PLAN.md) | Market, financials, hiring, investor brief |
| [Competitive Analysis](COMPETITIVE_ANALYSIS.md) | Deep comparison with Descript, Adobe, Opus, auto-editor |
| [Strategic Backlog](STRATEGIC_BACKLOG.md) | Future features and research areas |

---

## Archive & References


Historical documents for reference:

| Doc | Description |
|-----|-------------|
| [Decisions](archive/decisions.md) | Architecture Decision Records (ADRs) |
| [Post-Production Pivot](archive/post_production_pivot.md) | Original pivot from generation to editing |
| [Editor Decomposition](archive/editor_decomposition_plan.md) | Monolithic→modular refactoring (completed) |
| [Cloud Pipeline Design](archive/cloud_pipeline_design.md) | cgpu_jobs abstraction design |
| [Cloud Pipeline Spec](archive/cloud_pipeline_technical_spec.md) | Technical implementation details |
| [Q1 2025 Roadmap](archive/next_steps_q1_2025.md) | Phase 1-4 implementation plan |
| [Integration Status](archive/integration_status_report.md) | LLM & cgpu integration report |
| [Offloading Analysis](archive/offloading_analysis.md) | cgpu offloading recommendations |
