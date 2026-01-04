# Documentation

Welcome to Montage AI documentation.

> **Montage AI** is the free, open-source, privacy-first AI rough-cut tool for creators and professionals.
> 
> Drop your clips + music → Beat-synced edit → Professional NLE handoff (OTIO/EDL).
> 
> **Cost:** Free (OSS) | **Privacy:** 100% local | **Interop:** OTIO-native

---

## Quick Links (First Time Here?)

- 🚀 **[Getting Started](getting-started.md)** — Install and create your first montage in 5 minutes
- 🎯 **[Features Overview](features.md)** — Transcript Editor, Shorts Studio, Quality Profiles, Pro Handoff
- 📖 **[Full Architecture](architecture.md)** — How the system works under the hood
- 🤔 **[Why Montage AI?](../docs/COMPETITIVE_ANALYSIS.md)** — How we compare to Descript, Adobe, Opus Clip
- 💼 **[Business Plan](../docs/BUSINESS_PLAN.md)** — For investors and enterprise partners

---

## For Users

| Doc | Description |
|-----|-------------|
| [Getting Started](getting-started.md) | Installation, first montage, Web UI & CLI |
| [Configuration](configuration.md) | All environment variables explained |
| [Features](features.md) | Styles, Creative Loop, enhancements, timeline export |
| [Responsible AI](responsible_ai.md) | Transparency, data handling, OSS-first policy |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |

---

## For Decision Makers

| Doc | Description |
|-----|-------------|
| [Competitive Analysis](COMPETITIVE_ANALYSIS.md) | How we compare to Descript, Adobe, Opus Clip (with scoring) |
| [Business Plan](BUSINESS_PLAN.md) | Market fit, revenue model, financials, investor thesis |
| [Strategic Roadmap](STRATEGY.md) | Q1-Q4 2026 priorities, market positioning, consolidation plan |
| [Marketing Playbook](MARKETING_PLAYBOOK.md) | LinkedIn, Twitter, Product Hunt, email templates |

---

## For Developers

| Doc | Description |
|-----|-------------|
| [Architecture](architecture.md) | System design, three-stage pipeline, module overview |
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
