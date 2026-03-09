# Documentation Consolidation Plan

## Current State: 54 Documentation Files

## Target State: 5 Core Documents

### 1. **README.md** (Single Source of Truth)
- Quick Start
- Installation
- Basic Configuration
- Troubleshooting Quick Fixes

### 2. **docs/ARCHITECTURE.md**
- System Overview
- Component Diagram
- Data Flow
- Pipeline Phases
- Extension Points

### 3. **docs/CONFIGURATION.md**
- YAML Config Reference
- Environment Variables
- Feature Flags
- LLM Backends
- Hardware Acceleration

### 4. **docs/API.md**
- REST API Endpoints
- WebSocket/SSE Events
- Python API (MontageBuilder)
- Style Templates Format

### 5. **docs/OPERATIONS.md**
- Deployment (Docker, K8s)
- Monitoring & Metrics
- Performance Tuning
- Troubleshooting Deep Dive
- Maintenance Tasks

## Migration Strategy

### To Archive (Move to docs/archive/):
- CGPU_AUTH*.md (5 files) → consolidate into operations
- cluster-deploy.md → operations
- k3s-*.md → operations
- CGPU_SETUP*.md → operations
- All troubleshooting split files → consolidate

### To Consolidate:
- configuration.md + config-defaults.md → CONFIGURATION.md
- features.md → CONFIGURATION.md
- performance-tuning.md → OPERATIONS.md
- All troubleshooting files → OPERATIONS.md

### To Delete:
- Duplicate installation guides
- Outdated migration guides
- Obsolete feature descriptions

## Implementation

1. Create new structure
2. Migrate content
3. Update links
4. Archive old files
5. Update README navigator
