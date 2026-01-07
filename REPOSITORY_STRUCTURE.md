# Repository Structure - Clean Organization

This document describes the organized repository structure after consolidation.

## Public Repository Root

### User-Facing Documentation
- **README.md** — Main project overview, quick start, features
- **QUICK_START.md** — Fast reference for common tasks
- **CHANGELOG.md** — Release notes and version history
- **SECURITY.md** — Security policy and vulnerability reporting
- **CODE_OF_CONDUCT.md** — Community guidelines
- **CONTRIBUTING.md** — Contribution guidelines
- **THIRD_PARTY_LICENSES.md** — License information

### Development & Deployment
- **montage-ai.sh** — Main CLI tool (user-facing command)
- **Dockerfile** — Container build definition
- **docker-compose.yml** — Local development compose file
- **docker-compose.web.yml** — Web UI development variant

### Source Code & Tests
- **src/** — Python package source code
- **tests/** — Comprehensive test suite (586 tests)
- **scripts/** — Public utility scripts (build, validation)
- **deploy/** — Deployment configuration and tooling

## Public Subdirectories

### `src/montage_ai/`
Python package with all production code:
- **config.py** — Centralized configuration
- **core/** — Core orchestration engine
- **cli/** — Command-line interfaces
- **web_ui/** — Web interface
- **models/** — AI models and analysis
- **utils/** — Utilities and helpers

### `tests/`
Organized test suite (586 tests):
- **test_config.py** — Configuration tests
- **test_auto_reframe.py** — Vision/reframing tests
- **integration/** — End-to-end integration tests
- **performance_tests/** — Benchmarking tests

### `scripts/`
Public utility scripts:
- **ci.sh** — CI/CD pipeline runner
- **build_local_cache.sh** — Docker cache building
- **archive/** — Historical benchmarks (no longer active)

### `deploy/`
Deployment infrastructure:
- **config.env** — Centralized deployment configuration
- **CONFIGURATION.md** — Deployment config documentation
- **config-global.yaml** — Global configuration
- **k3s/** — Kubernetes deployment
  - **base/** — Base Kubernetes manifests
  - **build-and-push.sh** — Docker build and push
  - **deploy.sh** — Kubernetes deployment script
  - **undeploy.sh** — Cleanup script
  - **k3s-config.yaml** — K3s specific config

### `docs/`
User-facing documentation (21 files):
- **getting-started.md** — Installation and first steps
- **features.md** — Feature descriptions
- **configuration.md** — Configuration guide
- **architecture.md** — System architecture
- **performance-tuning.md** — Optimization guide
- **responsible-ai.md** — Ethics guidelines
- **privacy.md** — Privacy policy
- And more...

### `benchmark_results/`
Performance baseline tracking:
- **baseline.json** — Current baseline metrics
- **README.md** — Baseline documentation

### `data/`
Test data and development assets:
- **input/** — Test video clips
- **output/** — Generated outputs
- **music/** — Audio tracks
- **assets/** — Graphics and resources
- **luts/** — Color grading lookup tables

## Private Repository

### `private/docs/audits/`
Internal audit results:
- **DEPENDENCY_AUDIT.md** — Package analysis
- **DOCUMENTATION_CLEANUP_FINAL.md** — Cleanup summary

### `private/docs/status/`
Internal status tracking:
- **CLAUDE.md** — Agent session notes

### `private/archive/`
Historical internal documents:
- Strategic planning documents
- Deployment feedback and reports
- Business analysis
- Architecture decisions
- Implementation notes

### `private/scripts/`
Internal development tools:
- **FIX_DEPLOYMENT_NOW.sh** — Ad-hoc fixes
- **sync_and_run.sh** — Kubernetes debugging
- **test_in_docker.sh** — Docker testing
- **quick_check.sh** — Quick validation
- **verify_highlights.py** — Verification tool
- **check_env.py** — Environment checker

## Repository Principles

### Public Directory Rules
✓ Only user-facing content  
✓ Production-ready code  
✓ User documentation  
✓ Examples and guides  
✓ Public API reference  

### Private Directory Rules
✓ Internal development notes  
✓ Audit results  
✓ Status tracking  
✓ Strategic planning  
✓ Temporary debugging scripts  
✓ Historical documents  

### Enforcement
- Pre-push hook validates separation
- No internal docs can be committed to `docs/`
- No credentials allowed in any commit
- Status documents must go in `private/docs/status/`

## File Organization Examples

### Adding a New Feature
1. **Code**: `src/montage_ai/features/new_feature.py`
2. **Tests**: `tests/test_new_feature.py`
3. **Docs**: `docs/feature-guide.md` (if user-facing)
4. **Example**: `scripts/examples/feature_demo.sh` (if needed)

### Internal Debugging
1. **Script**: `private/scripts/debug_feature.sh`
2. **Notes**: `private/docs/status/debug-notes.md`
3. **Never**: Commit to public `scripts/` or `docs/`

### Performance Investigation
1. **Baseline**: Commit to `benchmark_results/baseline.json`
2. **Analysis**: `private/docs/status/perf-investigation.md`
3. **Results**: Document in PR description (public)

## Quick Commands

```bash
# Check repo structure
tree -L 2 -I '__pycache__|*.egg-info'

# Find all user-facing docs
find docs -name "*.md" | sort

# Find all internal docs
find private -name "*.md" | sort

# Validate structure
make validate-structure  # (if added)

# List all scripts
ls -la scripts/ private/scripts/
```

## Statistics

| Category | Count | Location |
|----------|-------|----------|
| Python source files | ~30 | `src/` |
| Test files | ~40 | `tests/` |
| Public docs | 21 | `docs/` |
| Private docs | ~100 | `private/` |
| Public scripts | ~6 | `scripts/` |
| Private scripts | ~7 | `private/scripts/` |
| Total tests | 586 | `tests/` |

## Next Steps

- ✅ Consolidation complete
- ⏳ Mirror private repository structure separately
- ⏳ Consider `.gitignore` for better separation
- ⏳ Document per-environment configurations
