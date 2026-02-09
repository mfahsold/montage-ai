# Repo Cleanup Plan

## Objective
Remove dev-only, test, and experimental files from public repo to reduce clutter and improve clarity.

## Phase 1: Large Non-Essential Directories (255 MB+)

### TO REMOVE (git rm -r)
These are test/benchmark outputs, not needed in public repo:

```bash
# Benchmarking results
git rm -r benchmark_results/

# Development downloads
git rm -r downloads/

# Legacy patches/experiments
git rm -r patches/

# Temporary build artifacts
git rm -r tmp/

# Private/internal docs (already in .gitignore but may be tracked)
git rm -r private/
```

**Expected savings: ~255 MB**

---

## Phase 2: Dev-Only YAML Files (Move to deploy/archive/)

### Kaniko Legacy Build Files (move to archive)
```bash
git mv kaniko-fallback.yaml deploy/k3s/overlays/archive/
git mv kaniko-flag-fix.yaml deploy/k3s/overlays/archive/
git mv kaniko-manifest-fix.yaml deploy/k3s/overlays/archive/
```

### Manual Test Jobs (move to archive)
```bash
git mv manual-test.yaml deploy/k3s/overlays/archive/
git mv montage-ai-manual-run.yaml deploy/k3s/overlays/archive/
git mv montage-ai-pipelinerun.yaml deploy/k3s/overlays/archive/
git mv seeder-job.yaml deploy/k3s/overlays/archive/
git mv service.build.yaml deploy/k3s/overlays/archive/
```

---

## Phase 3: Dev-Only Python Scripts (Move to scripts/archive/)

```bash
git mv submit_raw_jobs.py scripts/archive/
git mv submit_test_jobs.py scripts/archive/
git mv test_parameter_suggester.py scripts/archive/
git mv setup-distributed-builder-local.sh scripts/archive/
git mv verify_pipeline_deployment.sh scripts/archive/
```

---

## Phase 4: Benchmark Output Text Files (remove)

```bash
git rm micro_benchmark_baseline.txt
git rm micro_benchmark_optimized.txt
git rm vulture_findings*.txt
```

---

## Phase 5: .gitignore Updates

Update `.gitignore` to ensure these stay out:

```ini
# Dev-only outputs
benchmark_results/
downloads/
patches/
tmp/
private/
*.txt unless tracked elsewhere
```

---

## Git History Cleanup (Optional - only if needed)

If you want to **actually remove file history** from Git (reduce .git size):

```bash
# WARNING: This rewrites history - use only if beneficial
# Only do this if the public repo hasn't been forked much
git filter-branch --tree-filter 'rm -rf benchmark_results downloads patches tmp private' HEAD
```

**NOT RECOMMENDED** unless repo conflicts exist. Better to just `git rm` files and leave history.

---

## Summary

| Phase | Action | Savings | Status |
|-------|--------|---------|--------|
| 1 | Remove test dirs | -255 MB | ✅ Ready |
| 2 | Archive YAML files | Cleaner root | ✅ Ready |
| 3 | Archive Python scripts | Cleaner root | ✅ Ready |
| 4 | Remove benchmark txts | -1 MB | ✅ Ready |
| 5 | Update .gitignore | Prevents future | ✅ Ready |

---

## New Repo Structure (After Cleanup)

```
montage-ai/
├── docs/                    # Public documentation
├── src/                     # Python source (core, analysis, API)
├── cmd/                     # CLI scripts (montage-ai.sh)
├── go/                      # ✨ NEW: Go Worker (Phase 2 of Go Migration)
├── deploy/
│   ├── k3s/                 # Kubernetes configs
│   │   ├── base/            # Core manifests
│   │   ├── overlays/
│   │   │   ├── cluster/     # Production
│   │   │   └── archive/     # Legacy (moved from root)
│   │   └── README.md
│   ├── config.env           # Deployment vars
│   └── README.md
├── scripts/
│   ├── ci.sh                # CI runner
│   ├── check-*.sh           # Health checks
│   └── archive/             # Dev-only scripts (moved from root)
├── tests/                   # Test suite
├── .env.example             # Template only
├── docker-compose.yml       # Local dev
├── Dockerfile               # Container build
├── Makefile                 # Task runner
├── README.md                # Project intro
├── CONTRIBUTING.md          # Dev guide (updated with Go info)
├── LICENSE
└── requirements.txt         # Python deps
```

---

## Execution

Recommended order:
1. ✅ Create `/go/` directory structure (next step)
2. ✅ Create Go Worker POC
3. Run cleanup phases in order (commit after each phase)
4. Update docs to reference new structure

---

**Next:** Proceed with Go Worker scaffold while parallel branch for cleanup.
