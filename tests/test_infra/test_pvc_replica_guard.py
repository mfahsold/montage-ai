import glob
import yaml

BASE_PVC_FILE = "deploy/k3s/base/pvc.yaml"
OVERLAYS_DIR = "deploy/k3s/overlays"

# PVC names that, when RWO via local-path, must not be used by multi-replica workloads
CRITICAL_PVCS = {"montage-input", "montage-output", "montage-cache", "montage-assets"}


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(yaml.safe_load_all(f))


def test_no_rwo_localpath_with_replicas_gt_one():
    # collect RWO local-path PVC names from base
    base_docs = _load_yaml(BASE_PVC_FILE)
    rwo_pvcs = set()
    for doc in base_docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata", {})
        name = meta.get("name")
        spec = doc.get("spec", {})
        sc = spec.get("storageClassName")
        if name in CRITICAL_PVCS and sc and "local-path" in sc:
            rwo_pvcs.add(name)

    failures = []

    # scan overlays for deployments/patches that set replicas > 1 while still referencing RWO PVCs
    overlay_dirs = glob.glob(OVERLAYS_DIR + "/*")
    for od in overlay_dirs:
        yamls = glob.glob(f"{od}/*.yaml")
        overlay_has_rwx = False
        replicas_gt_one = False
        mounts_rwo = set()
        for y in yamls:
            try:
                docs = _load_yaml(y)
            except Exception:
                continue
            for d in docs:
                if not isinstance(d, dict):
                    continue
                kind = (d.get("kind") or "").lower()
                # detect RWX PVCs created in overlay
                if kind == "persistentvolumeclaim":
                    meta = d.get("metadata", {})
                    nm = meta.get("name", "")
                    spec = d.get("spec", {})
                    access = spec.get("accessModes", [])
                    if nm.endswith("-rwx") or "ReadWriteMany" in access:
                        overlay_has_rwx = True
                # detect replicas setting in Deployment/patch
                if kind == "deployment":
                    spec = d.get("spec", {})
                    rep = spec.get("replicas")
                    if rep and int(rep) > 1:
                        replicas_gt_one = True
                    template = spec.get("template", {})
                    podspec = template.get("spec", {})
                    for vol in podspec.get("volumes", []):
                        pvc = vol.get("persistentVolumeClaim")
                        if pvc:
                            claim = pvc.get("claimName")
                            if claim in rwo_pvcs:
                                mounts_rwo.add(claim)
        if replicas_gt_one and mounts_rwo and not overlay_has_rwx:
            failures.append((od, sorted(list(mounts_rwo))))

    assert not failures, (
        "Found overlays that declare replicas>1 while mounting RWO local-path PVCs without an RWX overlay: "
        f"{failures}"
    )
