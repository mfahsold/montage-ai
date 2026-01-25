import os
import time
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("RUN_DEV_E2E", "false").lower() != "true", reason="Dev e2e tests are opt-in")
def test_enqueue_preview_and_produce_artifact_on_dev_cluster(k8s_client):
    """Opt-in: enqueue a small preview job on a dev cluster and assert preview appears on PVC.
    Requires: RUN_DEV_E2E=true and a running DEV overlay with montage-input-rwx / montage-output-rwx PVCs.
    """
    namespace = os.getenv("MONTAGE_NAMESPACE") or os.getenv("CLUSTER_NAMESPACE", "montage-ai")
    job_name = f"e2e-preview-smoke-{int(time.time())}"

    # Use the project's Job manifest generator if available; otherwise skip
    try:
        from montage_ai.cluster.job_submitter import submit_generic_job
    except Exception:
        pytest.skip("job_submitter not available in this test environment")

    registry_url = os.getenv("REGISTRY_URL", "127.0.0.1:30500")
    image_name = os.getenv("IMAGE_NAME", "montage-ai")
    image_tag = os.getenv("MONTAGE_TEST_IMAGE_TAG", "dev")
    default_image = f"{registry_url}/{image_name}:{image_tag}"
    image_ref = os.getenv("MONTAGE_TEST_IMAGE", default_image)

    manifest = {
        "metadata": {"name": job_name, "namespace": namespace},
        "spec": {"template": {"spec": {"containers": [{
            "name": "runner",
            "image": image_ref,
            "env": [{"name": "QUALITY_PROFILE", "value": "preview"}],
            "volumeMounts": [{"name": "data-input", "mountPath": "/data/input"}, {"name": "data-output", "mountPath": "/data/output"}]
        }]}}}
    }

    submit_generic_job(manifest, namespace=namespace, wait=True, timeout=180)

    if hasattr(k8s_client, "list_pvc_files"):
        files = k8s_client.list_pvc_files("montage-output-rwx")
        assert any(f.startswith("preview_") and f.endswith(".mp4") for f in files)
    else:
        pytest.skip("Cluster PVC access not available in this test environment")
