import os
import subprocess
import tempfile
import shutil
import textwrap


def test_node_import_fallback_invoked(tmp_path, monkeypatch):
    repo_root = os.getcwd()
    scripts_path = os.path.join(repo_root, "scripts")
    load_script = os.path.join(scripts_path, "load-image-to-cluster.sh")

    # Backup original script
    backup = tmp_path / "load-image-to-cluster.sh.bak"
    shutil.copy(load_script, backup)

    try:
        # Create a fake loader script that logs its invocation
        log_file = tmp_path / "node_import.log"
        fake_script = textwrap.dedent(f"""
            #!/usr/bin/env bash
            echo "invoked: $@" >> "{log_file}"
            exit 0
        """)
        with open(load_script, "w") as f:
            f.write(fake_script)
        os.chmod(load_script, 0o755)

        # Fake docker binary (simulate registry and GHCR push failure to force node import)
        fake_docker = tmp_path / "docker"
        docker_log = tmp_path / "docker.log"
        fake_docker.write_text(textwrap.dedent(f"""
            #!/usr/bin/env bash
            echo "docker $@" >> "{docker_log}"
            cmd="$1"
            shift
            case "$cmd" in
              build)
                exit 0
                ;;
              run)
                echo "Worker module OK"
                exit 0
                ;;
              push)
                # Always fail pushes
                echo "push failed" >&2
                exit 1
                ;;
              login|tag|save)
                exit 0
                ;;
              *)
                exit 0
                ;;
            esac
        """))
        fake_docker.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH','')}")

        env = os.environ.copy()
        env.update({
            "REGISTRY": os.environ.get("REGISTRY", os.environ.get("REGISTRY_HOST", "localhost") + ':' + os.environ.get("REGISTRY_PORT", "5000")), 
            "IMAGE_NAME": "montage-ai",
            "IMAGE_TAG": "test",
            "BUILD_QUALITY": "preview",
            "NODE_IMPORT_NODES": "10.0.0.1",
            "SKIP_DEPLOY": "1",
        })

        script = os.path.abspath("scripts/build-and-deploy.sh")
        proc = subprocess.run([script], env=env, cwd=repo_root, capture_output=True, text=True)

        assert proc.returncode == 0, f"Script failed. stdout={proc.stdout} stderr={proc.stderr}"
        # Ensure the fake loader was invoked
        log_content = log_file.read_text()
        assert "invoked:" in log_content
    finally:
        # Restore original script
        shutil.copy(backup, load_script)
        os.chmod(load_script, 0o755)
