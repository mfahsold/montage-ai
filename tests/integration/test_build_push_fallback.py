import os
import subprocess
import tempfile
import textwrap
import shutil


def test_build_and_deploy_uses_ghcr_fallback(tmp_path, monkeypatch, capsys):
    # Arrange: Create a fake docker that simulates registry push failure but GHCR success
    fake_docker = tmp_path / "docker"
    log = tmp_path / "docker.log"
    fake_docker.write_text(textwrap.dedent(f"""
        #!/usr/bin/env bash
        echo "docker $@" >> "{log}"
        cmd="$1"
        shift
        case "$cmd" in
          build)
            exit 0
            ;;
          run)
            # Simulate verification output
            echo "Worker module OK"
            exit 0
            ;;
          push)
            # If pushing to internal registry, fail
            if [[ "$*" == *"${{REGISTRY}}"* ]]; then
                echo "push to registry failed" >&2
                exit 1
            fi
            # If pushing to ghcr, succeed
            if [[ "$*" == *"ghcr.io"* ]]; then
                echo "pushed to ghcr"
                exit 0
            fi
            exit 0
            ;;
          login)
            exit 0
            ;;
          tag)
            exit 0
            ;;
          save)
            exit 0
            ;;
          *)
            exit 0
            ;;
        esac
        """))
    fake_docker.chmod(0o755)

    # Prepend tmp_path to PATH so our fake docker is used
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")

    # Set env to simulate registry unreachable and enable GHCR token
    env = os.environ.copy()
    env.update({
        "REGISTRY": "192.168.1.12:30500",
        "IMAGE_NAME": "montage-ai",
        "IMAGE_TAG": "test",
        "BUILD_QUALITY": "preview",
        "GHCR_PAT": "dummy-token",
        "GITHUB_REPOSITORY_OWNER": "testorg",
        "GITHUB_ACTOR": "testactor",
        "SKIP_DEPLOY": "1",
    })

    # Run the script
    script = os.path.abspath("scripts/build-and-deploy.sh")
    proc = subprocess.run([script], env=env, cwd=os.getcwd(), capture_output=True, text=True)

    print(proc.stdout)
    print(proc.stderr)

    # Assert the script exited successfully and that GHCR push was attempted
    assert proc.returncode == 0, f"Script failed: stdout={proc.stdout} stderr={proc.stderr}"
    assert "ghcr.io" in proc.stdout.lower() or "pushed to ghcr" in proc.stdout.lower()
    # Also ensure the docker log contains a push attempt to ghcr
    content = log.read_text()
    assert "push ghcr.io" in content or "push ghcr.io/testorg/montage-ai" in content
