import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:5001"  # Assuming port-forward or local access
VIDEO_FILENAME = "short_london.mp4"
TRANSCRIPT_FILENAME = "short_london.json"

def test_transcript_flow():
    print(f"Testing Transcript Flow against {BASE_URL}...")

    # 1. Check if video exists
    print(f"1. Checking video {VIDEO_FILENAME}...")
    try:
        r = requests.get(f"{BASE_URL}/api/files")
        r.raise_for_status()
        files = r.json()
        if VIDEO_FILENAME not in files['videos']:
            print(f"❌ Video {VIDEO_FILENAME} not found in {files['videos']}")
            # Try to upload it if missing? For now, assume it's there or fail.
            # In a real test, we'd upload it here.
            return False
        print("✅ Video found.")
    except Exception as e:
        print(f"❌ Failed to list files: {e}")
        return False

    # 2. Get Transcript
    print(f"2. Fetching transcript for {VIDEO_FILENAME}...")
    try:
        r = requests.get(f"{BASE_URL}/api/transcript/{VIDEO_FILENAME}")
        if r.status_code == 404:
            print("⚠️ Transcript not found. This is expected if not generated yet.")
            # In a real scenario, we might trigger generation.
            # For this test, we need the mock file we created.
            print("❌ Mock transcript should exist!")
            return False
        r.raise_for_status()
        transcript = r.json()
        print(f"✅ Transcript fetched. {len(transcript.get('segments', []))} segments.")
    except Exception as e:
        print(f"❌ Failed to fetch transcript: {e}")
        return False

    # 3. Trigger Render (Edit)
    print("3. Triggering Preview Render (removing first word)...")
    edits = [{"index": 0, "removed": True}] # Remove first word
    
    try:
        r = requests.post(f"{BASE_URL}/api/transcript/render", json={
            "filename": VIDEO_FILENAME,
            "edits": edits
        })
        r.raise_for_status()
        data = r.json()
        job_id = data['job_id']
        print(f"✅ Job started: {job_id}")
    except Exception as e:
        print(f"❌ Failed to start render job: {e}")
        return False

    # 4. Poll for Completion
    print("4. Polling for completion...")
    for i in range(30): # Wait up to 30s
        try:
            r = requests.get(f"{BASE_URL}/api/jobs/{job_id}")
            r.raise_for_status()
            job = r.json()
            status = job.get('status')
            print(f"   Status: {status}")
            
            if status == 'completed':
                print(f"✅ Job completed! Output: {job.get('output_file')}")
                return True
            if status == 'failed':
                print(f"❌ Job failed: {job.get('error')}")
                return False
                
            time.sleep(1)
        except Exception as e:
            print(f"❌ Polling error: {e}")
            return False
            
    print("❌ Timeout waiting for job.")
    return False

if __name__ == "__main__":
    success = test_transcript_flow()
    sys.exit(0 if success else 1)
