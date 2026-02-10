#!/bin/bash
JOB_ID=$1
if [ -z "$JOB_ID" ]; then
  echo "Usage: $0 <JOB_ID>"
  exit 1
fi

echo "Polling job $JOB_ID..."

# Start port-forward in background
kubectl port-forward -n montage-ai svc/montage-ai-web 8080:80 > /dev/null 2>&1 &
PF_PID=$!
sleep 2

cleanup() {
  echo "Stopping port-forward..."
  kill $PF_PID
}
trap cleanup EXIT

# Poll status
STATUS="queued"
while [[ "$STATUS" != "completed" && "$STATUS" != "failed" ]]; do
  RESP=$(curl -s http://localhost:8080/api/jobs/$JOB_ID)
  STATUS=$(echo $RESP | jq -r '.status')
  PHASE=$(echo $RESP | jq -r '.phase.label')
  echo "Status: $STATUS ($PHASE)"
  
  if [[ "$STATUS" == "failed" ]]; then
    echo "Job failed!"
    echo $RESP
    exit 1
  fi
  
  if [[ "$STATUS" == "completed" ]]; then
    break
  fi
  sleep 5
done

echo "Job completed. Downloading result..."
mkdir -p downloads
curl -L -o downloads/result_${JOB_ID}.mp4 http://localhost:8080/api/jobs/$JOB_ID/download

if [ -f "downloads/result_${JOB_ID}.mp4" ]; then
  echo "Download successful: downloads/result_${JOB_ID}.mp4"
  ls -lh downloads/result_${JOB_ID}.mp4
else
  echo "Download failed."
  exit 1
fi
