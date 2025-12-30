#!/bin/bash
BUILD_PID=1213910
LOG_FILE="build.log"

echo "Monitoring build process $BUILD_PID..."
# Tail the log until the process exits
tail -f $LOG_FILE --pid=$BUILD_PID

echo "Build process finished."

# Check for success indicators in the log
if grep -q "exporting to image" $LOG_FILE; then
    echo "Build successful. Deploying to Jetson..."
    kubectl delete job montage-ai-render -n montage-ai --ignore-not-found
    kubectl apply -k deploy/k3s/overlays/jetson
    echo "Deployment applied. Watching pods..."
    kubectl get pods -n montage-ai -o wide
else
    echo "Build might have failed. Check $LOG_FILE for details."
fi
