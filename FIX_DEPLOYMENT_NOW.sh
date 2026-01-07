#!/bin/bash
set -euo pipefail

# =====================================================
# SOFORTMASSNAHME: Deployment selector immutable Fix
# =====================================================
# Problem: Flux kann Deployment nicht reconcilen wegen unveränderlichem Selector
# Lösung: Deployment löschen (Pods behalten) und neu reconcilen

echo "🔧 Fixing montage-ai deployment selector immutable issue..."

# 1. Delete deployment but keep pods running
echo "1/3 Deleting deployment (keeping pods)..."
kubectl -n montage-ai delete deploy montage-ai-web --cascade=orphan

# 2. Force Flux to reconcile immediately
echo "2/3 Triggering Flux reconcile..."
flux reconcile kustomization montage-ai -n flux-system

# 3. Wait for deployment and check status
echo "3/3 Waiting for deployment to be ready..."
kubectl -n montage-ai rollout status deploy/montage-ai-web --timeout=5m

echo ""
echo "✅ Deployment fixed! Checking status..."
kubectl -n montage-ai get deploy montage-ai-web
kubectl -n montage-ai get pods -l app.kubernetes.io/name=montage-ai

echo ""
echo "📊 Image currently running:"
kubectl -n montage-ai get deploy montage-ai-web -o jsonpath='{.spec.template.spec.containers[0].image}'
echo ""
