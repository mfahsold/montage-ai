#!/bin/bash
# Backup script for Montage-AI persistent data
# Usage: ./backup-montage-data.sh [backup-dir]

set -e

BACKUP_DIR="${1:-.}"
POD_NAME=$(kubectl get pods -n montage-ai -l app=montage-ai,component=web -o jsonpath='{.items[0].metadata.name}')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="montage-ai-backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

echo "🔄 Starting Montage-AI data backup..."
echo "Pod: $POD_NAME"
echo "Target: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup each data directory
echo "📦 Backing up /data/input..."
kubectl cp montage-ai/$POD_NAME:/data/input "$BACKUP_PATH/input" -c montage-ai || true

echo "📦 Backing up /data/output..."
kubectl cp montage-ai/$POD_NAME:/data/output "$BACKUP_PATH/output" -c montage-ai || true

echo "📦 Backing up /data/cache..."
kubectl cp montage-ai/$POD_NAME:/data/cache "$BACKUP_PATH/cache" -c montage-ai || true

echo "📦 Backing up /data/music..."
kubectl cp montage-ai/$POD_NAME:/data/music "$BACKUP_PATH/music" -c montage-ai || true

# Create metadata file
cat > "$BACKUP_PATH/BACKUP_INFO.txt" <<EOF
Montage-AI Backup Metadata
==========================
Timestamp: $TIMESTAMP
Pod: $POD_NAME
Node: $(kubectl get pods -n montage-ai $POD_NAME -o jsonpath='{.spec.nodeName}')
Cluster: $(kubectl config current-context)

Contents:
- /data/input    (Source footage)
- /data/output   (Rendered videos)
- /data/cache    (Analysis cache)
- /data/music    (Music library)

Restore Instructions:
1. kubectl cp $BACKUP_PATH/input montage-ai/\$POD_NAME:/data/input
2. kubectl cp $BACKUP_PATH/output montage-ai/\$POD_NAME:/data/output
3. kubectl cp $BACKUP_PATH/cache montage-ai/\$POD_NAME:/data/cache
4. kubectl cp $BACKUP_PATH/music montage-ai/\$POD_NAME:/data/music
EOF

# Create tarball for easy transfer
echo "📦 Creating compressed archive..."
tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME" 2>/dev/null || true

# Print summary
BACKUP_SIZE=$(du -sh "$BACKUP_PATH" 2>/dev/null | cut -f1)
echo ""
echo "✅ Backup complete!"
echo "Location: $BACKUP_PATH"
echo "Size: $BACKUP_SIZE"
echo "Archive: ${BACKUP_PATH}.tar.gz"
echo ""
echo "💾 Next steps:"
echo "  1. Copy archive to NAS: scp ${BACKUP_PATH}.tar.gz user@nas:/backups/"
echo "  2. Verify: tar -tzf ${BACKUP_PATH}.tar.gz | head"
echo "  3. Cleanup old backups: find . -name 'montage-ai-backup_*' -mtime +30 -delete"
