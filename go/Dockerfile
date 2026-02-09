# Multi-stage build for Go Worker
# This runs in parallel with the Python API image

# Stage 1: Build Go worker
FROM golang:1.22-alpine AS builder

WORKDIR /build

# Copy go source
COPY go/go.mod go/go.sum ./
RUN go mod download

COPY go/ .

# Build for multiple platforms
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o montage-worker-amd64 ./cmd/worker
RUN CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o montage-worker-arm64 ./cmd/worker

# Stage 2: Runtime image
FROM alpine:3.19

# Install runtime dependencies
RUN apk add --no-cache \
    ffmpeg \
    python3 \
    py3-pip \
    ca-certificates

# Copy Go binary (auto-select based on platform)
COPY --from=builder /build/montage-worker-* /usr/local/bin/
# Symlink the correct binary
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      ln -s /usr/local/bin/montage-worker-arm64 /usr/local/bin/montage-worker; \
    else \
      ln -s /usr/local/bin/montage-worker-amd64 /usr/local/bin/montage-worker; \
    fi

# Copy Python dependencies (from main Dockerfile)
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create app user
RUN addgroup -S app && adduser -S app -G app

# Data directories
RUN mkdir -p /data/{input,music,output,assets,luts} && \
    chown -R app:app /data

USER app
WORKDIR /data

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD montage-worker --health-check || exit 1

# Start worker
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV WORKER_CPUS=4
ENV WORKER_QUEUES=default,heavy
ENV LOG_LEVEL=info

ENTRYPOINT ["/usr/local/bin/montage-worker"]
