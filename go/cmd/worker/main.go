package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/mfahsold/montage-ai/go/internal/config"
	"github.com/mfahsold/montage-ai/go/internal/logger"
	"github.com/mfahsold/montage-ai/go/pkg/redis"
	"github.com/mfahsold/montage-ai/go/pkg/worker"
)

func main() {
	// Parse flags and environment
	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	log := logger.New(cfg.LogLevel)
	log.Infof("🚀 Starting Montage AI Go Worker")
	log.Infof("📡 Redis: %s:%d", cfg.RedisHost, cfg.RedisPort)
	log.Infof("👷 Worker CPUs: %d (goroutines: %d)", cfg.WorkerCPUs, cfg.WorkerCPUs*250)
	log.Infof("📋 Queues: %v", cfg.WorkerQueues)

	// Connect to Redis
	redisClient, err := redis.Connect(cfg.RedisHost, cfg.RedisPort, cfg.RedisPassword)
	if err != nil {
		log.Errorf("❌ Failed to connect to Redis: %v", err)
		os.Exit(1)
	}
	defer redisClient.Close()

	log.Infof("✅ Connected to Redis")

	// Create worker pool
	pool := worker.NewPool(
		redisClient,
		cfg.WorkerCPUs*250, // Scale goroutines by CPU count
		cfg.WorkerQueues,
		log,
		&worker.PoolOptions{
			MaxJobsPerWorker: 1000,
			RetryAttempts:    3,
			RetryDelayMs:     1000,
		},
	)

	// Setup signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

	// Start listening
	go func() {
		sig := <-sigChan
		log.Warnf("⚠️  Received signal: %v", sig)
		cancel()
	}()

	// Start work loop
	log.Infof("🎬 Worker ready. Listening for jobs...")
	if err := pool.Listen(ctx); err != nil {
		log.Errorf("❌ Worker error: %v", err)
		os.Exit(1)
	}

	log.Infof("✅ Worker shutdown gracefully")
}
