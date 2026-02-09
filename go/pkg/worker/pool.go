package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/mfahsold/montage-ai/go/internal/logger"
	"github.com/mfahsold/montage-ai/go/pkg/redis"
)

// Job represents a montage rendering job
type Job struct {
	ID              string            `json:"id"`
	Type            string            `json:"type"` // "render" or "analyze"
	InputDir        string            `json:"input_dir"`
	MusicFile       string            `json:"music_file"`
	OutputDir       string            `json:"output_dir"`
	Style           string            `json:"style"`
	CreativePrompt  string            `json:"creative_prompt"`
	VariantID       int               `json:"variant_id"`
	QualityProfile  string            `json:"quality_profile"`
	Params          map[string]interface{} `json:"params"`
}

// PoolOptions configures worker pool behavior
type PoolOptions struct {
	MaxJobsPerWorker int           // Restart worker after N jobs to prevent memory leaks
	RetryAttempts    int           // Number of retries on failure
	RetryDelayMs     int           // Delay between retries
}

// Pool manages a goroutine-based worker pool
type Pool struct {
	redis          *redis.Client
	maxWorkers     int
	queues         []string
	log            *logger.Logger
	opts           *PoolOptions
	
	// Concurrency control
	activeWorks    sync.WaitGroup
	workChan       chan *Job
	stopChan       chan struct{}
}

// NewPool creates a new worker pool
func NewPool(
	redisClient *redis.Client,
	maxWorkers int,
	queues []string,
	log *logger.Logger,
	opts *PoolOptions,
) *Pool {
	return &Pool{
		redis:      redisClient,
		maxWorkers: maxWorkers,
		queues:     queues,
		log:        log,
		opts:       opts,
		workChan:   make(chan *Job, maxWorkers),
		stopChan:   make(chan struct{}),
	}
}

// Listen blocks and processes jobs from Redis queues
func (p *Pool) Listen(ctx context.Context) error {
	p.log.Infof("🎬 Starting %d worker goroutines", p.maxWorkers)

	// Start worker goroutines
	for i := 0; i < p.maxWorkers; i++ {
		p.activeWorks.Add(1)
		go p.workerLoop(ctx, i)
	}

	// Main polling loop (in separate goroutine to not block)
	pollErr := make(chan error, 1)
	go func() {
		pollErr <- p.pollQueues(ctx)
	}()

	// Wait for context cancellation or poll error
	select {
	case <-ctx.Done():
		p.log.Infof("⏹️  Shutdown signal received")
		close(p.stopChan)
		// Wait for all workers to finish current jobs
		p.activeWorks.Wait()
		p.log.Infof("✅ All workers gracefully shut down")
		return nil
	case err := <-pollErr:
		close(p.stopChan)
		p.activeWorks.Wait()
		return fmt.Errorf("poll error: %w", err)
	}
}

// workerLoop processes jobs from workChan
func (p *Pool) workerLoop(ctx context.Context, workerID int) {
	defer p.activeWorks.Done()

	for {
		select {
		case <-p.stopChan:
			p.log.Debugf("🛑 Worker %d stopping", workerID)
			return
		case <-ctx.Done():
			p.log.Debugf("🛑 Worker %d context cancelled", workerID)
			return
		case job := <-p.workChan:
			if job == nil {
				return
			}
			p.processJob(ctx, workerID, job)
		}
	}
}

// processJob executes a single job
func (p *Pool) processJob(ctx context.Context, workerID int, job *Job) {
	p.log.Infof("[Worker %d] 🎬 Processing job: %s (type: %s)", workerID, job.ID, job.Type)

	start := time.Now()

	// TODO: Implement actual rendering logic
	// For now, this is a placeholder that demonstrates the structure
	
	// 1. Call Python analyzer if needed
	if job.Type == "render" {
		// err := callPythonAnalyzer(job)
		// if err != nil { ... }
	}

	// 2. Parallelize FFmpeg tasks (where goroutines shine!)
	// err := p.parallelizeRender(job)

	// Simulate work
	duration := time.Since(start)
	p.log.Infof("[Worker %d] ✅ Job %s completed in %.2fs", workerID, job.ID, duration.Seconds())
}

// pollQueues continuously polls Redis for new jobs
func (p *Pool) pollQueues(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			// Poll all queues for jobs
			for _, queueName := range p.queues {
				job, err := p.redis.PopJob(ctx, queueName)
				if err != nil {
					p.log.Warnf("❌ Error polling queue %s: %v", queueName, err)
					continue
				}

				if job != "" {
					p.log.Debugf("📥 Popped job from queue: %s", queueName)
					
					// Parse job
					var parsedJob Job
					if err := json.Unmarshal([]byte(job), &parsedJob); err != nil {
						p.log.Errorf("❌ Failed to parse job: %v", err)
						continue
					}

					// Send to worker (non-blocking with timeout)
					select {
					case p.workChan <- &parsedJob:
						// Job sent to worker
					case <-ctx.Done():
						return ctx.Err()
					case <-time.After(5 * time.Second):
						p.log.Warnf("⚠️  Work channel full, skipping job")
					}
				}
			}
		}
	}
}
