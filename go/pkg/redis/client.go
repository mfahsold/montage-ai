package redis

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Client wraps redis.Client for Montage AI
type Client struct {
	client *redis.Client
}

// Connect creates a new Redis connection
func Connect(host string, port int, password string) (*Client, error) {
	redisClient := redis.NewClient(&redis.Options{
		Addr:         fmt.Sprintf("%s:%d", host, port),
		Password:     password,
		DB:           0,
		MaxRetries:   3,
		PoolSize:     10,
		MinIdleConns: 2,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := redisClient.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping failed: %w", err)
	}

	return &Client{client: redisClient}, nil
}

// PopJob pops the next job from a queue (RQ-compatible)
// Returns job data or nil if no jobs available
func (c *Client) PopJob(ctx context.Context, queueName string) (string, error) {
	// RQ uses a Redis list with key "rq:queue:{name}"
	key := fmt.Sprintf("rq:queue:%s", queueName)

	// LPOP pops from left (FIFO)
	val, err := c.client.LPop(ctx, key).Result()
	if err == redis.Nil {
		// No jobs in queue
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("lpop error: %w", err)
	}

	return val, nil
}

// PushJob pushes a job to a queue (RQ-compatible)
func (c *Client) PushJob(ctx context.Context, queueName string, jobData string) error {
	key := fmt.Sprintf("rq:queue:%s", queueName)

	if err := c.client.RPush(ctx, key, jobData).Err(); err != nil {
		return fmt.Errorf("rpush error: %w", err)
	}

	return nil
}

// GetJobStatus retrieves job status from Redis (RQ-compatible)
func (c *Client) GetJobStatus(ctx context.Context, jobID string) (string, error) {
	// RQ stores job hash at "rq:job:{job_id}"
	key := fmt.Sprintf("rq:job:%s", jobID)

	status, err := c.client.HGet(ctx, key, "status").Result()
	if err == redis.Nil {
		return "nonexistent", nil
	}
	if err != nil {
		return "", fmt.Errorf("hget error: %w", err)
	}

	return status, nil
}

// SetJobStatus sets job status (for workers to communicate back)
func (c *Client) SetJobStatus(ctx context.Context, jobID string, status string) error {
	key := fmt.Sprintf("rq:job:%s", jobID)

	if err := c.client.HSet(ctx, key, "status", status).Err(); err != nil {
		return fmt.Errorf("hset error: %w", err)
	}

	return nil
}

// Close closes the Redis connection
func (c *Client) Close() error {
	return c.client.Close()
}

// Health checks Redis connectivity
func (c *Client) Health(ctx context.Context) error {
	return c.client.Ping(ctx).Err()
}
