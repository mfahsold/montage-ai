package config

import (
	"os"
	"strconv"
	"strings"
)

// Config holds all worker configuration
type Config struct {
	// Redis
	RedisHost     string
	RedisPort     int
	RedisPassword string

	// Worker
	WorkerCPUs   int      // Number of CPU cores
	WorkerQueues []string // Queues to listen to

	// Python interop
	PythonBin       string
	MontageAIPyPath string

	// Logging
	LogLevel string

	// Output
	OutputDir string
}

// LoadConfig loads configuration from environment variables
func LoadConfig() (*Config, error) {
	cfg := &Config{
		// Redis defaults
		RedisHost:     getEnv("REDIS_HOST", "localhost"),
		RedisPort:     getEnvInt("REDIS_PORT", 6379),
		RedisPassword: getEnv("REDIS_PASSWORD", ""),

		// Worker defaults
		WorkerCPUs: getEnvInt("WORKER_CPUS", 4),

		// Python defaults
		PythonBin:       getEnv("PYTHON_BIN", "/usr/bin/python3"),
		MontageAIPyPath: getEnv("MONTAGE_AI_PYTHON_PATH", "/app/src"),

		// Logging
		LogLevel: getEnv("LOG_LEVEL", "info"),

		// Output
		OutputDir: getEnv("OUTPUT_DIR", "/data/output"),
	}

	// Parse queues (comma-separated)
	queueStr := getEnv("WORKER_QUEUES", "default,heavy")
	cfg.WorkerQueues = parseQueues(queueStr)

	return cfg, nil
}

// Helper functions

func getEnv(key, defaultVal string) string {
	if val, exists := os.LookupEnv(key); exists {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	valStr := getEnv(key, "")
	if valStr == "" {
		return defaultVal
	}
	val, err := strconv.Atoi(valStr)
	if err != nil {
		return defaultVal
	}
	return val
}

func parseQueues(queueStr string) []string {
	var queues []string
	for _, q := range strings.Split(queueStr, ",") {
		if trimmed := strings.TrimSpace(q); trimmed != "" {
			queues = append(queues, trimmed)
		}
	}
	if len(queues) == 0 {
		queues = []string{"default"}
	}
	return queues
}
