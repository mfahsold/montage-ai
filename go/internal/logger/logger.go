package logger

import (
	"fmt"
	"strings"
	"time"

	"github.com/fatih/color"
)

// Level represents log level
type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
)

// Logger provides structured logging for the worker
type Logger struct {
	level Level
}

// New creates a new logger with given level
func New(levelStr string) *Logger {
	level := INFO
	switch strings.ToLower(levelStr) {
	case "debug":
		level = DEBUG
	case "info":
		level = INFO
	case "warn":
		level = WARN
	case "error":
		level = ERROR
	}
	return &Logger{level: level}
}

// Debugf logs debug-level messages
func (l *Logger) Debugf(format string, args ...interface{}) {
	if l.level <= DEBUG {
		l.print("DEBUG", color.BlueString(fmt.Sprintf(format, args...)))
	}
}

// Infof logs info-level messages
func (l *Logger) Infof(format string, args ...interface{}) {
	if l.level <= INFO {
		l.print("INFO", color.GreenString(fmt.Sprintf(format, args...)))
	}
}

// Warnf logs warning-level messages
func (l *Logger) Warnf(format string, args ...interface{}) {
	if l.level <= WARN {
		l.print("WARN", color.YellowString(fmt.Sprintf(format, args...)))
	}
}

// Errorf logs error-level messages
func (l *Logger) Errorf(format string, args ...interface{}) {
	if l.level <= ERROR {
		l.print("ERROR", color.RedString(fmt.Sprintf(format, args...)))
	}
}

// print is the internal printing function
func (l *Logger) print(level, msg string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fmt.Printf("[%s] [%s] %s\n", timestamp, level, msg)
}
