package logger

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRotatingFileRotatesAtMaxBytes verifies that writes past maxBytes trigger
// rotation and that no more than maxBackups rotated files are retained.
func TestRotatingFileRotatesAtMaxBytes(t *testing.T) {
	dir := t.TempDir()
	// 100-byte cap, keep 2 backups.
	rf, err := newRotatingFile(dir, "test.log", 100, 2)
	if err != nil {
		t.Fatalf("newRotatingFile: %v", err)
	}
	defer rf.Close()

	// Each record is 40 bytes; writing 10 of them forces several rotations.
	rec := []byte(strings.Repeat("x", 39) + "\n")
	for i := 0; i < 10; i++ {
		if _, err := rf.Write(rec); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}

	// Active file must exist and be within the size cap.
	info, err := os.Stat(filepath.Join(dir, "test.log"))
	if err != nil {
		t.Fatalf("stat active file: %v", err)
	}
	if info.Size() > 100 {
		t.Fatalf("active file size %d exceeds cap 100", info.Size())
	}

	// At most maxBackups rotated files (test.log.1, test.log.2) may remain.
	backups := rf.listBackups()
	if len(backups) > 2 {
		t.Fatalf("expected at most 2 backups, got %d: %v", len(backups), backups)
	}
}

// TestEnableFileOutputWritesToFile verifies that EnableFileOutput tees log
// records to the configured file path.
func TestEnableFileOutputWritesToFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "app.log")

	if err := EnableFileOutput(path, 50, 3); err != nil {
		t.Fatalf("EnableFileOutput: %v", err)
	}
	t.Cleanup(func() {
		fileMu.Lock()
		if fileWriter != nil {
			_ = fileWriter.Close()
			fileWriter = nil
		}
		fileMu.Unlock()
	})

	SetLevel(LevelDebug)
	Infof("hello %s", "world")

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	if !strings.Contains(string(data), "hello world") {
		t.Fatalf("log file missing expected record, got %q", string(data))
	}
}
