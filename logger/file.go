package logger

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// rotatingFile is a minimal size-based rotating log writer. It is intentionally
// dependency-free (the project pins only github.com/google/uuid) and safe for
// concurrent use by the leveled loggers, which may write from many goroutines.
//
// On each Write it checks whether appending would exceed maxBytes; if so it
// rotates the current file to "<name>.1", shifting older files up to maxBackups,
// then opens a fresh file. Rotation and writing are guarded by a single mutex so
// concurrent Debugf/Infof/etc. calls never interleave or race the rotation.
type rotatingFile struct {
	mu         sync.Mutex
	dir        string
	base       string // base filename, e.g. "kiro-go.log"
	maxBytes   int64
	maxBackups int
	size       int64
	file       *os.File
}

// newRotatingFile opens (creating dirs as needed) dir/base for appending and
// returns a writer that rotates at maxBytes, keeping maxBackups rotated files.
func newRotatingFile(dir, base string, maxBytes int64, maxBackups int) (*rotatingFile, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log dir: %w", err)
	}
	rf := &rotatingFile{
		dir:        dir,
		base:       base,
		maxBytes:   maxBytes,
		maxBackups: maxBackups,
	}
	if err := rf.openExisting(); err != nil {
		return nil, err
	}
	return rf, nil
}

func (rf *rotatingFile) path() string {
	return filepath.Join(rf.dir, rf.base)
}

// openExisting opens the active log file in append mode and records its size so
// rotation accounting survives process restarts.
func (rf *rotatingFile) openExisting() error {
	f, err := os.OpenFile(rf.path(), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("open log file: %w", err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("stat log file: %w", err)
	}
	rf.file = f
	rf.size = info.Size()
	return nil
}

// Write implements io.Writer. It rotates first if the incoming write would push
// the file past maxBytes, so a single record is never split across files.
func (rf *rotatingFile) Write(p []byte) (int, error) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	if rf.file == nil {
		return 0, fmt.Errorf("log file not open")
	}
	if rf.maxBytes > 0 && rf.size+int64(len(p)) > rf.maxBytes {
		if err := rf.rotate(); err != nil {
			// Rotation failed; keep writing to the current file rather than
			// dropping logs.
			n, werr := rf.file.Write(p)
			rf.size += int64(n)
			if werr != nil {
				return n, werr
			}
			return n, err
		}
	}
	n, err := rf.file.Write(p)
	rf.size += int64(n)
	return n, err
}

// rotate closes the current file, shifts "<base>.N" → "<base>.N+1" (dropping the
// oldest beyond maxBackups), moves the active file to "<base>.1", and opens a
// fresh active file.
func (rf *rotatingFile) rotate() error {
	if err := rf.file.Close(); err != nil {
		return err
	}

	// Shift existing backups upward, oldest first, so we never clobber.
	for i := rf.maxBackups; i >= 1; i-- {
		src := fmt.Sprintf("%s.%d", rf.path(), i)
		if i == rf.maxBackups {
			_ = os.Remove(src) // drop the oldest
			continue
		}
		dst := fmt.Sprintf("%s.%d", rf.path(), i+1)
		if _, err := os.Stat(src); err == nil {
			_ = os.Rename(src, dst)
		}
	}

	// Active file → .1
	if rf.maxBackups > 0 {
		_ = os.Rename(rf.path(), fmt.Sprintf("%s.1", rf.path()))
	} else {
		_ = os.Remove(rf.path())
	}

	return rf.openExisting()
}

// Close closes the underlying file.
func (rf *rotatingFile) Close() error {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	if rf.file == nil {
		return nil
	}
	err := rf.file.Close()
	rf.file = nil
	return err
}

// listBackups is a small helper (used in tests) returning existing rotated file
// names sorted oldest→newest by their numeric suffix.
func (rf *rotatingFile) listBackups() []string {
	entries, err := os.ReadDir(rf.dir)
	if err != nil {
		return nil
	}
	prefix := rf.base + "."
	var names []string
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), prefix) {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// timestampedName is reserved for callers that prefer time-based names; kept for
// completeness and future use.
func timestampedName(base string) string {
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	return fmt.Sprintf("%s-%s%s", stem, time.Now().Format("20060102-150405"), ext)
}
