package proxy

import (
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func withTempContextDebugDB(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("sqlite3"); err != nil {
		t.Skipf("sqlite3 not available: %v", err)
	}
	t.Setenv(contextDebugEnabledEnv, "1")
	oldPath := contextDebugDBPath
	dbPath := filepath.Join(t.TempDir(), "context_debug.db")
	contextDebugDBPath = dbPath
	t.Cleanup(func() { contextDebugDBPath = oldPath })
	return dbPath
}

func sqliteScalar(t *testing.T, dbPath, query string) string {
	t.Helper()
	out, err := exec.Command("sqlite3", dbPath, query).CombinedOutput()
	if err != nil {
		t.Fatalf("sqlite query failed: %v: %s", err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out))
}

func sqliteCount(t *testing.T, dbPath, where string) int {
	t.Helper()
	query := "SELECT count(*) FROM context_messages"
	if strings.TrimSpace(where) != "" {
		query += " WHERE " + where
	}
	raw := sqliteScalar(t, dbPath, query+";")
	n, err := strconv.Atoi(raw)
	if err != nil {
		t.Fatalf("parse sqlite count %q: %v", raw, err)
	}
	return n
}

func TestContextDebugStoreCreatesSchemaAndStableHashes(t *testing.T) {
	dbPath := withTempContextDebugDB(t)
	task := newContextDebugTask()
	task.ID = "task-test"
	task.CreatedAt = 123

	task.Append(
		contextDebugTextRow("user", "text", map[string]interface{}{"text": "same"}),
		contextDebugTextRow("user", "text", map[string]interface{}{"text": "same"}),
	)
	task.Append(contextDebugTextRow("assistant", "text", map[string]interface{}{"text": "done"}))

	if got := sqliteCount(t, dbPath, "task_id = 'task-test'"); got != 3 {
		t.Fatalf("expected 3 debug rows, got %d", got)
	}
	if got := sqliteScalar(t, dbPath, "SELECT group_concat(sequence_order, ',') FROM context_messages WHERE task_id = 'task-test' ORDER BY sequence_order;"); got != "1,2,3" {
		t.Fatalf("expected contiguous sequence order, got %q", got)
	}
	if got := sqliteScalar(t, dbPath, "SELECT count(*) FROM sqlite_master WHERE type = 'index' AND name IN ('idx_context_messages_task_id', 'idx_context_messages_created_at', 'idx_context_messages_task_sequence');"); got != "3" {
		t.Fatalf("expected context debug indexes to be created, got %q", got)
	}
	if got := sqliteScalar(t, dbPath, "SELECT count(DISTINCT content_hash) FROM context_messages WHERE task_id = 'task-test' AND sequence_order IN (1, 2);"); got != "1" {
		t.Fatalf("expected identical content_data rows to share content_hash, got %q", got)
	}
}

func TestContextDebugTaskDisabledByDefault(t *testing.T) {
	t.Setenv(contextDebugEnabledEnv, "")
	if task := newContextDebugTask(); task != nil {
		t.Fatalf("expected context debug task to be nil when %s is not enabled", contextDebugEnabledEnv)
	}
}

func TestBuildKiroPayloadContextRowsExpandsCurrentAndHistory(t *testing.T) {
	payload := &KiroPayload{}
	payload.ConversationState.History = []KiroHistoryMessage{
		{UserInputMessage: &KiroUserInputMessage{Content: "history user", ModelID: "claude-sonnet-4.5", Origin: "AI_EDITOR"}},
		{AssistantResponseMessage: &KiroAssistantResponseMessage{
			Content: "history assistant",
			ToolUses: []KiroToolUse{{
				ToolUseID: "toolu_1",
				Name:      "read_file",
				Input:     map[string]interface{}{"path": "README.md"},
			}},
		}},
	}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "current user",
		ModelID: "claude-sonnet-4.5",
		Origin:  "AI_EDITOR",
		UserInputMessageContext: &UserInputMessageContext{
			Tools: []KiroToolWrapper{{}},
			ToolResults: []KiroToolResult{{
				ToolUseID: "toolu_1",
				Content:   []KiroResultContent{{Text: "result text"}},
				Status:    "success",
			}},
		},
	}
	payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.Tools[0].ToolSpecification.Name = "read_file"

	rows := buildKiroPayloadContextRows(payload)
	joined := strings.Join(func() []string {
		out := make([]string, 0, len(rows))
		for _, row := range rows {
			out = append(out, row.ContentType+":"+row.ContentData)
		}
		return out
	}(), "\n")

	for _, want := range []string{
		`"source":"kiro_history"`,
		`"source":"kiro_current_message"`,
		"history user",
		"history assistant",
		"tool_use",
		"tool_spec",
		"tool_result",
		"current user",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected expanded Kiro context rows to contain %q, got:\n%s", want, joined)
		}
	}
}
