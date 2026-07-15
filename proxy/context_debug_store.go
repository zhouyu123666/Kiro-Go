package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"kiro-go/logger"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

var contextDebugDBPath = filepath.Join("data", "context_debug.db")

var contextDebugWriteMu sync.Mutex

const contextDebugEnabledEnv = "KIRO_CONTEXT_DEBUG"

type contextDebugTask struct {
	ID        string
	CreatedAt int64

	mu           sync.Mutex
	nextSequence int
}

type contextDebugRow struct {
	Role        string
	ContentType string
	ContentData string
}

type claudeContextDebugInitial struct {
	RawBody                []byte
	RawRequest             *ClaudeRequest
	EffectiveRequest       *ClaudeRequest
	Payload                *KiroPayload
	ClientModel            string
	ActualModel            string
	Thinking               bool
	Stream                 bool
	KiroPayloadInputTokens int
	ClientInputTokens      int
	BillingInputTokens     int
	UsageReportWindow      int
	CurrentTools           int
	CurrentToolResults     int
	CacheProfile           *promptCacheProfile
	PromptCacheProfileNil  bool
	PromptCacheBreakpoints int
	PromptCacheLastTokens  int
}

type claudeContextDebugFinal struct {
	Mode                   string                 `json:"mode"`
	Model                  string                 `json:"model"`
	AccountID              string                 `json:"account_id,omitempty"`
	ContextUsagePercentage float64                `json:"context_usage_percentage"`
	EstimatedInputTokens   int                    `json:"estimated_input_tokens"`
	BillingInputTokens     int                    `json:"billing_input_tokens"`
	UpstreamInputTokens    int                    `json:"upstream_input_tokens"`
	PublicInputTokens      int                    `json:"public_input_tokens"`
	VisibleInputTokens     int                    `json:"visible_input_tokens"`
	BillableInputTokens    int                    `json:"billable_input_tokens"`
	OutputTokens           int                    `json:"output_tokens"`
	Credits                float64                `json:"credits"`
	CacheUsage             promptCacheUsage       `json:"cache_usage"`
	PublicCacheUsage       promptCacheUsage       `json:"public_cache_usage"`
	ToolUses               []KiroToolUse          `json:"tool_uses,omitempty"`
	Usage                  map[string]interface{} `json:"usage,omitempty"`
	Error                  string                 `json:"error,omitempty"`
	Extra                  map[string]interface{} `json:"extra,omitempty"`
	CreatedAtUnix          int64                  `json:"created_at_unix"`
	CreatedAtUnixMilli     int64                  `json:"created_at_unix_milli"`
	GeneratedAtUnixMilli   int64                  `json:"generated_at_unix_milli"`
	SequenceNote           string                 `json:"sequence_note,omitempty"`
}

func newContextDebugTask() *contextDebugTask {
	if !contextDebugEnabled() {
		return nil
	}
	now := time.Now()
	return &contextDebugTask{
		ID:           fmt.Sprintf("%d-%s", now.UnixMilli(), strings.ReplaceAll(uuid.NewString()[:8], "-", "")),
		CreatedAt:    now.Unix(),
		nextSequence: 1,
	}
}

func contextDebugEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(contextDebugEnabledEnv))) {
	case "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

func (t *contextDebugTask) Append(rows ...contextDebugRow) {
	if t == nil || len(rows) == 0 {
		return
	}

	t.mu.Lock()
	startSequence := t.nextSequence
	t.nextSequence += len(rows)
	t.mu.Unlock()

	if err := writeContextDebugRows(t.ID, t.CreatedAt, startSequence, rows); err != nil {
		logger.Warnf("[ContextDebug] write failed task_id=%s rows=%d err=%v", t.ID, len(rows), err)
	}
}

func appendClaudeContextDebugInitialRows(task *contextDebugTask, snapshot claudeContextDebugInitial) {
	if task == nil {
		return
	}
	task.Append(buildClaudeInitialContextDebugRows(snapshot)...)
}

func appendClaudeContextDebugFinalRows(task *contextDebugTask, final claudeContextDebugFinal, assistantText, thinkingText string) {
	if task == nil {
		return
	}
	task.Append(buildClaudeFinalContextDebugRows(final, assistantText, thinkingText)...)
}

func writeContextDebugRows(taskID string, createdAt int64, startSequence int, rows []contextDebugRow) error {
	sqlitePath, err := exec.LookPath("sqlite3")
	if err != nil {
		return fmt.Errorf("sqlite3 not found in PATH: %w", err)
	}
	dbPath := filepath.Clean(contextDebugDBPath)
	if err := os.MkdirAll(filepath.Dir(dbPath), 0o755); err != nil {
		return fmt.Errorf("create context debug dir: %w", err)
	}

	var b strings.Builder
	b.WriteString("PRAGMA busy_timeout=5000;\n")
	b.WriteString("BEGIN IMMEDIATE;\n")
	b.WriteString(`CREATE TABLE IF NOT EXISTS context_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at INTEGER NOT NULL,
  task_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  role TEXT NOT NULL,
  content_type TEXT NOT NULL,
  content_data TEXT NOT NULL,
  sequence_order INTEGER NOT NULL,
  content_hash TEXT
);
`)
	b.WriteString("CREATE INDEX IF NOT EXISTS idx_context_messages_task_id ON context_messages(task_id, sequence_order);\n")
	b.WriteString("CREATE INDEX IF NOT EXISTS idx_context_messages_created_at ON context_messages(created_at);\n")
	b.WriteString("CREATE UNIQUE INDEX IF NOT EXISTS idx_context_messages_task_sequence ON context_messages(task_id, sequence_order);\n")

	nowMs := time.Now().UnixMilli()
	for i, row := range rows {
		role := normalizeContextDebugRole(row.Role)
		contentType := normalizeContextDebugContentType(row.ContentType)
		contentData := row.ContentData
		if contentData == "" {
			contentData = "{}"
		}
		hash := sha256.Sum256([]byte(contentData))
		sequence := startSequence + i
		b.WriteString("INSERT OR REPLACE INTO context_messages (created_at, task_id, ts, role, content_type, content_data, sequence_order, content_hash) VALUES (")
		b.WriteString(strconv.FormatInt(createdAt, 10))
		b.WriteString(", ")
		b.WriteString(sqlQuote(taskID))
		b.WriteString(", ")
		b.WriteString(strconv.FormatInt(nowMs, 10))
		b.WriteString(", ")
		b.WriteString(sqlQuote(role))
		b.WriteString(", ")
		b.WriteString(sqlQuote(contentType))
		b.WriteString(", ")
		b.WriteString(sqlQuote(contentData))
		b.WriteString(", ")
		b.WriteString(strconv.Itoa(sequence))
		b.WriteString(", ")
		b.WriteString(sqlQuote(hex.EncodeToString(hash[:])))
		b.WriteString(");\n")
	}
	b.WriteString("COMMIT;\n")

	contextDebugWriteMu.Lock()
	defer contextDebugWriteMu.Unlock()

	cmd := exec.Command(sqlitePath, dbPath)
	cmd.Stdin = strings.NewReader(b.String())
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("sqlite3 insert failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func sqlQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "''") + "'"
}

func normalizeContextDebugRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "user", "assistant":
		return strings.ToLower(strings.TrimSpace(role))
	default:
		return "system"
	}
}

func normalizeContextDebugContentType(contentType string) string {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	if contentType == "" {
		return "text"
	}
	return contentType
}

func buildClaudeInitialContextDebugRows(snapshot claudeContextDebugInitial) []contextDebugRow {
	messageCount := 0
	if snapshot.EffectiveRequest != nil {
		messageCount = len(snapshot.EffectiveRequest.Messages)
	}
	rows := make([]contextDebugRow, 0, 16+messageCount)
	cacheProfile := map[string]interface{}{
		"nil":             snapshot.CacheProfile == nil,
		"breakpoints":     snapshot.PromptCacheBreakpoints,
		"last_tokens":     snapshot.PromptCacheLastTokens,
		"total_input":     0,
		"model":           "",
		"breakpoint_list": []promptCacheBreakpoint(nil),
	}
	if snapshot.CacheProfile != nil {
		cacheProfile["total_input"] = snapshot.CacheProfile.TotalInputTokens
		cacheProfile["model"] = snapshot.CacheProfile.Model
		cacheProfile["breakpoint_list"] = snapshot.CacheProfile.Breakpoints
	}

	rows = append(rows, debugJSONTextRow("claude_context_debug_initial", map[string]interface{}{
		"client_model":               snapshot.ClientModel,
		"actual_model":               snapshot.ActualModel,
		"thinking":                   snapshot.Thinking,
		"stream":                     snapshot.Stream,
		"kiro_payload_input_tokens":  snapshot.KiroPayloadInputTokens,
		"client_input_tokens":        snapshot.ClientInputTokens,
		"billing_input_tokens":       snapshot.BillingInputTokens,
		"usage_report_window":        snapshot.UsageReportWindow,
		"current_tools":              snapshot.CurrentTools,
		"current_tool_results":       snapshot.CurrentToolResults,
		"prompt_cache_profile":       cacheProfile,
		"raw_body_bytes":             len(snapshot.RawBody),
		"generated_at_unix_milli":    time.Now().UnixMilli(),
		"debug_database":             contextDebugDBPath,
		"debug_database_table":       "context_messages",
		"debug_database_row_version": 1,
	}))
	rows = append(rows, debugJSONTextRow("claude_raw_request", json.RawMessage(snapshot.RawBody)))
	rows = append(rows, debugJSONTextRow("claude_effective_request", snapshot.EffectiveRequest))
	rows = append(rows, debugJSONTextRow("kiro_payload", snapshot.Payload))
	rows = append(rows, buildClaudeRequestContextRows(snapshot.EffectiveRequest, "claude_effective_request")...)
	rows = append(rows, buildKiroPayloadContextRows(snapshot.Payload)...)
	return rows
}

func buildClaudeFinalContextDebugRows(final claudeContextDebugFinal, assistantText, thinkingText string) []contextDebugRow {
	final.CreatedAtUnix = time.Now().Unix()
	final.CreatedAtUnixMilli = time.Now().UnixMilli()
	final.GeneratedAtUnixMilli = time.Now().UnixMilli()

	rows := []contextDebugRow{debugJSONTextRow("claude_context_debug_final", final)}
	if thinkingText != "" {
		rows = append(rows, contextDebugTextRow("assistant", "thinking", map[string]interface{}{"text": thinkingText}))
	}
	if assistantText != "" {
		rows = append(rows, contextDebugTextRow("assistant", "text", map[string]interface{}{"text": assistantText}))
	}
	for _, toolUse := range final.ToolUses {
		rows = append(rows, contextDebugRow{
			Role:        "assistant",
			ContentType: "tool_use",
			ContentData: mustContextDebugJSON(map[string]interface{}{
				"type":  "tool_use",
				"id":    toolUse.ToolUseID,
				"name":  toolUse.Name,
				"input": toolUse.Input,
			}),
		})
	}
	return rows
}

func buildClaudeRequestContextRows(req *ClaudeRequest, source string) []contextDebugRow {
	if req == nil {
		return nil
	}
	rows := make([]contextDebugRow, 0, len(req.Messages)+2)
	rows = append(rows, buildClaudeValueContextRows("system", req.System, map[string]interface{}{
		"source": source,
		"field":  "system",
	})...)
	if len(req.Tools) > 0 {
		rows = append(rows, contextDebugRow{
			Role:        "system",
			ContentType: "tool_spec",
			ContentData: mustContextDebugJSON(map[string]interface{}{
				"source": source,
				"field":  "tools",
				"tools":  req.Tools,
			}),
		})
	}
	if req.ToolChoice != nil {
		rows = append(rows, debugJSONTextRow("claude_tool_choice", map[string]interface{}{
			"source":      source,
			"tool_choice": req.ToolChoice,
		}))
	}
	for i, msg := range req.Messages {
		role := normalizeContextDebugRole(msg.Role)
		rows = append(rows, buildClaudeValueContextRows(role, msg.Content, map[string]interface{}{
			"source":        source,
			"message_index": i,
			"message_role":  msg.Role,
		})...)
	}
	return rows
}

func buildClaudeValueContextRows(role string, value interface{}, meta map[string]interface{}) []contextDebugRow {
	if value == nil {
		return nil
	}
	switch v := value.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []contextDebugRow{contextDebugTextRow(role, "text", mergeContextDebugMap(meta, map[string]interface{}{"text": v}))}
	case []interface{}:
		rows := make([]contextDebugRow, 0, len(v))
		for i, item := range v {
			rows = append(rows, buildClaudeValueContextRows(role, item, mergeContextDebugMap(meta, map[string]interface{}{"content_index": i}))...)
		}
		return rows
	case map[string]interface{}:
		contentType := contextDebugContentTypeForClaudeMap(v)
		rowRole := role
		if contentType == "tool_use" {
			rowRole = "assistant"
		} else if contentType == "tool_result" {
			rowRole = "user"
		}
		return []contextDebugRow{{
			Role:        rowRole,
			ContentType: contentType,
			ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, v)),
		}}
	default:
		return []contextDebugRow{debugJSONTextRow("claude_context_value", map[string]interface{}{
			"source": meta["source"],
			"value":  v,
		})}
	}
}

func buildKiroPayloadContextRows(payload *KiroPayload) []contextDebugRow {
	if payload == nil {
		return nil
	}
	rows := make([]contextDebugRow, 0, len(payload.ConversationState.History)+4)
	for i, h := range payload.ConversationState.History {
		if h.UserInputMessage != nil {
			rows = append(rows, buildKiroUserInputContextRows("kiro_history", i, h.UserInputMessage)...)
		}
		if h.AssistantResponseMessage != nil {
			rows = append(rows, buildKiroAssistantContextRows("kiro_history", i, h.AssistantResponseMessage)...)
		}
	}
	rows = append(rows, buildKiroUserInputContextRows("kiro_current_message", -1, &payload.ConversationState.CurrentMessage.UserInputMessage)...)
	return rows
}

func buildKiroUserInputContextRows(source string, historyIndex int, msg *KiroUserInputMessage) []contextDebugRow {
	if msg == nil {
		return nil
	}
	meta := map[string]interface{}{
		"source":        source,
		"history_index": historyIndex,
		"model_id":      msg.ModelID,
		"origin":        msg.Origin,
	}
	rows := make([]contextDebugRow, 0, 1+len(msg.Images)+len(msg.Documents))
	if msg.Content != "" {
		rows = append(rows, contextDebugTextRow("user", "text", mergeContextDebugMap(meta, map[string]interface{}{"text": msg.Content})))
	}
	for i, image := range msg.Images {
		rows = append(rows, contextDebugRow{
			Role:        "user",
			ContentType: "image",
			ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, map[string]interface{}{
				"image_index": i,
				"format":      image.Format,
				"bytes_len":   len(image.Source.Bytes),
			})),
		})
	}
	for i, document := range msg.Documents {
		rows = append(rows, contextDebugRow{
			Role:        "user",
			ContentType: "document",
			ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, map[string]interface{}{
				"document_index": i,
				"name":           document.Name,
				"format":         document.Format,
				"bytes_len":      len(document.Source.Bytes),
			})),
		})
	}
	if msg.UserInputMessageContext != nil {
		for i, tool := range msg.UserInputMessageContext.Tools {
			rows = append(rows, contextDebugRow{
				Role:        "system",
				ContentType: "tool_spec",
				ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, map[string]interface{}{
					"tool_index": i,
					"tool":       tool,
				})),
			})
		}
		for i, result := range msg.UserInputMessageContext.ToolResults {
			rows = append(rows, contextDebugRow{
				Role:        "user",
				ContentType: "tool_result",
				ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, map[string]interface{}{
					"tool_result_index": i,
					"tool_result":       result,
				})),
			})
		}
	}
	return rows
}

func buildKiroAssistantContextRows(source string, historyIndex int, msg *KiroAssistantResponseMessage) []contextDebugRow {
	if msg == nil {
		return nil
	}
	meta := map[string]interface{}{
		"source":        source,
		"history_index": historyIndex,
	}
	rows := make([]contextDebugRow, 0, 1+len(msg.ToolUses))
	if msg.Content != "" {
		rows = append(rows, contextDebugTextRow("assistant", "text", mergeContextDebugMap(meta, map[string]interface{}{"text": msg.Content})))
	}
	for i, toolUse := range msg.ToolUses {
		rows = append(rows, contextDebugRow{
			Role:        "assistant",
			ContentType: "tool_use",
			ContentData: mustContextDebugJSON(mergeContextDebugMap(meta, map[string]interface{}{
				"tool_use_index": i,
				"tool_use":       toolUse,
			})),
		})
	}
	return rows
}

func contextDebugContentTypeForClaudeMap(value map[string]interface{}) string {
	typeName, _ := value["type"].(string)
	switch strings.ToLower(strings.TrimSpace(typeName)) {
	case "text", "":
		return "text"
	case "thinking":
		return "thinking"
	case "tool_use":
		return "tool_use"
	case "tool_result":
		return "tool_result"
	case "image", "image_url":
		return "image"
	default:
		return strings.ToLower(strings.TrimSpace(typeName))
	}
}

func debugJSONTextRow(kind string, value interface{}) contextDebugRow {
	return contextDebugTextRow("system", "text", map[string]interface{}{
		"debug_kind": kind,
		"text":       prettyContextDebugJSON(value),
	})
}

func contextDebugTextRow(role, contentType string, value map[string]interface{}) contextDebugRow {
	return contextDebugRow{
		Role:        role,
		ContentType: contentType,
		ContentData: mustContextDebugJSON(value),
	}
}

func mergeContextDebugMap(base map[string]interface{}, extra map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{}, len(base)+len(extra))
	for k, v := range base {
		merged[k] = v
	}
	for k, v := range extra {
		merged[k] = v
	}
	return merged
}

func prettyContextDebugJSON(value interface{}) string {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Sprintf("<json marshal failed: %v>", err)
	}
	return string(b)
}

func mustContextDebugJSON(value interface{}) string {
	b, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf(`{"error":"json marshal failed","message":%q}`, err.Error())
	}
	return string(b)
}
