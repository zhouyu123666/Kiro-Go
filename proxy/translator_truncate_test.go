package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestClaudeToKiroTruncatesOversizedHistory builds a conversation whose history
// far exceeds the upstream input limit and verifies the converted payload is
// trimmed below maxPayloadBytes, that a truncation placeholder is inserted, and
// that the current message is preserved.
func TestClaudeToKiroTruncatesOversizedHistory(t *testing.T) {
	// ~2KB chunk repeated across many turns to blow past the byte limit.
	big := strings.Repeat("lorem ipsum dolor sit amet ", 80) // ~2.1KB

	msgs := []ClaudeMessage{
		{Role: "user", Content: "start the long task"},
	}
	for i := 0; i < 800; i++ {
		msgs = append(msgs,
			ClaudeMessage{Role: "assistant", Content: "step result: " + big},
			ClaudeMessage{Role: "user", Content: "next: " + big},
		)
	}
	msgs = append(msgs, ClaudeMessage{Role: "user", Content: "FINAL: summarize everything above"})

	req := &ClaudeRequest{
		Model:    "claude-opus-4.8",
		System:   "You are a helpful assistant.",
		Messages: msgs,
	}

	payload := ClaudeToKiro(req, false)

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal failed: %v", err)
	}
	if len(raw) > maxPayloadBytes {
		t.Fatalf("payload size %d exceeds limit %d after truncation", len(raw), maxPayloadBytes)
	}

	// The current message must be preserved.
	cur := payload.ConversationState.CurrentMessage.UserInputMessage
	if !strings.Contains(cur.Content, "FINAL: summarize everything above") {
		t.Fatalf("current message lost after truncation, got %q", cur.Content[:min(80, len(cur.Content))])
	}

	// A truncation placeholder must be present in history.
	foundPlaceholder := false
	for _, h := range payload.ConversationState.History {
		if h.UserInputMessage != nil && strings.Contains(h.UserInputMessage.Content, "truncated to fit") {
			foundPlaceholder = true
			break
		}
	}
	if !foundPlaceholder {
		t.Fatalf("expected a truncation placeholder in history")
	}

	// System priming should still be at the front.
	if len(payload.ConversationState.History) < 2 {
		t.Fatalf("expected priming retained, history too short")
	}
	primingUser := payload.ConversationState.History[0].UserInputMessage
	if primingUser == nil || !strings.Contains(primingUser.Content, "helpful assistant") {
		t.Fatalf("expected system priming retained at front")
	}
}

// TestClaudeToKiroSmallPayloadNotTruncated ensures normal-sized conversations
// are left untouched (no placeholder inserted).
func TestClaudeToKiroSmallPayloadNotTruncated(t *testing.T) {
	req := &ClaudeRequest{
		Model:  "claude-opus-4.8",
		System: "You are helpful.",
		Messages: []ClaudeMessage{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi"},
			{Role: "user", Content: "how are you?"},
		},
	}
	payload := ClaudeToKiro(req, false)
	for _, h := range payload.ConversationState.History {
		if h.UserInputMessage != nil && strings.Contains(h.UserInputMessage.Content, "truncated to fit") {
			t.Fatalf("small payload should not be truncated")
		}
	}
}

// TestClaudeToKiroTruncatesOversizedCurrentToolResult covers the case where the
// active tool turn's result is itself larger than the upstream input limit (e.g.
// reading a huge file). The structured tool result must be shrunk below
// maxPayloadBytes while preserving the toolUseId pairing so the request stays
// well-formed.
func TestClaudeToKiroTruncatesOversizedCurrentToolResult(t *testing.T) {
	huge := strings.Repeat("X", maxPayloadBytes+200*1024) // well over the limit

	req := &ClaudeRequest{
		Model: "claude-opus-4.8",
		Tools: []ClaudeTool{{Name: "read_file", Description: "read", InputSchema: map[string]interface{}{"type": "object"}}},
		Messages: []ClaudeMessage{
			{Role: "user", Content: "read the big file"},
			{Role: "assistant", Content: []interface{}{
				map[string]interface{}{"type": "tool_use", "id": "tr1", "name": "read_file", "input": map[string]interface{}{"path": "big.txt"}},
			}},
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "tool_result", "tool_use_id": "tr1", "content": huge},
			}},
		},
	}

	payload := ClaudeToKiro(req, false)

	if sz := payloadByteSize(payload); sz > maxPayloadBytes {
		t.Fatalf("payload size %d exceeds limit %d after truncating tool result", sz, maxPayloadBytes)
	}

	// The active tool turn must remain structured and paired (tr1 on both sides).
	cur := payload.ConversationState.CurrentMessage.UserInputMessage
	if cur.UserInputMessageContext == nil || len(cur.UserInputMessageContext.ToolResults) != 1 {
		t.Fatalf("expected the structured tool result to be preserved")
	}
	if cur.UserInputMessageContext.ToolResults[0].ToolUseID != "tr1" {
		t.Fatalf("expected tool result to still answer tr1, got %q", cur.UserInputMessageContext.ToolResults[0].ToolUseID)
	}
	hist := payload.ConversationState.History
	last := hist[len(hist)-1].AssistantResponseMessage
	if last == nil || len(last.ToolUses) != 1 || last.ToolUses[0].ToolUseID != "tr1" {
		t.Fatalf("expected the active assistant tool use tr1 to remain structured")
	}
}

// TestClaudeToKiroDropsOversizedCurrentImage covers the case where an attached
// image alone exceeds the upstream input limit. The image must be dropped (it
// cannot be shrunk losslessly) and a note left so the model knows.
func TestClaudeToKiroDropsOversizedCurrentImage(t *testing.T) {
	// A valid base64 blob larger than the payload limit.
	hugeB64 := strings.Repeat("QUFB", (maxPayloadBytes+200*1024)/4) // base64 of "AAA"*

	req := &ClaudeRequest{
		Model: "claude-opus-4.8",
		Messages: []ClaudeMessage{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "what is in this screenshot?"},
				map[string]interface{}{"type": "image", "source": map[string]interface{}{
					"type": "base64", "media_type": "image/png", "data": hugeB64,
				}},
			}},
		},
	}

	payload := ClaudeToKiro(req, false)

	if sz := payloadByteSize(payload); sz > maxPayloadBytes {
		t.Fatalf("payload size %d exceeds limit %d after dropping image", sz, maxPayloadBytes)
	}
	cur := payload.ConversationState.CurrentMessage.UserInputMessage
	if len(cur.Images) != 0 {
		t.Fatalf("expected oversized image to be dropped, still have %d", len(cur.Images))
	}
	if !strings.Contains(cur.Content, "image") {
		t.Fatalf("expected current content to retain the user's question text, got %q", cur.Content)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
