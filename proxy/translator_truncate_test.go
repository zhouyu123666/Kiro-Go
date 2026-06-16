package proxy

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
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

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestTruncateCurrentMessageDropsOversizedImages verifies that when the current
// message is oversized because of large base64 images (not text Content), the
// truncation path drops the images so the payload fits. The previous behavior
// only shrank Content and left the payload over-limit → upstream HTTP 400.
func TestTruncateCurrentMessageDropsOversizedImages(t *testing.T) {
	// A single image whose base64 bytes alone exceed maxPayloadBytes.
	bigImg := KiroImage{Format: "png"}
	bigImg.Source.Bytes = strings.Repeat("A", maxPayloadBytes+50*1024)

	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "describe this image",
		ModelID: "claude-opus-4.8",
		Origin:  "AI_EDITOR",
		Images:  []KiroImage{bigImg},
	}

	truncateCurrentMessage(payload)

	if got := payloadByteSize(payload); got > maxPayloadBytes {
		t.Fatalf("payload size %d still exceeds limit %d after truncation", got, maxPayloadBytes)
	}
	if len(payload.ConversationState.CurrentMessage.UserInputMessage.Images) != 0 {
		t.Fatalf("expected oversized images to be dropped")
	}
}

// TestTruncateCurrentMessageShrinksOversizedToolResults verifies that an
// oversized current message whose bulk lives in structured tool-result text is
// shrunk below the limit rather than left over-limit.
func TestTruncateCurrentMessageShrinksOversizedToolResults(t *testing.T) {
	bigText := strings.Repeat("x", maxPayloadBytes+50*1024)

	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "Tool results:",
		ModelID: "claude-opus-4.8",
		Origin:  "AI_EDITOR",
		UserInputMessageContext: &UserInputMessageContext{
			ToolResults: []KiroToolResult{
				{
					ToolUseID: "tool-1",
					Status:    "success",
					Content:   []KiroResultContent{{Text: bigText}},
				},
			},
		},
	}

	truncateCurrentMessage(payload)

	if got := payloadByteSize(payload); got > maxPayloadBytes {
		t.Fatalf("payload size %d still exceeds limit %d after truncation", got, maxPayloadBytes)
	}
}

// TestTruncateUTF8RespectsRuneBoundaries verifies that shrinking never splits a
// multi-byte rune, which would otherwise yield invalid UTF-8 that upstream
// rejects as a malformed request.
func TestTruncateUTF8RespectsRuneBoundaries(t *testing.T) {
	// "你好" is two 3-byte runes. Cutting at byte index 4 lands mid-rune.
	s := "你好世界"
	for budget := 0; budget <= len(s); budget++ {
		got := truncateUTF8(s, budget)
		if !utf8.ValidString(got) {
			t.Fatalf("truncateUTF8(%q, %d) = %q is not valid UTF-8", s, budget, got)
		}
		if len(got) > budget {
			t.Fatalf("truncateUTF8(%q, %d) = %q exceeds budget", s, budget, got)
		}
	}
}
