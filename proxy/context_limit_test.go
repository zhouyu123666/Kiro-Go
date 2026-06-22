package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"kiro-go/config"
)

func TestMain(m *testing.M) {
	if err := config.Init(filepath.Join(os.TempDir(), "kiro-context-limit-test-config.json")); err != nil {
		panic("config.Init: " + err.Error())
	}
	os.Exit(m.Run())
}

func oversizedInputText() string {
	return strings.Repeat("context ", maxKiroInputTokens+1)
}

func TestClaudeToKiroDoesNotAutoTruncateOversizedHistory(t *testing.T) {
	big := strings.Repeat("context ", 30_000)

	msgs := []ClaudeMessage{{Role: "user", Content: "start"}}
	for i := 0; i < 8; i++ {
		msgs = append(msgs,
			ClaudeMessage{Role: "assistant", Content: "assistant history " + big},
			ClaudeMessage{Role: "user", Content: "user history " + big},
		)
	}
	msgs = append(msgs, ClaudeMessage{Role: "user", Content: "FINAL: keep this exact current turn"})

	req := &ClaudeRequest{
		Model:    "claude-opus-4.7",
		System:   "You are helpful.",
		Messages: msgs,
	}
	if got := estimateClaudeRequestInputTokens(req); got <= maxKiroInputTokens {
		t.Fatalf("test request should exceed input limit, estimated %d", got)
	}

	payload := ClaudeToKiro(req, false)

	cur := payload.ConversationState.CurrentMessage.UserInputMessage
	if !strings.Contains(cur.Content, "FINAL: keep this exact current turn") {
		t.Fatalf("current message was changed during conversion: %q", cur.Content)
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	if strings.Contains(string(raw), "truncated to fit") {
		t.Fatalf("converter inserted a truncation placeholder")
	}
	if !strings.Contains(string(raw), "assistant history") {
		t.Fatalf("converter dropped older history")
	}
}

func TestClaudeMessagesRejectsOverKiroInputTokenLimit(t *testing.T) {
	body, err := json.Marshal(ClaudeRequest{
		Model: "claude-opus-4.7",
		Messages: []ClaudeMessage{
			{Role: "user", Content: oversizedInputText()},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleClaudeMessagesInternal(rr, req)

	assertContextLimitError(t, rr)
}

func TestOpenAIChatRejectsOverKiroInputTokenLimit(t *testing.T) {
	body, err := json.Marshal(OpenAIRequest{
		Model: "claude-opus-4.7",
		Messages: []OpenAIMessage{
			{Role: "user", Content: oversizedInputText()},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleOpenAIChat(rr, req)

	assertContextLimitError(t, rr)
}

func TestResponsesRejectsOverKiroInputTokenLimit(t *testing.T) {
	body, err := json.Marshal(ResponsesRequest{
		Model: "claude-opus-4.7",
		Input: json.RawMessage(`[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"` + oversizedInputText() + `"}]}
		]`),
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleOpenAIResponses(rr, req)

	assertContextLimitError(t, rr)
}

func TestBuildModelInfoAdvertisesKiroInputLimit(t *testing.T) {
	model := buildModelInfo("claude-opus-4.7", "anthropic", true)
	tokenLimits, ok := model["tokenLimits"].(map[string]int)
	if !ok {
		t.Fatalf("expected tokenLimits on model info, got %#v", model["tokenLimits"])
	}
	if tokenLimits["maxInputTokens"] != maxKiroInputTokens {
		t.Fatalf("expected maxInputTokens=%d, got %d", maxKiroInputTokens, tokenLimits["maxInputTokens"])
	}
}

func TestTokenEstimateDoesNotUndercountShortRepeatedWords(t *testing.T) {
	text := strings.Repeat("a ", maxKiroInputTokens+1)
	got := estimateApproxTokens(text)
	if got <= maxKiroInputTokens {
		t.Fatalf("expected short repeated words to exceed %d tokens, got %d", maxKiroInputTokens, got)
	}
}

// largeBase64 returns a base64-like string whose raw byte length far exceeds
// the input token limit, simulating an inlined image/document payload.
func largeBase64() string {
	return strings.Repeat("A", maxKiroInputTokens*8)
}

func TestClaudeImageBlockDoesNotTripInputLimit(t *testing.T) {
	req := &ClaudeRequest{
		Model: "claude-opus-4.7",
		Messages: []ClaudeMessage{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "describe this"},
				map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": "image/png",
						"data":       largeBase64(),
					},
				},
			}},
		},
	}
	if got := estimateClaudeRequestInputTokens(req); got > maxKiroInputTokens {
		t.Fatalf("image block must not be estimated by base64 length, got %d tokens", got)
	}
}

func TestClaudeDocumentBlockDoesNotTripInputLimit(t *testing.T) {
	req := &ClaudeRequest{
		Model: "claude-opus-4.7",
		Messages: []ClaudeMessage{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type": "document",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": "application/pdf",
						"data":       largeBase64(),
					},
				},
			}},
		},
	}
	if got := estimateClaudeRequestInputTokens(req); got > maxKiroInputTokens {
		t.Fatalf("document block must not be estimated by base64 length, got %d tokens", got)
	}
}

func TestOpenAIImageContentDoesNotTripInputLimit(t *testing.T) {
	req := &OpenAIRequest{
		Model: "claude-opus-4.7",
		Messages: []OpenAIMessage{
			{Role: "user", Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "describe this"},
				map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": "data:image/png;base64," + largeBase64(),
					},
				},
			}},
		},
	}
	if got := estimateOpenAIRequestInputTokens(req); got > maxKiroInputTokens {
		t.Fatalf("image_url content must not be estimated by base64 length, got %d tokens", got)
	}
}

func assertContextLimitError(t *testing.T, rr *httptest.ResponseRecorder) {
	t.Helper()
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d with body %s", rr.Code, rr.Body.String())
	}
	body := rr.Body.String()
	for _, want := range []string{
		"Model context limit reached",
		"Estimated input tokens",
		"limit: 200000",
		"silent context truncation",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected response to contain %q, got %s", want, body)
		}
	}
}
