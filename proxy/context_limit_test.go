package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
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

func clientCompactionInputText() string {
	return strings.Repeat("context ", 185_000)
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
	content := oversizedInputText()
	body, err := json.Marshal(ClaudeRequest{
		Model: "claude-opus-4.7",
		Messages: []ClaudeMessage{
			{Role: "user", Content: content},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleClaudeMessagesInternal(rr, req)

	assertPromptTooLongError(t, rr, estimateApproxTokens(content))
}

func TestClaudeMessagesOverClientCompactionLimitReturnsPromptTooLong(t *testing.T) {
	content := clientCompactionInputText()
	estimated := estimateApproxTokens(content)
	if estimated <= clientKiroInputTokens {
		t.Fatalf("test input should exceed client compaction limit %d, estimated %d", clientKiroInputTokens, estimated)
	}
	if exceedsKiroInputTokenLimit(estimated) {
		t.Fatalf("test input should stay below hard safety limit, estimated %d", estimated)
	}

	body, err := json.Marshal(ClaudeRequest{
		Model: "claude-opus-4.7",
		Messages: []ClaudeMessage{
			{Role: "user", Content: content},
		},
	})
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleClaudeMessagesInternal(rr, req)

	assertPromptTooLongError(t, rr, estimated)
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

func TestBuildModelInfoAdvertisesClientCompactionLimit(t *testing.T) {
	model := buildModelInfo("claude-opus-4.7", "anthropic", true)
	tokenLimits, ok := model["tokenLimits"].(map[string]int)
	if !ok {
		t.Fatalf("expected tokenLimits on model info, got %#v", model["tokenLimits"])
	}
	if tokenLimits["maxInputTokens"] != clientKiroInputTokens {
		t.Fatalf("expected maxInputTokens=%d, got %d", clientKiroInputTokens, tokenLimits["maxInputTokens"])
	}
}

func TestClientCompactionLimitLeavesHeadroomBeforeHardLimit(t *testing.T) {
	if clientKiroInputTokens >= maxKiroInputTokens {
		t.Fatalf("client limit %d must stay below hard limit %d", clientKiroInputTokens, maxKiroInputTokens)
	}
	if exceedsKiroInputTokenLimit(clientKiroInputTokens + 1) {
		t.Fatalf("client compaction headroom must not be treated as a hard reject")
	}
}

func TestEstimatorSafetyFactorDoesNotBlockCompactionHeadroom(t *testing.T) {
	estimated := estimateApproxTokens(strings.Repeat("context ", 185_000))
	if estimated <= maxKiroInputTokens {
		t.Fatalf("test setup expected corrected estimate above hard limit, got %d", estimated)
	}
	if exceedsKiroInputTokenLimit(estimated) {
		t.Fatalf("corrected estimate %d should remain inside hard-limit safety margin", estimated)
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

// A PDF arrives as a small document block but inlinePDFDocuments folds its
// extracted text into the current message content during conversion. Gating on
// the converted payload must count that inlined text, otherwise a near-limit
// conversation plus a text-heavy PDF slips past the local check and trips
// Kiro's "Input is too long." rejection upstream.
func TestKiroPayloadCountsInlinedMessageContent(t *testing.T) {
	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: oversizedInputText(),
	}
	if got := estimateKiroPayloadInputTokens(payload); got <= maxKiroInputTokens {
		t.Fatalf("expected inlined content to exceed %d tokens, got %d", maxKiroInputTokens, got)
	}
}

func TestKiroPayloadMediaUsesFixedCost(t *testing.T) {
	payload := &KiroPayload{}
	cur := &payload.ConversationState.CurrentMessage.UserInputMessage
	cur.Content = "describe these"
	cur.Images = []KiroImage{{Format: "png"}, {Format: "png"}}
	cur.Documents = []KiroDocument{{Format: "pdf"}}
	cur.Images[0].Source.Bytes = largeBase64()
	cur.Images[1].Source.Bytes = largeBase64()
	cur.Documents[0].Source.Bytes = largeBase64()

	want := estimateApproxTokens("describe these") + 2*approxImageInputTokens + approxDocumentInputTokens
	if got := estimateKiroPayloadInputTokens(payload); got != want {
		t.Fatalf("expected media to use fixed cost (%d), got %d", want, got)
	}
}

func TestKiroPayloadCountsHistory(t *testing.T) {
	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{Content: "final turn"}
	payload.ConversationState.History = []KiroHistoryMessage{
		{UserInputMessage: &KiroUserInputMessage{Content: oversizedInputText()}},
		{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "ok"}},
	}
	if got := estimateKiroPayloadInputTokens(payload); got <= maxKiroInputTokens {
		t.Fatalf("expected history to push payload over %d tokens, got %d", maxKiroInputTokens, got)
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

func assertPromptTooLongError(t *testing.T, rr *httptest.ResponseRecorder, estimatedInputTokens int) {
	t.Helper()
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d with body %s", rr.Code, rr.Body.String())
	}
	var parsed struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &parsed); err != nil {
		t.Fatalf("decode response body: %v; body=%s", err, rr.Body.String())
	}
	body := parsed.Error.Message
	for _, want := range []string{
		"Prompt is too long",
		"tokens > 180000 maximum",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected response to contain %q, got %s", want, body)
		}
	}
	if !strings.Contains(body, strconv.Itoa(estimatedInputTokens)) {
		t.Fatalf("expected response to contain estimated token count %d, got %s", estimatedInputTokens, body)
	}
}
