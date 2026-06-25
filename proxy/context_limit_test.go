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

	tiktoken "github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
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

func testGatewayTokenCount(t *testing.T, text string) int {
	t.Helper()
	tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
	encoder, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		t.Fatalf("get cl100k_base encoding: %v", err)
	}
	return len(encoder.EncodeOrdinary(text))
}

func TestClaudeCompactionEstimateUsesKiroGatewayTokenizer(t *testing.T) {
	content := strings.Repeat("context ", 1_000)
	req := &ClaudeRequest{
		Model: "claude-opus-4.8",
		Messages: []ClaudeMessage{
			{Role: "user", Content: content},
		},
	}

	gatewayBaseTokens := 4 + testGatewayTokenCount(t, "user") + testGatewayTokenCount(t, content) + 3
	want := int(float64(gatewayBaseTokens) * claudeTokenCorrectionFactor)
	if got := estimateClaudeCompactionInputTokens(req); got != want {
		t.Fatalf("compaction estimate=%d, want kiro-gateway estimate %d", got, want)
	}
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
	claudeReq := ClaudeRequest{
		Model: "claude-sonnet-4.5",
		Messages: []ClaudeMessage{
			{Role: "user", Content: content},
		},
	}
	body, err := json.Marshal(claudeReq)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleClaudeMessagesInternal(rr, req)

	assertPromptTooLongError(t, rr, estimateClaudeCompactionInputTokens(&claudeReq), "claude-sonnet-4.5")
}

func TestClaudeMessagesOverGatewayCompactionLimitReturnsPromptTooLong(t *testing.T) {
	const model = "claude-sonnet-4.5"
	compactionLimit := modelClientCompactionLimit(model)
	repeatCount := compactionLimit*100/115 + 1_000
	content := strings.Repeat("context ", repeatCount)
	claudeReq := ClaudeRequest{
		Model: model,
		Messages: []ClaudeMessage{
			{Role: "user", Content: content},
		},
	}
	gatewayEstimated := estimateClaudeCompactionInputTokens(&claudeReq)
	if gatewayEstimated <= compactionLimit {
		t.Fatalf("test setup expected gateway estimate above %d, got %d",
			compactionLimit, gatewayEstimated)
	}
	if exceedsKiroInputTokenLimit(gatewayEstimated, model) {
		t.Fatalf("test setup expected gateway estimate below hard safety limit, got %d", gatewayEstimated)
	}

	body, err := json.Marshal(claudeReq)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleClaudeMessagesInternal(rr, req)

	assertPromptTooLongError(t, rr, gatewayEstimated, model)
}

func TestClaudeCountTokensUsesGatewayEstimate(t *testing.T) {
	claudeReq := ClaudeRequest{
		Model: "claude-opus-4.8",
		Messages: []ClaudeMessage{
			{Role: "user", Content: strings.Repeat("context ", 1_000)},
		},
	}
	body, err := json.Marshal(claudeReq)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(string(body)))

	h := &Handler{}
	h.handleCountTokens(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d body=%s", rr.Code, rr.Body.String())
	}
	var got map[string]int
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	want := estimateClaudeCompactionInputTokens(&claudeReq)
	if got["input_tokens"] != want {
		t.Fatalf("count_tokens input_tokens=%d, want gateway estimate %d", got["input_tokens"], want)
	}
}

func TestOpenAIChatRejectsOverKiroInputTokenLimit(t *testing.T) {
	body, err := json.Marshal(OpenAIRequest{
		Model: "claude-sonnet-4.5",
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
		Model: "claude-sonnet-4.5",
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
	want := modelClientCompactionLimit("claude-opus-4.7")
	if tokenLimits["maxInputTokens"] != want {
		t.Fatalf("expected maxInputTokens=%d, got %d", want, tokenLimits["maxInputTokens"])
	}
}

func TestClientCompactionLimitStaysWithinAdvertisedHardLimit(t *testing.T) {
	for _, model := range []string{"claude-sonnet-4.5", "claude-opus-4.7", "qwen3-coder-next"} {
		compactionLimit := modelClientCompactionLimit(model)
		if compactionLimit > modelHardInputTokenLimit(model) {
			t.Fatalf("%s: client limit %d must not exceed hard limit %d", model, compactionLimit, modelHardInputTokenLimit(model))
		}
		if exceedsKiroInputTokenLimit(compactionLimit+1, model) {
			t.Fatalf("%s: client compaction threshold must not be treated as a hard reject", model)
		}
	}
}

func TestEstimatorSafetyFactorDoesNotBlockCompactionHeadroom(t *testing.T) {
	const model = "claude-sonnet-4.5"
	estimated := estimateApproxTokens(strings.Repeat("context ", 185_000))
	if estimated <= maxKiroInputTokens {
		t.Fatalf("test setup expected corrected estimate above hard limit, got %d", estimated)
	}
	if exceedsKiroInputTokenLimit(estimated, model) {
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

func assertPromptTooLongError(t *testing.T, rr *httptest.ResponseRecorder, estimatedInputTokens int, model string) {
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
		"tokens",
		"tokens > " + strconv.Itoa(modelClientCompactionLimit(model)),
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("expected response to contain %q, got %s", want, body)
		}
	}
	if strings.Contains(body, "maximum") {
		t.Fatalf("expected response to omit maximum token text, got %s", body)
	}
	if !strings.Contains(body, strconv.Itoa(estimatedInputTokens)) {
		t.Fatalf("expected response to contain estimated token count %d, got %s", estimatedInputTokens, body)
	}
}
