package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"kiro-go/config"
	accountpool "kiro-go/pool"
)

// captured records one send() call from the parser as an (text, state) pair.
type captured struct {
	text  string
	state int
}

// collectFeed drives a fresh parser through the given chunks (last one flushed)
// and returns every send() call in order.
func collectFeed(chunks []string) []captured {
	var out []captured
	p := &thinkingTagParser{allowTag: func() bool { return true }}
	send := func(text string, state int) { out = append(out, captured{text, state}) }
	for i, c := range chunks {
		p.feed(c, i == len(chunks)-1, send)
	}
	return out
}

// answerText concatenates every state-0 (answer) emission.
func answerText(events []captured) string {
	var b strings.Builder
	for _, e := range events {
		if e.state == 0 {
			b.WriteString(e.text)
		}
	}
	return b.String()
}

// thinkingText concatenates every thinking emission (states 1/2/3).
func thinkingText(events []captured) string {
	var b strings.Builder
	for _, e := range events {
		if e.state == 1 || e.state == 2 || e.state == 3 {
			b.WriteString(e.text)
		}
	}
	return b.String()
}

// TestThinkingTagParserLiteralTagMidAnswerStaysAnswer is the core "断片儿"
// regression: once the answer has started, a literal <thinking> tag appearing
// later in the stream is answer content, never a delimiter. Previously the tag
// scanner flipped into thinking mode on that tag and reclassified the entire
// remainder of the answer as reasoning, truncating the visible reply.
func TestThinkingTagParserLiteralTagMidAnswerStaysAnswer(t *testing.T) {
	open := "<" + "thinking" + ">"
	// Reproduce the exact corruption shape: answer text, split so the literal
	// tag lands in a later chunk, and never closes.
	chunks := []string{
		"这是答案的前半部分，内容含字面 `",
		open,
		"` 标签，后半部分应当继续作为答案而不是思考。",
	}
	events := collectFeed(chunks)

	if got := thinkingText(events); got != "" {
		t.Fatalf("literal mid-answer tag must not produce thinking output, got %q", got)
	}

	full := strings.Join(chunks, "")
	if got := answerText(events); got != full {
		t.Fatalf("answer must be emitted verbatim including the literal tag\n got:  %q\n want: %q", got, full)
	}
}

// TestThinkingTagParserLeadingBlockExtracted confirms a genuine leading block —
// first non-whitespace content, properly closed — is emitted as thinking and the
// trailing answer as answer text.
func TestThinkingTagParserLeadingBlockExtracted(t *testing.T) {
	open := "<" + "thinking" + ">"
	closeTag := "</" + "thinking" + ">"
	chunks := []string{"  " + open + "weighing options", closeTag + "the answer"}
	events := collectFeed(chunks)

	if got := thinkingText(events); got != "weighing options" {
		t.Fatalf("expected leading block as thinking, got %q", got)
	}
	if got := answerText(events); got != "the answer" {
		t.Fatalf("expected trailing answer text, got %q", got)
	}
}

// TestThinkingTagParserLeadingBlockAcrossChunks confirms a leading block whose
// tags are split across many small chunks (opening tag and closing tag each
// straddling boundaries) is still recognized.
func TestThinkingTagParserLeadingBlockAcrossChunks(t *testing.T) {
	chunks := []string{"<thin", "king>rea", "soning</think", "ing>vis", "ible"}
	events := collectFeed(chunks)

	if got := thinkingText(events); got != "reasoning" {
		t.Fatalf("expected reassembled thinking %q, got %q", "reasoning", got)
	}
	if got := answerText(events); got != "visible" {
		t.Fatalf("expected reassembled answer %q, got %q", "visible", got)
	}
}

// TestThinkingTagParserUnclosedLeadingTagIsAnswer confirms the "closing
// required" half of the rule: a leading opening tag that never closes is not a
// real block, so the whole thing (tag included) is preserved as answer.
func TestThinkingTagParserUnclosedLeadingTagIsAnswer(t *testing.T) {
	open := "<" + "thinking" + ">"
	chunks := []string{open + "this never closes so it is all answer"}
	events := collectFeed(chunks)

	if got := thinkingText(events); got != "" {
		t.Fatalf("unclosed leading tag must not produce thinking, got %q", got)
	}
	want := open + "this never closes so it is all answer"
	if got := answerText(events); got != want {
		t.Fatalf("unclosed leading tag must be preserved verbatim\n got:  %q\n want: %q", got, want)
	}
}

// TestThinkingTagParserBalancedLiteralInsideThinking is the closing-side "断片儿"
// regression (Image #3): the reasoning inside a genuine leading block itself
// mentions a balanced literal <thinking>…</thinking> pair. The first literal
// </thinking> must NOT terminate the real block — only the close that balances the
// outermost open does. Previously strings.Index matched the first </thinking>,
// truncating the reasoning and leaking its tail (plus the real close tag) into the
// visible answer.
func TestThinkingTagParserBalancedLiteralInsideThinking(t *testing.T) {
	open := "<" + "thinking" + ">"
	closeTag := "</" + "thinking" + ">"
	// reasoning discusses a literal tag pair, then the real block closes.
	reasoning := "discussing the " + open + "literal" + closeTag + " pair here"
	chunks := []string{open + reasoning + closeTag + "the visible answer"}
	events := collectFeed(chunks)

	if got := thinkingText(events); got != reasoning {
		t.Fatalf("balanced literal pair inside reasoning must stay in thinking\n got:  %q\n want: %q", got, reasoning)
	}
	if got := answerText(events); got != "the visible answer" {
		t.Fatalf("answer after real close must survive intact, got %q", got)
	}
}

// TestThinkingTagParserBalancedLiteralAcrossChunks is the same closing-side case
// but with the balanced literal pair and the real close tag split across chunk
// boundaries, exercising the buffer-until-balanced behavior.
func TestThinkingTagParserBalancedLiteralAcrossChunks(t *testing.T) {
	open := "<" + "thinking" + ">"
	closeTag := "</" + "thinking" + ">"
	chunks := []string{
		open + "before ",
		open + "nested",
		closeTag + " after",
		closeTag + "answer",
	}
	events := collectFeed(chunks)

	wantThinking := "before " + open + "nested" + closeTag + " after"
	if got := thinkingText(events); got != wantThinking {
		t.Fatalf("nested pair across chunks must stay in thinking\n got:  %q\n want: %q", got, wantThinking)
	}
	if got := answerText(events); got != "answer" {
		t.Fatalf("answer after balanced close must survive, got %q", got)
	}
}

// TestThinkingTagParserPlainAnswerStreamsPromptly confirms plain answer text
// with no tag streams through without being held hostage by tag detection.
func TestThinkingTagParserPlainAnswerStreamsPromptly(t *testing.T) {
	// A single non-tag leading char resolves the start phase immediately.
	events := collectFeed([]string{"Hello, world"})
	if got := answerText(events); got != "Hello, world" {
		t.Fatalf("expected plain answer verbatim, got %q", got)
	}
	if got := thinkingText(events); got != "" {
		t.Fatalf("expected no thinking for plain answer, got %q", got)
	}
}

// TestClaudeStreamLiteralThinkingTagPreservedInAnswer is the end-to-end lock:
// an assistantResponseEvent answer that contains a literal <thinking> tag must
// arrive in full as text_delta output, with nothing dropped into a thinking
// block. This exercises the whole handleClaudeStream path, not just the parser.
func TestClaudeStreamLiteralThinkingTagPreservedInAnswer(t *testing.T) {
	cfgFile := t.TempDir() + "/config.json"
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:          "acct-literal",
		Enabled:     true,
		AccessToken: "token",
		ProfileArn:  "arn:aws:codewhisperer:profile/acct-literal",
	}); err != nil {
		t.Fatalf("add account: %v", err)
	}
	if err := config.UpdatePreferredEndpoint("kiro"); err != nil {
		t.Fatalf("set preferred endpoint: %v", err)
	}
	if err := config.UpdateEndpointFallback(false); err != nil {
		t.Fatalf("disable endpoint fallback: %v", err)
	}

	open := "<" + "thinking" + ">"
	// Split the answer so the literal tag lands mid-stream, mirroring the bug.
	part1 := "answer head with literal "
	part2 := open + " tag inline and a long tail that must survive"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": part1}))
		_, _ = w.Write(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": part2}))
	}))
	defer server.Close()

	oldEndpoints := kiroEndpoints
	kiroEndpoints = []kiroEndpoint{{URL: server.URL, Origin: "AI_EDITOR", Name: "test"}}
	defer func() { kiroEndpoints = oldEndpoints }()
	oldClient := kiroHttpStore.Load()
	kiroHttpStore.Store(&http.Client{Timeout: time.Second, Transport: &http.Transport{}})
	defer kiroHttpStore.Store(oldClient)

	p := accountpool.GetPool()
	p.Reload()
	h := &Handler{pool: p, promptCache: newPromptCacheTracker(defaultPromptCacheTTL)}

	payload := &KiroPayload{}
	payload.ConversationState.CurrentMessage.UserInputMessage = KiroUserInputMessage{
		Content: "hello",
		ModelID: "claude-sonnet-4.5",
		Origin:  "AI_EDITOR",
	}

	rec := httptest.NewRecorder()
	h.handleClaudeStream(rec, payload, "claude-sonnet-4.5", true, claudeThinkingResponseOptions{Format: "thinking"}, 1, 1, nil, nil, "", maxKiroInputTokens)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected success status, got %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()

	if strings.Contains(body, `"type":"thinking_delta"`) {
		t.Fatalf("literal tag in answer must not create a thinking block:\n%s", body)
	}
	// The tail after the literal tag must survive rather than being swallowed.
	if !strings.Contains(body, "long tail that must survive") {
		t.Fatalf("answer tail after literal tag was dropped:\n%s", body)
	}
}
