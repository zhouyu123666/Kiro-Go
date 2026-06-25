package proxy

import "testing"

func TestCapInputTokens(t *testing.T) {
	cases := []struct {
		name      string
		userInput int
		computed  int
		want      int
	}{
		{"user smaller than computed -> user", 1, 4397, 1},
		{"user equal to computed -> user", 100, 100, 100},
		{"user larger than computed -> computed", 5000, 4397, 4397},
		{"no user message (0) -> computed unchanged", 0, 4397, 4397},
		{"negative user (defensive) -> computed unchanged", -3, 4397, 4397},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := capInputTokens(tc.userInput, tc.computed); got != tc.want {
				t.Fatalf("capInputTokens(%d, %d) = %d, want %d", tc.userInput, tc.computed, got, tc.want)
			}
		})
	}
}

func TestCapInputTokensToContextWindow(t *testing.T) {
	cases := []struct {
		name       string
		inputToken int
		model      string
		want       int
	}{
		// 200k 模型：压缩触发线 = 200000 * 0.95 = 190000
		{"200k model under limit unchanged", 150000, "claude-sonnet-4.5", 150000},
		{"200k model over limit clamped to 190k", 210000, "claude-sonnet-4.5", 190000},
		{"200k model exactly at limit", 190000, "claude-sonnet-4.5", 190000},
		// 1M 模型：压缩触发线 = 1000000 * 0.95 = 950000，验证不会超过 100 万
		{"1M model 1.05M clamped to 950k", 1050000, "claude-opus-4.8", 950000},
		{"1M model under limit unchanged", 800000, "claude-opus-4.8", 800000},
		// 未知模型回退 maxKiroInputTokens(200k)，触发线 190k
		{"unknown model falls back to 190k", 250000, "some-unknown-model", 190000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := capInputTokensToContextWindow(tc.inputToken, tc.model); got != tc.want {
				t.Fatalf("capInputTokensToContextWindow(%d, %q) = %d, want %d", tc.inputToken, tc.model, got, tc.want)
			}
		})
	}
}

func TestEstimateClaudeLastUserInputTokens(t *testing.T) {
	req := &ClaudeRequest{
		System: "you are a very long system prompt with lots of instructions",
		Messages: []ClaudeMessage{
			{Role: "user", Content: "an earlier and much longer user question that should be ignored"},
			{Role: "assistant", Content: "some assistant reply"},
			{Role: "user", Content: "hi"},
		},
	}
	got := estimateClaudeLastUserInputTokens(req)
	want := estimateClaudeValueTokens("hi")
	if got != want {
		t.Fatalf("last user tokens = %d, want %d (only the final user message counts)", got, want)
	}
	if got <= 0 {
		t.Fatalf("expected positive token count for non-empty user message, got %d", got)
	}
}

func TestEstimateClaudeLastUserInputTokensNoUser(t *testing.T) {
	req := &ClaudeRequest{
		Messages: []ClaudeMessage{
			{Role: "assistant", Content: "only assistant content"},
		},
	}
	if got := estimateClaudeLastUserInputTokens(req); got != 0 {
		t.Fatalf("expected 0 when no user message present, got %d", got)
	}
}

func TestEstimateOpenAILastUserInputTokens(t *testing.T) {
	req := &OpenAIRequest{
		Messages: []OpenAIMessage{
			{Role: "system", Content: "long system content"},
			{Role: "user", Content: "first long user turn ignored"},
			{Role: "assistant", Content: "reply"},
			{Role: "user", Content: "hi"},
		},
	}
	got := estimateOpenAILastUserInputTokens(req)
	want := estimateOpenAIContentTokens("hi")
	if got != want {
		t.Fatalf("openai last user tokens = %d, want %d", got, want)
	}
}
