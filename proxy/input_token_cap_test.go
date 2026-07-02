package proxy

import "testing"

func TestFinalizeKiroInputTokensUsesUpstreamInput(t *testing.T) {
	got := finalizeKiroInputTokens(4397, 100, 50, maxKiroInputTokens, 9000, "claude-sonnet-4.5")
	if got != 4397 {
		t.Fatalf("expected upstream input tokens to be preserved, got %d", got)
	}
}

func TestFinalizeKiroInputTokensUsesContextUsageBeforeEstimate(t *testing.T) {
	got := finalizeKiroInputTokens(0, 1234, 2.5, maxKiroInputTokens, 9000, "claude-sonnet-4.5")
	want := inputTokensFromContextUsagePercentage(2.5, maxKiroInputTokens, 1234)
	if got != want {
		t.Fatalf("expected context usage input tokens %d, got %d", want, got)
	}
}

func TestFinalizeKiroInputTokensFallsBackToEstimate(t *testing.T) {
	got := finalizeKiroInputTokens(0, 100, 0, maxKiroInputTokens, 4397, "claude-sonnet-4.5")
	if got != 4397 {
		t.Fatalf("expected estimated input tokens, got %d", got)
	}
}

func TestClaudeUsageMapKeepsDisplayInputWithCacheBreakdown(t *testing.T) {
	totalInput := finalizeKiroInputTokens(0, 1038, 0, maxKiroInputTokens, 59081, "claude-haiku-4.5")
	cacheUsage := promptCacheUsage{
		CacheReadInputTokens:     42686,
		CacheCreationInputTokens: 7533,
	}
	visibleInput := finalizeKiroDisplayInputTokens(1, totalInput, "claude-haiku-4.5")
	usage := buildClaudeUsageMap(visibleInput, 1038, cacheUsage, true)

	if got := usage["input_tokens"]; got != 1 {
		t.Fatalf("expected displayed input tokens to exclude prompt, got %#v", got)
	}
	if got := usage["cache_read_input_tokens"]; got != 42686 {
		t.Fatalf("expected cache read tokens to be preserved, got %#v", got)
	}
	if got := usage["cache_creation_input_tokens"]; got != 7533 {
		t.Fatalf("expected cache creation tokens to be preserved, got %#v", got)
	}
}

func TestFinalizeKiroInputTokensClampsToContextWindow(t *testing.T) {
	got := finalizeKiroInputTokens(210000, 100, 0, maxKiroInputTokens, 0, "claude-sonnet-4.5")
	if got != 190000 {
		t.Fatalf("expected 200k model to clamp to 190000, got %d", got)
	}
}

func TestCapInputTokensToContextWindow(t *testing.T) {
	cases := []struct {
		name       string
		inputToken int
		model      string
		want       int
	}{
		{"200k model under limit unchanged", 150000, "claude-sonnet-4.5", 150000},
		{"200k model over limit clamped to 190k", 210000, "claude-sonnet-4.5", 190000},
		{"200k model exactly at limit", 190000, "claude-sonnet-4.5", 190000},
		{"1M model 1.05M clamped to 950k", 1050000, "claude-opus-4.8", 950000},
		{"1M model under limit unchanged", 800000, "claude-opus-4.8", 800000},
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
