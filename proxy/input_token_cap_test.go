package proxy

import (
	"math"
	"testing"
)

func TestFinalizeKiroInputTokensFallsBackToUpstreamInput(t *testing.T) {
	got := finalizeKiroInputTokens(4397, 100, 50, maxKiroInputTokens, 0, "claude-sonnet-4.5")
	if got != 4397 {
		t.Fatalf("expected upstream input tokens when request estimate is unavailable, got %d", got)
	}
}

func TestFinalizeKiroInputTokensPrefersRequestEstimateOverContextUsage(t *testing.T) {
	got := finalizeKiroInputTokens(0, 100, 50, maxKiroInputTokens, 1100, "claude-sonnet-4.5")
	want := applyClaudeBillingInputTokenCorrection(1100)
	if got != want {
		t.Fatalf("expected billable request-estimated input tokens, got %d", got)
	}
}

func TestFinalizeKiroInputTokensFallsBackToContextUsage(t *testing.T) {
	got := finalizeKiroInputTokens(0, 1234, 2.5, maxKiroInputTokens, 0, "claude-sonnet-4.5")
	want := inputTokensFromContextUsagePercentage(2.5, maxKiroInputTokens, 1234)
	if got != want {
		t.Fatalf("expected context usage input tokens when request and upstream estimates are unavailable: want %d, got %d", want, got)
	}
}

func TestFinalizeKiroInputTokensFallsBackToEstimate(t *testing.T) {
	got := finalizeKiroInputTokens(0, 100, 0, maxKiroInputTokens, 1100, "claude-sonnet-4.5")
	want := applyClaudeBillingInputTokenCorrection(1100)
	if got != want {
		t.Fatalf("expected billable estimated input tokens, got %d", got)
	}
}

func TestFinalizeClaudeUsageInputTokensPrefersUpstreamUsageOverEstimate(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(1234, 100, 0, maxKiroInputTokens, 9999, "claude-sonnet-4.5")
	if got != 1234 {
		t.Fatalf("expected upstream usage 1234 to win over request estimate, got %d", got)
	}
}

func TestFinalizeClaudeUsageInputTokensPrefersContextUsageOverUpstreamAndEstimate(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(1234, 100, 5, maxKiroInputTokens, 9999, "claude-sonnet-4.5")
	want := calibratedClaudeInputTokensFromContextUsage(5, maxKiroInputTokens, 100)
	if got != want {
		t.Fatalf("expected calibrated context-derived input tokens %d to win, got %d", want, got)
	}
}

func TestFinalizeClaudeUsageInputTokensFallsBackToContextUsageLast(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(0, 100, 5, maxKiroInputTokens, 0, "claude-sonnet-4.5")
	want := calibratedClaudeInputTokensFromContextUsage(5, maxKiroInputTokens, 100)
	if got != want {
		t.Fatalf("expected context-derived fallback input tokens %d, got %d", want, got)
	}
}

func TestFinalizeClaudeUsageInputTokensPrefersContextUsageOverEstimate(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(0, 100, 5, maxKiroInputTokens, 9999, "claude-sonnet-4.5")
	want := calibratedClaudeInputTokensFromContextUsage(5, maxKiroInputTokens, 100)
	if got != want {
		t.Fatalf("expected calibrated context-derived input tokens %d to win over request estimate, got %d", want, got)
	}
}

func TestFinalizeClaudeUsageInputTokensExcludesKiroDefaultSystemPrompt(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(0, 42, 2.2, maxKiroInputTokens, 2, "claude-haiku-4.5")
	want := calibratedClaudeInputTokensFromContextUsage(2.2, maxKiroInputTokens, 42)
	if got != want {
		t.Fatalf("expected public input tokens %d after removing Kiro system prompt, got %d", want, got)
	}
}

func TestFinalizeClaudeUsageInputTokensUsesRawEstimateWithoutBillingMultiplier(t *testing.T) {
	got := finalizeClaudeUsageInputTokens(0, 100, 0, maxKiroInputTokens, 1100, "claude-sonnet-4.5")
	if got != 1100 {
		t.Fatalf("expected raw request estimate 1100, got %d", got)
	}
}

func TestFinalizeClaudeUsageInputTokensMatchesR8BaselineTiers(t *testing.T) {
	cases := []struct {
		contextInput int
		baseline     int
	}{
		// Context inputs reconstructed from the latest isolated aitokentest
		// retest. These must remain within 1% of the work.tokencheap.io baseline.
		{6733, 8},
		{7518, 1005},
		{12375, 9986},
		{60799, 99813},
		{277116, 499036},
	}
	for _, tc := range cases {
		percentage := float64(tc.contextInput) / 10000 // 1M context window
		got := finalizeClaudeUsageInputTokens(1, 0, percentage, 1_000_000, 1, "claude-opus-4.8")
		delta := math.Abs(float64(got-tc.baseline)) / float64(tc.baseline)
		if delta > 0.01 {
			t.Fatalf("context input %d: got %d, baseline %d, delta %.2f%%", tc.contextInput, got, tc.baseline, delta*100)
		}
	}
}

func TestClaudeUsageMapReportsUncachedInputWithCacheBreakdown(t *testing.T) {
	totalInput := finalizeKiroInputTokens(0, 1038, 0, maxKiroInputTokens, 59081, "claude-haiku-4.5")
	cacheUsage := promptCacheUsage{
		CacheReadInputTokens:     42686,
		CacheCreationInputTokens: 7533,
	}
	reportedInput := finalizeKiroReportedInputTokens(totalInput, "claude-haiku-4.5")
	usage := buildClaudeUsageMap(reportedInput, 1038, cacheUsage, true)

	expectedUncachedInput := totalInput - cacheUsage.CacheReadInputTokens - cacheUsage.CacheCreationInputTokens
	if got := usage["input_tokens"]; got != expectedUncachedInput {
		t.Fatalf("expected reported input tokens to use uncached input %d, got %#v", expectedUncachedInput, got)
	}
	if got := usage["cache_read_input_tokens"]; got != 42686 {
		t.Fatalf("expected cache read tokens to be preserved, got %#v", got)
	}
	if got := usage["cache_creation_input_tokens"]; got != 7533 {
		t.Fatalf("expected cache creation tokens to be preserved, got %#v", got)
	}
	reconstructedTotal := usage["input_tokens"].(int) + usage["cache_read_input_tokens"].(int) + usage["cache_creation_input_tokens"].(int)
	if reconstructedTotal != totalInput {
		t.Fatalf("expected input + cache read + cache creation to reconstruct total input %d, got %d", totalInput, reconstructedTotal)
	}
}

func TestClaudeUsageMapRebalancesCacheAgainstFinalTotal(t *testing.T) {
	usage := promptCacheUsage{
		CacheReadInputTokens:       80,
		CacheCreationInputTokens:   20,
		CacheCreation5mInputTokens: 8,
		CacheCreation1hInputTokens: 12,
		CacheCoveredEstimate:       100,
		PromptTotalEstimate:        100,
	}

	m := buildClaudeUsageMap(50, 10, usage, true)

	if m["input_tokens"].(int)+m["cache_read_input_tokens"].(int)+m["cache_creation_input_tokens"].(int) != 50 {
		t.Fatalf("expected rebalance to preserve total input 50, got %#v", m)
	}
	if m["input_tokens"].(int) != claudeUsageEnvelopeMinTokens {
		t.Fatalf("expected fully cached prompt to keep envelope floor %d, got %#v", claudeUsageEnvelopeMinTokens, m["input_tokens"])
	}
}

func TestClaudeUsageMapScalesCacheAgainstPromptCoverageRatio(t *testing.T) {
	usage := promptCacheUsage{
		CacheReadInputTokens:       80,
		CacheCreationInputTokens:   20,
		CacheCreation5mInputTokens: 8,
		CacheCreation1hInputTokens: 12,
		CacheCoveredEstimate:       100,
		PromptTotalEstimate:        200,
	}

	m := buildClaudeUsageMap(50, 10, usage, true)

	if got := m["input_tokens"]; got != 25 {
		t.Fatalf("expected 25 uncached tokens after ratio scaling, got %#v", got)
	}
	if got := m["cache_read_input_tokens"]; got != 20 {
		t.Fatalf("expected read cache to scale to 20, got %#v", got)
	}
	if got := m["cache_creation_input_tokens"]; got != 5 {
		t.Fatalf("expected creation cache to scale to 5, got %#v", got)
	}
	if m["input_tokens"].(int)+m["cache_read_input_tokens"].(int)+m["cache_creation_input_tokens"].(int) != 50 {
		t.Fatalf("expected usage fields to reconstruct total input 50, got %#v", m)
	}
}

func TestClaudeUsageMapAppliesEnvelopeFloorWhenCacheConsumesAllPublicInput(t *testing.T) {
	usage := promptCacheUsage{
		CacheCreationInputTokens:   10,
		CacheReadInputTokens:       30,
		CacheCreation5mInputTokens: 10,
		CacheCoveredEstimate:       40,
		PromptTotalEstimate:        40,
	}

	m := buildClaudeUsageMap(40, 1, usage, true)

	if got := m["input_tokens"]; got != claudeUsageEnvelopeMinTokens {
		t.Fatalf("expected envelope floor %d, got %#v", claudeUsageEnvelopeMinTokens, got)
	}
	if m["input_tokens"].(int)+m["cache_read_input_tokens"].(int)+m["cache_creation_input_tokens"].(int) != 40 {
		t.Fatalf("expected usage fields to reconstruct total input 40, got %#v", m)
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
