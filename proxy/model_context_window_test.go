package proxy

import "testing"

func TestGetClaudeCodeUsageReportWindowDefaultsPlainModelsToKiroMax(t *testing.T) {
	cases := []string{
		"claude-opus-4.8",
		"claude-opus-4-8",
		"claude-sonnet-4.6",
		"unknown-model",
	}
	for _, model := range cases {
		if got := getClaudeCodeUsageReportWindow(model); got != maxKiroInputTokens {
			t.Errorf("getClaudeCodeUsageReportWindow(%q) = %d, want %d", model, got, maxKiroInputTokens)
		}
	}
}

func TestInputTokensFromContextUsagePercentageUsesClaudeCodeWindow(t *testing.T) {
	if got := inputTokensFromContextUsagePercentage(93.5, maxKiroInputTokens); got != 187_000 {
		t.Fatalf("93.5%% of %d = %d, want 187000", maxKiroInputTokens, got)
	}
	if got := inputTokensFromContextUsagePercentage(100, maxKiroInputTokens); got != maxKiroInputTokens {
		t.Fatalf("100%% of %d = %d, want %d", maxKiroInputTokens, got, maxKiroInputTokens)
	}
	if got := inputTokensFromContextUsagePercentage(150, maxKiroInputTokens); got != maxKiroInputTokens {
		t.Fatalf("over-100%% usage should clamp at %d, got %d", maxKiroInputTokens, got)
	}
	if got := inputTokensFromContextUsagePercentage(-1, maxKiroInputTokens); got != 0 {
		t.Fatalf("negative usage should report 0, got %d", got)
	}
}
