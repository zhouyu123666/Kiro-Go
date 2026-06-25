package proxy

import "testing"

func TestGetClaudeCodeUsageReportWindowUsesModelContext(t *testing.T) {
	cases := []struct {
		model string
		want  int
	}{
		{model: "claude-opus-4.8", want: 1_000_000},
		{model: "claude-opus-4-8", want: 1_000_000},
		{model: "claude-sonnet-4.6", want: 1_000_000},
		{model: "anthropic:claude-opus-4-5-20251101", want: 200_000},
		{model: "claude-opus-4.5", want: 200_000},
		{model: "claude-sonnet-4-5-20250929", want: 200_000},
		{model: "claude-haiku-4.5", want: 200_000},
		{model: "qwen3-coder-next", want: 256_000},
		{model: "deepseek-3.2", want: 164_000},
		{model: "unknown-model", want: maxKiroInputTokens},
	}
	for _, tc := range cases {
		if got := getClaudeCodeUsageReportWindow(tc.model); got != tc.want {
			t.Errorf("getClaudeCodeUsageReportWindow(%q) = %d, want %d", tc.model, got, tc.want)
		}
	}
}

func TestInputTokensFromContextUsagePercentageSubtractsOutputTokens(t *testing.T) {
	if got := totalTokensFromContextUsagePercentage(93.5, maxKiroInputTokens); got != 187_000 {
		t.Fatalf("total 93.5%% of %d = %d, want 187000", maxKiroInputTokens, got)
	}
	if got := inputTokensFromContextUsagePercentage(93.5, maxKiroInputTokens, 1_234); got != 185_766 {
		t.Fatalf("input tokens should subtract output tokens, got %d", got)
	}
	if got := inputTokensFromContextUsagePercentage(100, maxKiroInputTokens, 700); got != 199_300 {
		t.Fatalf("100%% of %d minus output = %d, want 199300", maxKiroInputTokens, got)
	}
	if got := inputTokensFromContextUsagePercentage(150, maxKiroInputTokens, 0); got != 300_000 {
		t.Fatalf("over-100%% usage should preserve upstream percentage, got %d", got)
	}
	if got := inputTokensFromContextUsagePercentage(1, maxKiroInputTokens, 3_000); got != 0 {
		t.Fatalf("output larger than total context should clamp input at 0, got %d", got)
	}
	if got := inputTokensFromContextUsagePercentage(-1, maxKiroInputTokens, 0); got != 0 {
		t.Fatalf("negative usage should report 0, got %d", got)
	}
}
