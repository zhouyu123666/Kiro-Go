package proxy

import (
	"math"
	"strings"
)

// modelMaxInputTokensByModel maps each upstream model ID to its real maximum
// input-token window, as reported by the Kiro ListAvailableModels API. This is
// the single source of truth for both context-usage accounting and the local
// input-size gate. Models absent from this table fall back to maxKiroInputTokens.
var modelMaxInputTokensByModel = map[string]int{
	"auto":              1_000_000,
	"claude-opus-4.8":   1_000_000,
	"claude-opus-4.7":   1_000_000,
	"claude-opus-4.6":   1_000_000,
	"claude-sonnet-5":   1_000_000,
	"claude-sonnet-4.6": 1_000_000,
	"gpt-5.6-sol":       1_000_000,
	"gpt-5.6-terra":     1_000_000,
	"gpt-5.6-luna":      1_000_000,
	"claude-opus-4.5":   200_000,
	"claude-sonnet-4.5": 200_000,
	"claude-sonnet-4":   200_000,
	"claude-haiku-4.5":  200_000,
	"qwen3-coder-next":  256_000,
	"glm-5":             200_000,
	"minimax-m2.5":      196_000,
	"minimax-m2.1":      196_000,
	"deepseek-3.2":      164_000,
}

// getModelMaxInputTokens returns the upstream maximum input-token window for the
// given client model, falling back to maxKiroInputTokens when the model is not
// in the table.
func getModelMaxInputTokens(model string) int {
	normalized := normalizeContextModelID(model)
	if tokens, ok := modelMaxInputTokensByModel[normalized]; ok {
		return tokens
	}
	return maxKiroInputTokens
}

func getClaudeCodeUsageReportWindow(model string) int {
	return getModelMaxInputTokens(model)
}

func normalizeContextModelID(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	if idx := strings.LastIndex(normalized, ":"); idx >= 0 {
		normalized = strings.TrimSpace(normalized[idx+1:])
	}
	normalized = strings.TrimSuffix(normalized, "-thinking")

	mapped, _ := ParseModelAndThinking(normalized, "-thinking")
	normalized = strings.ToLower(strings.TrimSpace(mapped))

	parts := strings.Split(normalized, "-")
	if len(parts) >= 4 && parts[0] == "claude" && isClaudeContextFamily(parts[1]) && strings.Contains(parts[2], ".") && isDigits(parts[3]) {
		return strings.Join(parts[:3], "-")
	}
	return normalized
}

func isClaudeContextFamily(family string) bool {
	return family == "opus" || family == "sonnet" || family == "haiku"
}

func isDigits(value string) bool {
	if value == "" {
		return false
	}
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func totalTokensFromContextUsagePercentage(percentage float64, reportWindow int) int {
	if percentage <= 0 || reportWindow <= 0 {
		return 0
	}
	return int(math.Round(float64(reportWindow) * percentage / 100.0))
}

func inputTokensFromContextUsagePercentage(percentage float64, reportWindow, outputTokens int) int {
	totalTokens := totalTokensFromContextUsagePercentage(percentage, reportWindow)
	if totalTokens <= 0 {
		return 0
	}
	return maxInt(totalTokens-outputTokens, 0)
}
