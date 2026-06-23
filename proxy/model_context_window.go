package proxy

func getClaudeCodeUsageReportWindow(_ string) int {
	return maxKiroInputTokens
}

func inputTokensFromContextUsagePercentage(percentage float64, reportWindow int) int {
	if percentage <= 0 || reportWindow <= 0 {
		return 0
	}
	if percentage > 100 {
		percentage = 100
	}
	return int(percentage * float64(reportWindow) / 100.0)
}
