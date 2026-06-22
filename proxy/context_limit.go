package proxy

import "fmt"

const maxKiroInputTokens = 200_000

const contextLimitMessage = "Model context limit reached. Conversation size exceeds model capacity."

func exceedsKiroInputTokenLimit(estimatedInputTokens int) bool {
	return estimatedInputTokens > maxKiroInputTokens
}

func contextLimitErrorMessage(estimatedInputTokens int) string {
	if estimatedInputTokens <= 0 {
		return contextLimitMessage
	}
	return fmt.Sprintf("%s Estimated input tokens: %d; limit: %d. Request rejected before upstream to avoid silent context truncation.",
		contextLimitMessage, estimatedInputTokens, maxKiroInputTokens)
}
