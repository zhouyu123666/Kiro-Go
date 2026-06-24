package proxy

import (
	"fmt"
	"kiro-go/config"
	"kiro-go/logger"
)

const maxKiroInputTokens = 200_000
const clientKiroInputTokens = 190_000
const maxKiroEstimatedInputTokens = int(maxKiroInputTokens * claudeTokenCorrectionFactor)

const contextLimitMessage = "Model context limit reached. Conversation size exceeds model capacity."

func exceedsKiroInputTokenLimit(estimatedInputTokens int) bool {
	return estimatedInputTokens > maxKiroEstimatedInputTokens
}

func exceedsClientCompactionLimit(estimatedInputTokens int) bool {
	return estimatedInputTokens > clientKiroInputTokens
}

func estimateClaudeCompactionInputTokens(req *ClaudeRequest) int {
	if req == nil {
		return 0
	}
	thinkingCfg := config.GetThinkingConfig()
	actualModel, thinking := resolveClaudeThinkingMode(req.Model, req.Thinking, thinkingCfg.Suffix)
	cloned := *req
	cloned.Model = actualModel
	effectiveReq := cloneClaudeRequestForThinking(&cloned, thinking)
	if tokens, ok := estimateClaudeRequestTikTokenInputTokens(effectiveReq); ok {
		return tokens
	}
	logger.Warnf("[ClaudeContextGate] kiro-gateway token estimate unavailable; falling back to approximate token estimator")
	return estimateClaudeRequestInputTokens(effectiveReq)
}

func contextLimitErrorMessage(estimatedInputTokens int) string {
	if estimatedInputTokens <= 0 {
		return contextLimitMessage
	}
	return fmt.Sprintf("%s Estimated input tokens: %d; limit: %d. Request rejected before upstream to avoid silent context truncation.",
		contextLimitMessage, estimatedInputTokens, maxKiroInputTokens)
}

func promptTooLongErrorMessage(estimatedInputTokens int) string {
	if estimatedInputTokens <= 0 {
		estimatedInputTokens = clientKiroInputTokens + 1
	}
	return fmt.Sprintf("Prompt is too long: %d tokens > %d", estimatedInputTokens, clientKiroInputTokens)
}
