package proxy

import (
	"fmt"
	"kiro-go/config"
	"kiro-go/logger"
)

const maxKiroInputTokens = 200_000

// clientCompactionRatio is the fraction of a model's real input window at which
// the gateway tells the client to compact (Claude Code shrinks its context when
// the reported maxInputTokens is approached). 0.95 reproduces the historical
// 190k trigger for 200k-window models while scaling up for larger windows.
const clientCompactionRatio = 0.95

const contextLimitMessage = "Model context limit reached. Conversation size exceeds model capacity."

// modelHardInputTokenLimit returns the upstream input-token ceiling for the
// model. Estimates are inflated by claudeTokenCorrectionFactor, so the gate
// scales the ceiling by the same factor to compare in the same space.
func modelHardInputTokenLimit(model string) int {
	return getModelMaxInputTokens(model)
}

func modelEstimatedHardInputTokenLimit(model string) int {
	return int(float64(modelHardInputTokenLimit(model)) * claudeTokenCorrectionFactor)
}

// modelClientCompactionLimit returns the per-model token count at which the
// client should compact its context.
func modelClientCompactionLimit(model string) int {
	return int(float64(getModelMaxInputTokens(model)) * clientCompactionRatio)
}

// capInputTokensToContextWindow clamps the reported input tokens to the model's
// client-compaction line (window × clientCompactionRatio). Two upstream paths can
// otherwise surface an input count above the model's advertised window: the hard
// gate admits requests up to window × claudeTokenCorrectionFactor (estimate
// inflation), and the context-usage percentage path can report ~100% of the
// window. Clamping here keeps the displayed/recorded input at or below the
// compaction trigger so it never exceeds the model's context capacity.
func capInputTokensToContextWindow(inputTokens int, model string) int {
	limit := modelClientCompactionLimit(model)
	if limit > 0 && inputTokens > limit {
		return limit
	}
	return inputTokens
}

func exceedsKiroInputTokenLimit(estimatedInputTokens int, model string) bool {
	return estimatedInputTokens > modelEstimatedHardInputTokenLimit(model)
}

func exceedsClientCompactionLimit(estimatedInputTokens int, model string) bool {
	return estimatedInputTokens > modelClientCompactionLimit(model)
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

func contextLimitErrorMessage(estimatedInputTokens int, model string) string {
	if estimatedInputTokens <= 0 {
		return contextLimitMessage
	}
	return fmt.Sprintf("%s Estimated input tokens: %d; limit: %d. Request rejected before upstream to avoid silent context truncation.",
		contextLimitMessage, estimatedInputTokens, modelHardInputTokenLimit(model))
}

func promptTooLongErrorMessage(estimatedInputTokens int, model string) string {
	limit := modelClientCompactionLimit(model)
	if estimatedInputTokens <= 0 {
		estimatedInputTokens = limit + 1
	}
	return fmt.Sprintf("Prompt is too long: %d tokens > %d", estimatedInputTokens, limit)
}
