package proxy

import (
	"encoding/json"
	"math"
	"unicode"
)

const (
	claudeTokenCorrectionFactor        = 1.10
	claudeBillingTokenCorrectionFactor = 1.86
	// Kiro's contextUsagePercentage includes a backend-owned default system
	// prompt that is not part of the client's visible first request. Exclude
	// that fixed overhead only from first-turn public Claude usage derived from
	// context occupancy; later turns, billing, and context limits keep it.
	kiroDefaultSystemPromptTokens = 4096
)

// Binary media (images, documents) is sent to the model as attachments, not as
// the raw base64 string in the request body. Estimating these blocks by JSON-
// marshalling them would count the base64 payload (~bytes/4 tokens), which makes
// even a modest image look like hundreds of thousands of tokens and trips the
// input limit before the request reaches upstream. Use fixed per-attachment
// estimates that approximate the model's actual accounting instead.
const (
	approxImageInputTokens    = 1600
	approxDocumentInputTokens = 3000
)

func estimateApproxTokens(text string) int {
	if text == "" {
		return 0
	}

	runes := []rune(text)
	length := len(runes)
	if length == 0 {
		return 0
	}
	if length < 5 {
		return max(1, int(math.Ceil(float64(length)/3.0)))
	}

	lexicalEstimate := estimateLexicalTokenFloor(runes)
	estimated := int(math.Ceil(float64(lexicalEstimate) * claudeTokenCorrectionFactor))

	if estimated < 1 {
		return 1
	}
	return estimated
}

func estimateLexicalTokenFloor(runes []rune) int {
	tokens := 0
	asciiWordLen := 0
	flushASCIIWord := func() {
		if asciiWordLen == 0 {
			return
		}
		tokens += estimateASCIIWordTokens(asciiWordLen)
		asciiWordLen = 0
	}

	for _, r := range runes {
		if isASCIILetter(r) {
			asciiWordLen++
			continue
		}
		flushASCIIWord()

		switch {
		case unicode.IsSpace(r):
			continue
		case r >= '0' && r <= '9':
			tokens++
		case r >= 0x80:
			tokens++
		default:
			tokens++
		}
	}
	flushASCIIWord()
	return tokens
}

func estimateASCIIWordTokens(length int) int {
	if length <= 0 {
		return 0
	}
	if length <= 12 {
		return 1
	}
	return int(math.Ceil(float64(length) / 6.0))
}

func isASCIILetter(r rune) bool {
	return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z')
}

func estimateClaudeRequestInputTokens(req *ClaudeRequest) int {
	if req == nil {
		return 0
	}

	total := estimateClaudeValueTokens(req.System)

	for _, msg := range req.Messages {
		total += estimateClaudeValueTokens(msg.Content)
	}

	for _, tool := range req.Tools {
		total += estimateApproxTokens(tool.Name)
		total += estimateApproxTokens(tool.Description)
		total += estimateJSONTokens(tool.InputSchema)
	}

	return total
}

// estimateKiroPayloadInputTokens estimates the input tokens of the exact payload
// that will be sent upstream. This is what the 200k limit check should use: it
// accounts for PDF text inlined into message content, the system prompt re-emitted
// as priming history, tool-spec deduplication, and tool-result flattening — none
// of which are visible when estimating the raw client request. Estimating the raw
// request can under- or over-count relative to the wire body (most importantly,
// a text-heavy PDF is ~3k as a document block but ~15k once inlined), so gating on
// the payload keeps our local check aligned with what Kiro actually receives.
func estimateKiroPayloadInputTokens(payload *KiroPayload) int {
	if payload == nil {
		return 0
	}

	total := estimateKiroUserMessageTokens(&payload.ConversationState.CurrentMessage.UserInputMessage)

	for _, h := range payload.ConversationState.History {
		if h.UserInputMessage != nil {
			total += estimateKiroUserMessageTokens(h.UserInputMessage)
		}
		if h.AssistantResponseMessage != nil {
			total += estimateApproxTokens(h.AssistantResponseMessage.Content)
			for _, tu := range h.AssistantResponseMessage.ToolUses {
				total += estimateApproxTokens(tu.Name)
				total += estimateJSONTokens(tu.Input)
			}
		}
	}

	return total
}

func estimateKiroUserMessageTokens(msg *KiroUserInputMessage) int {
	if msg == nil {
		return 0
	}

	total := estimateApproxTokens(msg.Content)
	total += len(msg.Images) * approxImageInputTokens
	total += len(msg.Documents) * approxDocumentInputTokens

	if ctx := msg.UserInputMessageContext; ctx != nil {
		for _, tool := range ctx.Tools {
			total += estimateApproxTokens(tool.ToolSpecification.Name)
			total += estimateApproxTokens(tool.ToolSpecification.Description)
			total += estimateJSONTokens(tool.ToolSpecification.InputSchema.JSON)
		}
		for _, tr := range ctx.ToolResults {
			for _, c := range tr.Content {
				total += estimateApproxTokens(c.Text)
			}
		}
	}

	return total
}

func estimateClaudeOutputTokens(content, thinkingContent string, toolUses []KiroToolUse) int {
	total := estimateApproxTokens(content)
	total += estimateApproxTokens(thinkingContent)

	for _, tu := range toolUses {
		total += estimateApproxTokens(tu.Name)
		total += estimateJSONTokens(tu.Input)
	}

	return total
}

func estimateClaudeValueTokens(v interface{}) int {
	switch value := v.(type) {
	case nil:
		return 0
	case string:
		return estimateApproxTokens(value)
	case []interface{}:
		total := 0
		for _, part := range value {
			total += estimateClaudeValueTokens(part)
		}
		return total
	case map[string]interface{}:
		typeName, _ := value["type"].(string)
		switch typeName {
		case "text":
			if text, ok := value["text"].(string); ok {
				return estimateApproxTokens(text)
			}
		case "thinking":
			if thinking, ok := value["thinking"].(string); ok {
				return estimateApproxTokens(thinking)
			}
		case "tool_use":
			total := 0
			if name, ok := value["name"].(string); ok {
				total += estimateApproxTokens(name)
			}
			if input, ok := value["input"]; ok {
				total += estimateJSONTokens(input)
			}
			if total > 0 {
				return total
			}
		case "tool_result":
			if content, ok := value["content"]; ok {
				return estimateClaudeValueTokens(content)
			}
		case "image", "image_url", "input_image":
			return approxImageInputTokens
		case "document", "input_file", "file":
			return approxDocumentInputTokens
		}

		total := 0
		if text, ok := value["text"].(string); ok {
			total += estimateApproxTokens(text)
		}
		if thinking, ok := value["thinking"].(string); ok {
			total += estimateApproxTokens(thinking)
		}
		if content, ok := value["content"]; ok {
			total += estimateClaudeValueTokens(content)
		}
		if total > 0 {
			return total
		}

		return estimateJSONTokens(value)
	default:
		return estimateJSONTokens(value)
	}
}

func estimateJSONTokens(v interface{}) int {
	if v == nil {
		return 0
	}

	b, err := json.Marshal(v)
	if err != nil {
		return 0
	}

	return estimateApproxTokens(string(b))
}

func estimateOpenAIRequestInputTokens(req *OpenAIRequest) int {
	if req == nil {
		return 0
	}

	total := 0

	for _, msg := range req.Messages {
		total += estimateOpenAIContentTokens(msg.Content)
		total += estimateApproxTokens(msg.ToolCallID)
		for _, tc := range msg.ToolCalls {
			total += estimateApproxTokens(tc.Function.Name)
			total += estimateApproxTokens(tc.Function.Arguments)
		}
	}

	for _, tool := range req.Tools {
		total += estimateApproxTokens(tool.Function.Name)
		total += estimateApproxTokens(tool.Function.Description)
		total += estimateJSONTokens(tool.Function.Parameters)
	}

	return total
}

func estimateOpenAIContentTokens(content interface{}) int {
	switch value := content.(type) {
	case nil:
		return 0
	case string:
		return estimateApproxTokens(value)
	case []interface{}:
		total := 0
		for _, part := range value {
			total += estimateOpenAIContentTokens(part)
		}
		return total
	case map[string]interface{}:
		switch partType, _ := value["type"].(string); partType {
		case "text", "input_text", "output_text":
			if text, ok := value["text"].(string); ok {
				return estimateApproxTokens(text)
			}
		case "image", "image_url", "input_image":
			return approxImageInputTokens
		case "file", "input_file", "document":
			return approxDocumentInputTokens
		}
		if nested, ok := value["content"]; ok {
			if n := estimateOpenAIContentTokens(nested); n > 0 {
				return n
			}
		}
		if text, ok := value["text"].(string); ok {
			return estimateApproxTokens(text)
		}
		return estimateJSONTokens(value)
	default:
		return estimateJSONTokens(value)
	}
}

func estimateOpenAIOutputTokens(content, reasoningContent string, toolUses []KiroToolUse) int {
	return estimateClaudeOutputTokens(content, reasoningContent, toolUses)
}

func applyClaudeBillingInputTokenCorrection(estimatedInputTokens int) int {
	if estimatedInputTokens <= 0 {
		return 0
	}
	return int(math.Ceil(float64(estimatedInputTokens) * claudeBillingTokenCorrectionFactor / claudeTokenCorrectionFactor))
}

// Match public billing to the client-visible request context. Request-side
// estimates use the context-limit correction factor, then billing applies its
// own multiplier so prompt-too-long checks can remain conservative and stable.
// Kiro's contextUsagePercentage is backend context occupancy and can include
// hidden service state, so use it only as a last-resort fallback when no
// request-side estimate is available.
func finalizeKiroInputTokens(upstreamInputTokens, outputTokens int, contextUsagePercentage float64, usageReportWindow, estimatedInputTokens int, model string) int {
	inputTokens := 0
	if estimatedInputTokens > 0 {
		inputTokens = applyClaudeBillingInputTokenCorrection(estimatedInputTokens)
	} else if upstreamInputTokens > 0 {
		inputTokens = upstreamInputTokens
	}
	if inputTokens <= 0 && contextUsagePercentage > 0 {
		inputTokens = inputTokensFromContextUsagePercentage(contextUsagePercentage, usageReportWindow, outputTokens)
	}
	return capInputTokensToContextWindow(inputTokens, model)
}

func finalizeClaudeUsageInputTokens(upstreamInputTokens, outputTokens int, contextUsagePercentage float64, usageReportWindow, estimatedInputTokens int, model string) int {
	return finalizeClaudeUsageInputTokensForTurn(upstreamInputTokens, outputTokens, contextUsagePercentage, usageReportWindow, estimatedInputTokens, model, false)
}

func finalizeClaudeUsageInputTokensForTurn(upstreamInputTokens, outputTokens int, contextUsagePercentage float64, usageReportWindow, estimatedInputTokens int, model string, firstConversationTurn bool) int {
	inputTokens := 0
	if contextUsagePercentage > 0 {
		inputTokens = inputTokensFromContextUsagePercentage(contextUsagePercentage, usageReportWindow, outputTokens)
		if firstConversationTurn {
			inputTokens = maxInt(inputTokens-kiroDefaultSystemPromptTokens, 0)
		}
	} else if upstreamInputTokens > 0 {
		inputTokens = upstreamInputTokens
	} else if estimatedInputTokens > 0 {
		inputTokens = estimatedInputTokens
	}
	return capInputTokensToContextWindow(inputTokens, model)
}

func finalizeKiroReportedInputTokens(inputTokens int, model string) int {
	return capInputTokensToContextWindow(inputTokens, model)
}
