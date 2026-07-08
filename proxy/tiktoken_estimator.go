package proxy

import (
	"encoding/json"
	"sync"

	tiktoken "github.com/pkoukk/tiktoken-go"
	tiktoken_loader "github.com/pkoukk/tiktoken-go-loader"
)

const tiktokenEncodingName = "cl100k_base"

var (
	tiktokenOnce    sync.Once
	tiktokenEncoder *tiktoken.Tiktoken
	tiktokenErr     error
)

func getTikTokenEncoder() (*tiktoken.Tiktoken, error) {
	tiktokenOnce.Do(func() {
		tiktoken.SetBpeLoader(tiktoken_loader.NewOfflineLoader())
		tiktokenEncoder, tiktokenErr = tiktoken.GetEncoding(tiktokenEncodingName)
	})
	return tiktokenEncoder, tiktokenErr
}

func estimateClaudeRequestTikTokenInputTokens(req *ClaudeRequest) (int, bool) {
	rawTokens, ok := estimateClaudeRequestRawTikTokenInputTokens(req)
	if !ok {
		return 0, false
	}
	return applyClaudeTikTokenCorrection(rawTokens), true
}

func estimateClaudeRequestRawTikTokenInputTokens(req *ClaudeRequest) (int, bool) {
	if req == nil {
		return 0, true
	}

	systemTokens, ok := estimateClaudeValueTikTokenTokens(req.System)
	if !ok {
		return 0, false
	}

	messageTokens := 0
	for _, msg := range req.Messages {
		messageTokens += 4
		roleTokens, ok := estimateTikTokenTextTokens(msg.Role)
		if !ok {
			return 0, false
		}
		contentTokens, ok := estimateClaudeValueTikTokenTokens(msg.Content)
		if !ok {
			return 0, false
		}
		messageTokens += roleTokens + contentTokens
	}
	if len(req.Messages) > 0 {
		messageTokens += 3
	}

	toolTokens := 0
	for _, tool := range req.Tools {
		toolTokens += 4
		nameTokens, ok := estimateTikTokenTextTokens(tool.Name)
		if !ok {
			return 0, false
		}
		descTokens, ok := estimateTikTokenTextTokens(tool.Description)
		if !ok {
			return 0, false
		}
		schemaTokens, ok := estimateJSONTikTokenTokens(tool.InputSchema)
		if !ok {
			return 0, false
		}
		toolTokens += nameTokens + descTokens + schemaTokens
	}

	return systemTokens + messageTokens + toolTokens, true
}

func applyClaudeTikTokenCorrection(tokens int) int {
	return int(float64(tokens) * claudeTokenCorrectionFactor)
}

func estimateClaudeValueTikTokenTokens(v interface{}) (int, bool) {
	switch value := v.(type) {
	case nil:
		return 0, true
	case string:
		return estimateTikTokenTextTokens(value)
	case []interface{}:
		total := 0
		for _, part := range value {
			n, ok := estimateClaudeValueTikTokenTokens(part)
			if !ok {
				return 0, false
			}
			total += n
		}
		return total, true
	case map[string]interface{}:
		typeName, _ := value["type"].(string)
		switch typeName {
		case "text":
			if text, ok := value["text"].(string); ok {
				return estimateTikTokenTextTokens(text)
			}
		case "thinking":
			if thinking, ok := value["thinking"].(string); ok {
				return estimateTikTokenTextTokens(thinking)
			}
		case "tool_use":
			total := 0
			if id, ok := value["id"].(string); ok {
				n, ok := estimateTikTokenTextTokens(id)
				if !ok {
					return 0, false
				}
				total += n
			}
			if name, ok := value["name"].(string); ok {
				n, ok := estimateTikTokenTextTokens(name)
				if !ok {
					return 0, false
				}
				total += n
			}
			if input, ok := value["input"]; ok {
				n, ok := estimateJSONTikTokenTokens(input)
				if !ok {
					return 0, false
				}
				total += n
			}
			if total > 0 {
				return total, true
			}
		case "tool_result":
			total := 0
			if toolUseID, ok := value["tool_use_id"].(string); ok {
				n, ok := estimateTikTokenTextTokens(toolUseID)
				if !ok {
					return 0, false
				}
				total += n
			}
			if isError, ok := value["is_error"]; ok {
				n, ok := estimateTikTokenTextTokens(stringifyTikTokenValue(isError))
				if !ok {
					return 0, false
				}
				total += n
			}
			if content, ok := value["content"]; ok {
				n, ok := estimateClaudeValueTikTokenTokens(content)
				if !ok {
					return 0, false
				}
				total += n
			}
			if total > 0 {
				return total, true
			}
		case "image", "image_url", "input_image":
			return 100, true
		case "document", "input_file", "file":
			return approxDocumentInputTokens, true
		}

		total := 0
		if text, ok := value["text"].(string); ok {
			n, ok := estimateTikTokenTextTokens(text)
			if !ok {
				return 0, false
			}
			total += n
		}
		if thinking, ok := value["thinking"].(string); ok {
			n, ok := estimateTikTokenTextTokens(thinking)
			if !ok {
				return 0, false
			}
			total += n
		}
		if content, ok := value["content"]; ok {
			n, ok := estimateClaudeValueTikTokenTokens(content)
			if !ok {
				return 0, false
			}
			total += n
		}
		if total > 0 {
			return total, true
		}

		return estimateJSONTikTokenTokens(value)
	default:
		return estimateJSONTikTokenTokens(value)
	}
}

func estimateJSONTikTokenTokens(v interface{}) (int, bool) {
	if v == nil {
		return 0, true
	}

	b, err := json.Marshal(v)
	if err != nil {
		return 0, false
	}

	return estimateTikTokenTextTokens(string(b))
}

func estimateTikTokenTextTokens(text string) (int, bool) {
	if text == "" {
		return 0, true
	}
	encoder, err := getTikTokenEncoder()
	if err != nil {
		return 0, false
	}
	return len(encoder.EncodeOrdinary(text)), true
}

func stringifyTikTokenValue(v interface{}) string {
	switch value := v.(type) {
	case string:
		return value
	case bool:
		if value {
			return "true"
		}
		return "false"
	default:
		b, err := json.Marshal(value)
		if err != nil {
			return ""
		}
		return string(b)
	}
}
