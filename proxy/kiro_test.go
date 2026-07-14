package proxy

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"kiro-go/config"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"
)

func TestParseEventStreamPreservesRepeatedDeltaCharacters(t *testing.T) {
	tests := []struct {
		name   string
		chunks []string
		want   string
	}{
		{
			name:   "repeated chinese character",
			chunks: []string{"谢", "谢", "配合"},
			want:   "谢谢配合",
		},
		{
			name:   "repeated chinese character at chunk boundary",
			chunks: []string{"谢", "谢配合"},
			want:   "谢谢配合",
		},
		{
			name:   "repeated latin character across chunks",
			chunks: []string{"deepse", "ek"},
			want:   "deepseek",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var frames [][]byte
			for _, chunk := range tt.chunks {
				frames = append(frames, awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
					"content": chunk,
				}))
			}

			var got strings.Builder
			err := parseEventStream(bytes.NewReader(bytes.Join(frames, nil)), &KiroStreamCallback{
				OnText: func(text string, isThinking bool) {
					if isThinking {
						t.Fatalf("unexpected thinking text %q", text)
					}
					got.WriteString(text)
				},
			})
			if err != nil {
				t.Fatalf("unexpected parse error: %v", err)
			}
			if got.String() != tt.want {
				t.Fatalf("expected preserved repeated text %q, got %q", tt.want, got.String())
			}
		})
	}
}

func TestParseEventStreamFinishesPendingToolUseOnEOF(t *testing.T) {
	stream := bytes.NewReader(awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
		"toolUseId": "toolu_1",
		"name":      "mcpIdaProMcpStatus",
		"input":     `{"server":"ida-pro-mcp"}`,
	}))

	var toolUses []KiroToolUse
	var completed bool
	err := parseEventStream(stream, &KiroStreamCallback{
		OnToolUse: func(toolUse KiroToolUse) {
			toolUses = append(toolUses, toolUse)
		},
		OnComplete: func(_, _ int) {
			completed = true
		},
	})
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if !completed {
		t.Fatalf("expected stream completion callback")
	}
	if len(toolUses) != 1 {
		t.Fatalf("expected pending tool use to be emitted on EOF, got %d", len(toolUses))
	}
	if toolUses[0].ToolUseID != "toolu_1" || toolUses[0].Name != "mcpIdaProMcpStatus" {
		t.Fatalf("unexpected tool use: %#v", toolUses[0])
	}
	if got := toolUses[0].Input["server"]; got != "ida-pro-mcp" {
		t.Fatalf("expected parsed tool input, got %#v", toolUses[0].Input)
	}
}

func TestParseEventStreamNilCallbackIsNoOp(t *testing.T) {
	stream := bytes.NewReader(bytes.Join([][]byte{
		awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "hello"}),
		awsEventStreamFrame(t, "reasoningContentEvent", map[string]interface{}{"text": "thinking"}),
		awsEventStreamFrame(t, "contextUsageEvent", map[string]interface{}{"contextUsagePercentage": 12.5}),
		awsEventStreamFrame(t, "meteringEvent", map[string]interface{}{"usage": 1.25}),
		awsEventStreamFrame(t, "toolUseEvent", map[string]interface{}{
			"name":  "mcpIdaProMcpStatus",
			"input": `{"server":"ida-pro-mcp"}`,
			"stop":  true,
		}),
	}, nil))

	if err := parseEventStream(stream, nil); err != nil {
		t.Fatalf("expected nil callback to be a no-op, got %v", err)
	}
}

func TestParseEventStreamNilCallbackFieldsAreNoOp(t *testing.T) {
	stream := bytes.NewReader(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
		"content": "hello",
	}))

	if err := parseEventStream(stream, &KiroStreamCallback{}); err != nil {
		t.Fatalf("expected empty callback to be a no-op, got %v", err)
	}
}

func TestParseEventStreamRequiresCompletionSignalAfterAssistantText(t *testing.T) {
	stream := bytes.NewReader(awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{
		"content": "partial response",
	}))

	var completed bool
	err := parseEventStream(stream, &KiroStreamCallback{
		RequireCompletionSignal: true,
		OnComplete: func(_, _ int) {
			completed = true
		},
	})
	if !errors.Is(err, errKiroStreamIncomplete) {
		t.Fatalf("expected incomplete stream error, got %v", err)
	}
	if completed {
		t.Fatalf("incomplete stream must not call OnComplete")
	}
}

func TestParseEventStreamAllowsCompletionSignalAfterAssistantText(t *testing.T) {
	stream := bytes.NewReader(bytes.Join([][]byte{
		awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "complete response"}),
		awsEventStreamFrame(t, "contextUsageEvent", map[string]interface{}{"contextUsagePercentage": 10.0}),
	}, nil))

	if err := parseEventStream(stream, &KiroStreamCallback{RequireCompletionSignal: true}); err != nil {
		t.Fatalf("expected completed stream, got %v", err)
	}
}

func TestParseEventStreamRequiresCompletionSignalAfterTextOrder(t *testing.T) {
	stream := bytes.NewReader(bytes.Join([][]byte{
		awsEventStreamFrame(t, "contextUsageEvent", map[string]interface{}{"contextUsagePercentage": 10.0}),
		awsEventStreamFrame(t, "assistantResponseEvent", map[string]interface{}{"content": "partial response"}),
	}, nil))

	err := parseEventStream(stream, &KiroStreamCallback{RequireCompletionSignal: true})
	if !errors.Is(err, errKiroStreamIncomplete) {
		t.Fatalf("expected incomplete stream error when terminal metadata precedes text, got %v", err)
	}
}

func TestParseEventStreamReturnsErrorForExceptionFrame(t *testing.T) {
	payload, err := json.Marshal(map[string]interface{}{"message": "upstream exploded"})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	stream := bytes.NewReader(awsEventStreamFrameWithHeaders(t, map[string]string{
		":message-type":   "exception",
		":exception-type": "InternalFailure",
	}, payload))

	err = parseEventStream(stream, &KiroStreamCallback{})
	if err == nil {
		t.Fatalf("expected exception frame to return an error")
	}
	if got := err.Error(); !strings.Contains(got, "InternalFailure") || !strings.Contains(got, "upstream exploded") {
		t.Fatalf("unexpected exception error: %v", err)
	}
}

func TestHandleToolUseEventGeneratesMissingToolUseID(t *testing.T) {
	var toolUses []KiroToolUse
	current := handleToolUseEvent(map[string]interface{}{
		"name":  "mcpIdaProMcpStatus",
		"input": `{"server":"ida-pro-mcp"}`,
		"stop":  true,
	}, nil, &KiroStreamCallback{
		OnToolUse: func(toolUse KiroToolUse) {
			toolUses = append(toolUses, toolUse)
		},
	})

	if current != nil {
		t.Fatalf("expected stopped tool use to clear current state")
	}
	if len(toolUses) != 1 {
		t.Fatalf("expected one tool use, got %d", len(toolUses))
	}
	if toolUses[0].ToolUseID == "" {
		t.Fatalf("expected generated tool use id")
	}
	if toolUses[0].Name != "mcpIdaProMcpStatus" {
		t.Fatalf("unexpected tool name: %q", toolUses[0].Name)
	}
}

func TestHandleToolUseEventReplacesGeneratedIDWhenRealIDArrives(t *testing.T) {
	var toolUses []KiroToolUse
	callback := &KiroStreamCallback{
		OnToolUse: func(toolUse KiroToolUse) {
			toolUses = append(toolUses, toolUse)
		},
	}

	current := handleToolUseEvent(map[string]interface{}{
		"name":  "mcpIdaProMcpStatus",
		"input": `{"server":`,
	}, nil, callback)
	current = handleToolUseEvent(map[string]interface{}{
		"toolUseId": "toolu_real",
		"name":      "mcpIdaProMcpStatus",
		"input":     `"ida-pro-mcp"}`,
		"stop":      true,
	}, current, callback)

	if current != nil {
		t.Fatalf("expected stopped tool use to clear current state")
	}
	if len(toolUses) != 1 {
		t.Fatalf("expected one completed tool use, got %d", len(toolUses))
	}
	if toolUses[0].ToolUseID != "toolu_real" {
		t.Fatalf("expected real tool id to replace generated id, got %q", toolUses[0].ToolUseID)
	}
	if got := toolUses[0].Input["server"]; got != "ida-pro-mcp" {
		t.Fatalf("expected joined tool input, got %#v", toolUses[0].Input)
	}
}

func TestBuildKiroTransportUsesExplicitProxyURL(t *testing.T) {
	transport := buildKiroTransport("http://proxy.local:8080")
	req := &http.Request{URL: mustParseURL(t, "https://q.us-east-1.amazonaws.com")}

	got, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("unexpected proxy error: %v", err)
	}
	assertProxyURL(t, got, "http://proxy.local:8080")
}

func TestBuildKiroTransportFallsBackToEnvironmentProxy(t *testing.T) {
	t.Setenv("HTTPS_PROXY", "http://env-proxy.local:2323")
	t.Setenv("NO_PROXY", "")
	t.Setenv("no_proxy", "")

	transport := buildKiroTransport("")
	req := &http.Request{URL: mustParseURL(t, "https://q.us-east-1.amazonaws.com")}

	got, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("unexpected proxy error: %v", err)
	}
	assertProxyURL(t, got, "http://env-proxy.local:2323")
}

func TestInitKiroHttpClientKeepsShortRestTimeout(t *testing.T) {
	InitKiroHttpClient("")
	t.Cleanup(func() { InitKiroHttpClient("") })

	streamClient := kiroHttpStore.Load()
	restClient := kiroRestHttpStore.Load()

	if streamClient.Timeout != 20*time.Minute {
		t.Fatalf("expected streaming timeout to be 20m, got %s", streamClient.Timeout)
	}
	if restClient.Timeout != 30*time.Second {
		t.Fatalf("expected REST timeout to stay 30s, got %s", restClient.Timeout)
	}
}

func TestSetPayloadProfileArnForAccountUsesAccountArn(t *testing.T) {
	payload := &KiroPayload{ProfileArn: "arn:aws:codewhisperer:profile/stale"}

	setPayloadProfileArnForAccount(payload, &config.Account{ProfileArn: " arn:aws:codewhisperer:profile/current "})
	if payload.ProfileArn != "arn:aws:codewhisperer:profile/current" {
		t.Fatalf("expected current account profile ARN, got %q", payload.ProfileArn)
	}
}

func TestSetPayloadProfileArnForAccountPreservesExplicitPayloadArn(t *testing.T) {
	payload := &KiroPayload{ProfileArn: " arn:aws:codewhisperer:profile/explicit "}

	setPayloadProfileArnForAccount(payload, &config.Account{})
	if payload.ProfileArn != "arn:aws:codewhisperer:profile/explicit" {
		t.Fatalf("expected explicit payload profile ARN to be preserved, got %q", payload.ProfileArn)
	}
}

func mustParseURL(t *testing.T, raw string) *url.URL {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("invalid test URL: %v", err)
	}
	return parsed
}

func assertProxyURL(t *testing.T, got *url.URL, want string) {
	t.Helper()
	if got == nil {
		t.Fatalf("expected proxy URL %q, got nil", want)
	}
	if got.String() != want {
		t.Fatalf("expected proxy URL %q, got %q", want, got.String())
	}
}

func awsEventStreamFrame(t *testing.T, eventType string, payload map[string]interface{}) []byte {
	t.Helper()

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	return awsEventStreamFrameWithHeaders(t, map[string]string{":event-type": eventType}, payloadBytes)
}

func awsEventStreamFrameWithHeaders(t *testing.T, headerValues map[string]string, payloadBytes []byte) []byte {
	t.Helper()

	var headers []byte
	for name, value := range headerValues {
		headerName := []byte(name)
		headerValue := []byte(value)
		headers = append(headers, byte(len(headerName)))
		headers = append(headers, headerName...)
		headers = append(headers, byte(7))
		headers = append(headers, byte(len(headerValue)>>8), byte(len(headerValue)))
		headers = append(headers, headerValue...)
	}

	totalLength := 12 + len(headers) + len(payloadBytes) + 4
	frame := make([]byte, 12, totalLength)
	binary.BigEndian.PutUint32(frame[0:4], uint32(totalLength))
	binary.BigEndian.PutUint32(frame[4:8], uint32(len(headers)))
	frame = append(frame, headers...)
	frame = append(frame, payloadBytes...)
	frame = append(frame, 0, 0, 0, 0)
	return frame
}
