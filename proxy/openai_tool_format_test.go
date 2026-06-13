package proxy

import (
	"encoding/json"
	"testing"
)

// TestOpenAIToolAcceptsResponsesFlatFormat verifies that the Responses API tool
// shape (name/description/parameters at the top level) is parsed correctly, not
// just the Chat Completions nested {"function":{...}} shape. Previously the flat
// form produced an empty Function.Name, which Kiro rejected with HTTP 400
// "Improperly formed request".
func TestOpenAIToolAcceptsResponsesFlatFormat(t *testing.T) {
	flat := `{"type":"function","name":"exec_command","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}`
	var tool OpenAITool
	if err := json.Unmarshal([]byte(flat), &tool); err != nil {
		t.Fatalf("unmarshal flat tool: %v", err)
	}
	if tool.Function.Name != "exec_command" {
		t.Fatalf("expected name exec_command, got %q", tool.Function.Name)
	}
	if tool.Function.Description != "Run a shell command" {
		t.Fatalf("expected description preserved, got %q", tool.Function.Description)
	}
	if tool.Function.Parameters == nil {
		t.Fatalf("expected parameters preserved")
	}
}

// TestOpenAIToolAcceptsNestedFormat verifies the Chat Completions nested shape
// still works after adding flat-format support.
func TestOpenAIToolAcceptsNestedFormat(t *testing.T) {
	nested := `{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object"}}}`
	var tool OpenAITool
	if err := json.Unmarshal([]byte(nested), &tool); err != nil {
		t.Fatalf("unmarshal nested tool: %v", err)
	}
	if tool.Function.Name != "get_weather" {
		t.Fatalf("expected name get_weather, got %q", tool.Function.Name)
	}
}

// TestConvertOpenAIToolsEmitsNonEmptyNames ensures the converter never emits a
// tool spec with an empty name (Kiro rejects those) and records original names
// so client tool registries still match returned function calls.
func TestConvertOpenAIToolsEmitsNonEmptyNames(t *testing.T) {
	tools := []OpenAITool{
		mustTool(t, `{"type":"function","name":"exec_command","parameters":{"type":"object"}}`),
		mustTool(t, `{"type":"function","function":{"name":"functions.update-plan","parameters":{"type":"object"}}}`),
		mustTool(t, `{"type":"function","function":{"name":"123_search","parameters":{"type":"object"}}}`),
	}
	wrappers, nameMap := convertOpenAITools(tools)
	if len(wrappers) != 3 {
		t.Fatalf("expected 3 tool wrappers, got %d", len(wrappers))
	}
	for i, w := range wrappers {
		if w.ToolSpecification.Name == "" {
			t.Fatalf("tool %d has empty name", i)
		}
	}
	if wrappers[0].ToolSpecification.Name != "execCommand" {
		t.Fatalf("expected exec_command sanitized for Kiro, got %q", wrappers[0].ToolSpecification.Name)
	}
	if wrappers[1].ToolSpecification.Name != "functionsUpdatePlan" {
		t.Fatalf("expected dotted function name sanitized for Kiro, got %q", wrappers[1].ToolSpecification.Name)
	}
	if wrappers[2].ToolSpecification.Name != "tool123Search" {
		t.Fatalf("expected digit-prefixed name to receive tool prefix, got %q", wrappers[2].ToolSpecification.Name)
	}
	if nameMap["execCommand"] != "exec_command" ||
		nameMap["functionsUpdatePlan"] != "functions.update-plan" ||
		nameMap["tool123Search"] != "123_search" {
		t.Fatalf("expected original tool name mapping, got %#v", nameMap)
	}
}

func mustTool(t *testing.T, s string) OpenAITool {
	t.Helper()
	var tool OpenAITool
	if err := json.Unmarshal([]byte(s), &tool); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return tool
}
