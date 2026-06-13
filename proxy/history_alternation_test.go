package proxy

import "testing"

// TestEnforceHistoryAlternation verifies the final alternation guarantee: no two
// adjacent history turns may share a role, regardless of how they arose. Upstream
// rejects any consecutive same-role pair with HTTP 400 "Improperly formed
// request", so this is the hard invariant the converter must always satisfy.
func TestEnforceHistoryAlternation(t *testing.T) {
	cases := []struct {
		name string
		in   []KiroHistoryMessage
	}{
		{
			name: "two plain assistant turns adjacent",
			in: []KiroHistoryMessage{
				{UserInputMessage: &KiroUserInputMessage{Content: "hi"}},
				{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "a1"}},
				{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "a2"}},
			},
		},
		{
			name: "plain assistant before active tool turn",
			in: []KiroHistoryMessage{
				{UserInputMessage: &KiroUserInputMessage{Content: "hi"}},
				{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "thinking out loud"}},
				{AssistantResponseMessage: &KiroAssistantResponseMessage{
					Content:  "calling tool",
					ToolUses: []KiroToolUse{{ToolUseID: "t1", Name: "execCommand"}},
				}},
			},
		},
		{
			name: "two user turns one with images",
			in: []KiroHistoryMessage{
				{UserInputMessage: &KiroUserInputMessage{Content: "first"}},
				{UserInputMessage: &KiroUserInputMessage{Content: "second", Images: []KiroImage{{Format: "png"}}}},
				{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "ok"}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := enforceHistoryAlternation(tc.in)
			for i := 1; i < len(out); i++ {
				if historyRole(out[i-1]) == historyRole(out[i]) {
					t.Fatalf("adjacent same-role turns at %d/%d (role=%s)", i-1, i, historyRole(out[i]))
				}
			}
		})
	}
}

// TestEnforceHistoryAlternationKeepsActiveToolUses ensures merging a plain-text
// assistant turn into an adjacent active tool turn preserves the structured
// toolUses (so current-message tool-result pairing survives).
func TestEnforceHistoryAlternationKeepsActiveToolUses(t *testing.T) {
	in := []KiroHistoryMessage{
		{UserInputMessage: &KiroUserInputMessage{Content: "hi"}},
		{AssistantResponseMessage: &KiroAssistantResponseMessage{Content: "preface"}},
		{AssistantResponseMessage: &KiroAssistantResponseMessage{
			ToolUses: []KiroToolUse{{ToolUseID: "t1", Name: "execCommand"}},
		}},
	}
	out := enforceHistoryAlternation(in)
	last := out[len(out)-1].AssistantResponseMessage
	if last == nil || len(last.ToolUses) != 1 || last.ToolUses[0].ToolUseID != "t1" {
		t.Fatalf("active tool use lost after alternation merge: %+v", out)
	}
}
