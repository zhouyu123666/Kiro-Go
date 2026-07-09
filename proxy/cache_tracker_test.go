package proxy

import (
	"strings"
	"testing"
	"time"
)

func TestPromptCacheTrackerComputeAndUpdate(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	longSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)
	req := &ClaudeRequest{
		Model: "claude-sonnet-4.5",
		System: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": longSystem,
				"cache_control": map[string]interface{}{
					"type": "ephemeral",
				},
			},
		},
		Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
	}

	profile := tracker.BuildClaudeProfile(req, 120)
	if profile == nil {
		t.Fatalf("expected cache profile to be built")
	}

	first := tracker.Compute("acct-1", profile)
	if first.CacheCreationInputTokens <= 0 {
		t.Fatalf("expected first request to create cache tokens, got %+v", first)
	}
	if first.CacheReadInputTokens != 0 {
		t.Fatalf("expected first request to have zero cache reads, got %+v", first)
	}

	tracker.Update("acct-1", profile)
	second := tracker.Compute("acct-1", profile)
	if second.CacheReadInputTokens <= 0 {
		t.Fatalf("expected repeated request to read cache tokens, got %+v", second)
	}
	if second.CacheCreationInputTokens != 0 {
		t.Fatalf("expected repeated request to avoid cache creation, got %+v", second)
	}
}

func TestBuildClaudeUsageMapIncludesCacheFields(t *testing.T) {
	usage := promptCacheUsage{
		CacheCreationInputTokens:   30,
		CacheReadInputTokens:       20,
		CacheCreation5mInputTokens: 10,
		CacheCreation1hInputTokens: 20,
	}

	m := buildClaudeUsageMap(100, 50, usage, true)

	if got := m["input_tokens"]; got != 50 {
		t.Fatalf("expected displayed uncached input tokens 50, got %#v", got)
	}
	if got := m["cache_creation_input_tokens"]; got != 30 {
		t.Fatalf("expected cache creation tokens 30, got %#v", got)
	}
	if got := m["cache_read_input_tokens"]; got != 20 {
		t.Fatalf("expected cache read tokens 20, got %#v", got)
	}
	if m["input_tokens"].(int)+m["cache_creation_input_tokens"].(int)+m["cache_read_input_tokens"].(int) != 100 {
		t.Fatalf("expected usage fields to reconstruct total input 100, got %#v", m)
	}
	creation, ok := m["cache_creation"].(map[string]int)
	if !ok {
		t.Fatalf("expected typed cache creation map, got %#v", m["cache_creation"])
	}
	if creation["ephemeral_5m_input_tokens"] != 10 || creation["ephemeral_1h_input_tokens"] != 20 {
		t.Fatalf("unexpected ttl breakdown: %#v", creation)
	}
}

// TestPromptCacheStableAcrossBillingHeaderDrift verifies that Claude Code's
// per-request "x-anthropic-billing-header: cc_version=...; cch=...;" system
// block (whose content drifts on every request) does not break cache hits.
// The tracker should ignore that metadata when fingerprinting cached prefixes.
func TestPromptCacheStableAcrossBillingHeaderDrift(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	mainSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	build := func(billingHdr string) *ClaudeRequest {
		return &ClaudeRequest{
			Model: "claude-sonnet-4.5",
			System: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": billingHdr,
				},
				map[string]interface{}{
					"type": "text",
					"text": mainSystem,
					"cache_control": map[string]interface{}{
						"type": "ephemeral",
					},
				},
			},
			Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
		}
	}

	req1 := build("x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;")
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	first := tracker.Compute("acct-1", profile1)
	if first.CacheReadInputTokens != 0 {
		t.Fatalf("expected no cache read on first request, got %+v", first)
	}
	tracker.Update("acct-1", profile1)

	req2 := build("x-anthropic-billing-header: cc_version=2.1.87.42; cch=bbbb; padding=xxyyzz;")
	profile2 := tracker.BuildClaudeProfile(req2, 2048)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	second := tracker.Compute("acct-1", profile2)
	if second.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read after billing header drift, got %+v", second)
	}
}

func TestPromptCacheStableWhenBillingHeaderAppearsOrDisappears(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	mainSystem := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	build := func(includeBilling bool) *ClaudeRequest {
		system := []interface{}{}
		if includeBilling {
			system = append(system, map[string]interface{}{
				"type": "text",
				"text": "x-anthropic-billing-header: cc_version=2.1.87.1; cch=aaaa;",
			})
		}
		system = append(system, map[string]interface{}{
			"type": "text",
			"text": mainSystem,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		})
		return &ClaudeRequest{
			Model:    "claude-sonnet-4.5",
			System:   system,
			Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
		}
	}

	withBilling := tracker.BuildClaudeProfile(build(true), 2048)
	if withBilling == nil {
		t.Fatalf("profile with billing header should be built")
	}
	tracker.Update("acct-1", withBilling)

	withoutBilling := tracker.BuildClaudeProfile(build(false), 2048)
	if withoutBilling == nil {
		t.Fatalf("profile without billing header should be built")
	}
	result := tracker.Compute("acct-1", withoutBilling)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read when billing header disappears, got %+v", result)
	}
}

func TestCanonicalCacheValueIgnoresPositionKeys(t *testing.T) {
	first := canonicalizeCacheValue(stripCachePositionKeys(map[string]interface{}{
		"kind":         "system",
		"system_index": 0,
		"block": map[string]interface{}{
			"type": "text",
			"text": "stable",
		},
	}))
	second := canonicalizeCacheValue(stripCachePositionKeys(map[string]interface{}{
		"kind":         "system",
		"system_index": 1,
		"block": map[string]interface{}{
			"type": "text",
			"text": "stable",
		},
	}))
	if first != second {
		t.Fatalf("expected position keys to be ignored, got %q vs %q", first, second)
	}
}

func TestCanonicalCacheValuePreservesSemanticPositionKeys(t *testing.T) {
	first := canonicalizeCacheValue(map[string]interface{}{
		"kind": "system",
		"block": map[string]interface{}{
			"type":        "text",
			"text":        "stable",
			"block_index": 1,
		},
	})
	second := canonicalizeCacheValue(map[string]interface{}{
		"kind": "system",
		"block": map[string]interface{}{
			"type":        "text",
			"text":        "stable",
			"block_index": 2,
		},
	})
	if first == second {
		t.Fatalf("expected semantic block_index fields to remain fingerprinted")
	}
}

// TestPromptCacheImplicitBreakpointAtMessageEnd verifies that once any
// explicit cache_control breakpoint has been seen, subsequent message-end
// boundaries act as implicit breakpoints. This allows multi-turn conversations
// to hit earlier stored prefix fingerprints even when the newest messages
// lack explicit cache_control.
func TestPromptCacheImplicitBreakpointAtMessageEnd(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go, Rust, Python, and TypeScript. ", 80)

	baseSystem := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": systemText,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	// Round 1: single user message.
	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: "question one"}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	tracker.Update("acct-1", profile1)

	// Round 2: conversation continues with new messages. The latest user
	// message has no explicit cache_control; it should still hit the stored
	// prefix via the implicit message-end breakpoint.
	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: "question one"},
			{Role: "assistant", Content: "answer one"},
			{Role: "user", Content: "follow-up question"},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	result := tracker.Compute("acct-1", profile2)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read via implicit message-end breakpoint, got %+v", result)
	}
}

// TestPromptCacheTailMessageStaysVisible reproduces the "every request shows
// input_tokens=6" bug: Claude Code puts cache_control on system/tools, which
// turned every later message-end (including the brand-new user message) into an
// implicit breakpoint, so the fresh tail was absorbed into the cacheable prefix
// and visible uncached input collapsed to the envelope floor. After the fix the
// new tail must remain visible as genuine uncached input_tokens.
func TestPromptCacheTailMessageStaysVisible(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go. ", 120)
	baseSystem := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": systemText,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	// Round 1: prime the cache with the stable system prefix.
	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: "first question"}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 4096)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	tracker.Update("acct-1", profile1)

	// Round 2: same cached prefix plus a substantial NEW user question.
	newQuestion := strings.Repeat("please explain this fresh follow-up in detail. ", 40)
	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: "first question"},
			{Role: "assistant", Content: "the first answer"},
			{Role: "user", Content: newQuestion},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	usage := tracker.Compute("acct-1", profile2)
	if usage.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read on stable prefix, got %+v", usage)
	}

	// The new tail must not be swallowed by the cache: visible input has to
	// exceed the envelope floor of 6.
	m := buildClaudeUsageMap(profile2.TotalInputTokens, 50, usage, true)
	visible, ok := m["input_tokens"].(int)
	if !ok {
		t.Fatalf("input_tokens missing or wrong type: %v", m["input_tokens"])
	}
	if visible <= claudeUsageEnvelopeMinTokens {
		t.Fatalf("expected visible input above envelope floor %d, got %d (tail absorbed into cache)", claudeUsageEnvelopeMinTokens, visible)
	}

	// Sanity: the public accounting identity still holds.
	created, _ := m["cache_creation_input_tokens"].(int)
	read, _ := m["cache_read_input_tokens"].(int)
	if visible+created+read != profile2.TotalInputTokens {
		t.Fatalf("usage identity broken: %d + %d + %d != %d", visible, created, read, profile2.TotalInputTokens)
	}
}

// TestPromptCacheExplicitCacheControlOnTailStaysVisible reproduces the
// production symptom where usage.input_tokens collapsed to the envelope floor
// (6) on almost every request. Claude Code marks the newest message block with
// explicit cache_control to prime the cache for the next turn. When that
// explicit marker on the final block was honored, the last breakpoint's
// cumulative tokens equaled the full prompt total, so CacheCoveredEstimate ==
// PromptTotalEstimate, the coverage ratio hit ~1.0, and the fresh tail was
// swallowed into cache_creation. The final block must never become a
// breakpoint, even with explicit cache_control, so the new tail stays visible.
func TestPromptCacheExplicitCacheControlOnTailStaysVisible(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go. ", 120)
	baseSystem := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": systemText,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	// Round 1: prime the cache with the stable system prefix.
	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: "first question"}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 4096)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	tracker.Update("acct-1", profile1)

	// Round 2: same cached prefix plus a substantial NEW user message that
	// carries an explicit cache_control marker on its (final) content block,
	// exactly as Claude Code sends it.
	newQuestion := strings.Repeat("please explain this fresh follow-up in detail. ", 40)
	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: "first question"},
			{Role: "assistant", Content: "the first answer"},
			{Role: "user", Content: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": newQuestion,
					"cache_control": map[string]interface{}{
						"type": "ephemeral",
					},
				},
			}},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	usage := tracker.Compute("acct-1", profile2)
	if usage.CacheReadInputTokens == 0 {
		t.Fatalf("expected cache read on stable prefix, got %+v", usage)
	}

	// The tail carrying explicit cache_control must NOT be counted as covered:
	// the fresh input has to remain visible above the envelope floor.
	m := buildClaudeUsageMap(profile2.TotalInputTokens, 50, usage, true)
	visible, ok := m["input_tokens"].(int)
	if !ok {
		t.Fatalf("input_tokens missing or wrong type: %v", m["input_tokens"])
	}
	if visible <= claudeUsageEnvelopeMinTokens {
		t.Fatalf("expected visible input above envelope floor %d, got %d (explicit tail marker absorbed into cache)", claudeUsageEnvelopeMinTokens, visible)
	}

	created, _ := m["cache_creation_input_tokens"].(int)
	read, _ := m["cache_read_input_tokens"].(int)
	if visible+created+read != profile2.TotalInputTokens {
		t.Fatalf("usage identity broken: %d + %d + %d != %d", visible, created, read, profile2.TotalInputTokens)
	}
}

func TestPromptCacheFinalBlockStoredForNextTurn(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	systemText := strings.Repeat("You are a helpful coding assistant with deep knowledge of Go. ", 120)
	firstQuestion := strings.Repeat("first question with enough stable detail. ", 40)
	baseSystem := []interface{}{
		map[string]interface{}{
			"type": "text",
			"text": systemText,
			"cache_control": map[string]interface{}{
				"type": "ephemeral",
			},
		},
	}

	req1 := &ClaudeRequest{
		Model:    "claude-sonnet-4.5",
		System:   baseSystem,
		Messages: []ClaudeMessage{{Role: "user", Content: firstQuestion}},
	}
	profile1 := tracker.BuildClaudeProfile(req1, 4096)
	if profile1 == nil {
		t.Fatalf("profile1 should be built")
	}
	if len(profile1.Breakpoints) == 0 || len(profile1.StorageBreakpoints) <= len(profile1.Breakpoints) {
		t.Fatalf("expected final block to be stored separately from usage breakpoints: %+v", profile1)
	}
	lastUsageTokens := profile1.Breakpoints[len(profile1.Breakpoints)-1].CumulativeTokens
	finalStoredTokens := profile1.StorageBreakpoints[len(profile1.StorageBreakpoints)-1].CumulativeTokens
	if finalStoredTokens <= lastUsageTokens {
		t.Fatalf("expected final stored breakpoint %d to extend beyond usage breakpoint %d", finalStoredTokens, lastUsageTokens)
	}
	tracker.Update("acct-1", profile1)

	repeated := tracker.Compute("acct-1", profile1)
	if repeated.CacheReadInputTokens <= lastUsageTokens {
		t.Fatalf("expected repeated request to read the stored final block, got usage=%+v last_usage_tokens=%d", repeated, lastUsageTokens)
	}

	req2 := &ClaudeRequest{
		Model:  "claude-sonnet-4.5",
		System: baseSystem,
		Messages: []ClaudeMessage{
			{Role: "user", Content: firstQuestion},
			{Role: "assistant", Content: "the first answer"},
			{Role: "user", Content: strings.Repeat("fresh second question. ", 40)},
		},
	}
	profile2 := tracker.BuildClaudeProfile(req2, 4096)
	if profile2 == nil {
		t.Fatalf("profile2 should be built")
	}
	usage := tracker.Compute("acct-1", profile2)
	if usage.CacheReadInputTokens <= lastUsageTokens {
		t.Fatalf("expected next turn to read through prior final block, got usage=%+v last_usage_tokens=%d", usage, lastUsageTokens)
	}
	if usage.CacheCoveredEstimate >= profile2.TotalInputTokens {
		t.Fatalf("expected current final block to stay out of current-turn coverage, got usage=%+v total=%d", usage, profile2.TotalInputTokens)
	}
}

func TestPromptCacheSkipsDynamicSystemPreludeBeforeFirstExplicitCacheBlock(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	stableSystem := strings.Repeat("stable cacheable instructions ", 320)

	build := func(dynamic string) *ClaudeRequest {
		return &ClaudeRequest{
			Model: "claude-sonnet-4.5",
			System: []interface{}{
				map[string]interface{}{
					"type": "text",
					"text": dynamic,
				},
				map[string]interface{}{
					"type": "text",
					"text": stableSystem,
					"cache_control": map[string]interface{}{
						"type": "ephemeral",
					},
				},
			},
			Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
		}
	}

	req1 := build("current time: 2026-07-07T11:00:00Z")
	profile1 := tracker.BuildClaudeProfile(req1, 2048)
	if profile1 == nil {
		t.Fatalf("expected profile1 to be built")
	}
	tracker.Update("acct-1", profile1)

	req2 := build("current time: 2026-07-07T11:05:00Z")
	profile2 := tracker.BuildClaudeProfile(req2, 2048)
	if profile2 == nil {
		t.Fatalf("expected profile2 to be built")
	}
	result := tracker.Compute("acct-1", profile2)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected stable suffix to hit cache despite dynamic prelude drift, got %+v", result)
	}
}

func TestPromptCacheHitDoesNotExtendExpiry(t *testing.T) {
	tracker := newPromptCacheTracker(time.Hour)
	longSystem := strings.Repeat("stable cacheable instructions ", 320)
	req := &ClaudeRequest{
		Model: "claude-sonnet-4.5",
		System: []interface{}{
			map[string]interface{}{
				"type": "text",
				"text": longSystem,
				"cache_control": map[string]interface{}{
					"type": "ephemeral",
				},
			},
		},
		Messages: []ClaudeMessage{{Role: "user", Content: "hello world"}},
	}

	profile := tracker.BuildClaudeProfile(req, 2048)
	if profile == nil {
		t.Fatalf("expected profile to be built")
	}
	tracker.Update("acct-1", profile)

	tracker.mu.Lock()
	for fp, entry := range tracker.entriesByAccount["acct-1"] {
		entry.ExpiresAt = time.Now().Add(50 * time.Millisecond)
		tracker.entriesByAccount["acct-1"][fp] = entry
	}
	var before time.Time
	for _, entry := range tracker.entriesByAccount["acct-1"] {
		before = entry.ExpiresAt
		break
	}
	tracker.mu.Unlock()

	result := tracker.Compute("acct-1", profile)
	if result.CacheReadInputTokens == 0 {
		t.Fatalf("expected warmed cache hit, got %+v", result)
	}

	tracker.mu.Lock()
	var after time.Time
	for _, entry := range tracker.entriesByAccount["acct-1"] {
		after = entry.ExpiresAt
		break
	}
	tracker.mu.Unlock()

	if after.UnixNano() != before.UnixNano() {
		t.Fatalf("expected cache hit to keep original expiry, before=%s after=%s", before, after)
	}
}
