package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"kiro-go/logger"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const defaultPromptCacheTTL = 5 * time.Minute

// Anthropic requires cached prefixes to reach a minimum token count before
// caching takes effect. Breakpoints below this threshold are excluded from
// matching and storage to avoid reporting unrealistic 100% cache hits on
// short requests.
const defaultMinCacheableTokens = 1024
const opusMinCacheableTokens = 4096
const claudeUsageEnvelopeMinTokens = 6

type promptCacheUsage struct {
	CacheCreationInputTokens   int
	CacheReadInputTokens       int
	CacheCreation5mInputTokens int
	CacheCreation1hInputTokens int
	CacheCoveredEstimate       int
	PromptTotalEstimate        int
}

type promptCacheBreakpoint struct {
	Fingerprint      [32]byte
	CumulativeTokens int
	TTL              time.Duration
}

type promptCacheProfile struct {
	Breakpoints        []promptCacheBreakpoint
	StorageBreakpoints []promptCacheBreakpoint
	TotalInputTokens   int
	Model              string
}

func minCacheableTokensForModel(model string) int {
	lower := strings.ToLower(model)
	if strings.Contains(lower, "opus") {
		return opusMinCacheableTokens
	}
	return defaultMinCacheableTokens
}

type promptCacheEntry struct {
	ExpiresAt time.Time
	TTL       time.Duration
}

type promptCacheTracker struct {
	mu               sync.Mutex
	entriesByAccount map[string]map[[32]byte]promptCacheEntry
	maxSupportedTTL  time.Duration
}

func newPromptCacheTracker(maxTTL time.Duration) *promptCacheTracker {
	if maxTTL <= 0 {
		maxTTL = defaultPromptCacheTTL
	}
	return &promptCacheTracker{
		entriesByAccount: make(map[string]map[[32]byte]promptCacheEntry),
		maxSupportedTTL:  maxTTL,
	}
}

func (t *promptCacheTracker) BuildClaudeProfile(req *ClaudeRequest, totalInputTokens int) *promptCacheProfile {
	blocks := flattenClaudeCacheBlocks(req)
	if len(blocks) == 0 {
		return nil
	}

	hasher := sha256.New()
	breakpoints := make([]promptCacheBreakpoint, 0)
	storageBreakpoints := make([]promptCacheBreakpoint, 0)
	cumulativeTokens := 0
	var activeTTL time.Duration

	lastBlockIndex := len(blocks) - 1
	for blockIndex, block := range blocks {
		canonical := canonicalizeCacheValue(block.Value)
		writeHashChunk(hasher, canonical)
		cumulativeTokens += block.Tokens

		// Track the active TTL from any explicit cache_control marker so that
		// later message-end boundaries can inherit it as implicit breakpoints.
		// This must happen even for the final block, otherwise a trailing
		// explicit marker would be lost for future turns.
		if block.TTL > 0 {
			activeTTL = block.TTL
		}

		breakpointTTL := time.Duration(0)
		if block.TTL > 0 {
			breakpointTTL = block.TTL
		} else if block.IsMessageEnd && activeTTL > 0 {
			breakpointTTL = activeTTL
		}

		if breakpointTTL <= 0 {
			continue
		}

		var fingerprint [32]byte
		copy(fingerprint[:], hasher.Sum(nil))
		breakpoint := promptCacheBreakpoint{
			Fingerprint:      fingerprint,
			CumulativeTokens: cumulativeTokens,
			TTL:              breakpointTTL,
		}
		storageBreakpoints = append(storageBreakpoints, breakpoint)

		// The final block is not added to current-turn cache-creation coverage
		// breakpoints, whether its cache_control is explicit or implicit.
		//
		// Claude Code marks the newest message (the last block) with
		// cache_control to prime the cache for the *next* turn, and it also
		// places cache_control on system/tools which makes every later
		// message-end an implicit breakpoint. Either way, if the last block
		// contributes to current-turn coverage its cumulative token count equals
		// the full prompt total, so CacheCoveredEstimate == PromptTotalEstimate,
		// the coverage ratio approaches 1.0, and the visible uncached
		// input_tokens collapses to the envelope floor (6).
		//
		// Still keep it in storage breakpoints so Compute can count an exact
		// prior match as a real cache read, and Update can warm it for the next
		// turn once a newer message follows it.
		if blockIndex == lastBlockIndex {
			continue
		}
		breakpoints = append(breakpoints, breakpoint)
	}

	if len(breakpoints) == 0 && len(storageBreakpoints) == 0 {
		logger.Debugf("[PromptCache] build_profile model=%s total_input=%d breakpoints=0 storage_breakpoints=0", req.Model, totalInputTokens)
		return nil
	}

	if totalInputTokens < cumulativeTokens {
		totalInputTokens = cumulativeTokens
	}

	profile := &promptCacheProfile{
		Breakpoints:        breakpoints,
		StorageBreakpoints: storageBreakpoints,
		TotalInputTokens:   totalInputTokens,
		Model:              req.Model,
	}
	lastTokens := 0
	if len(profile.Breakpoints) > 0 {
		lastTokens = profile.Breakpoints[len(profile.Breakpoints)-1].CumulativeTokens
	}
	lastStorageTokens := 0
	if len(profile.StorageBreakpoints) > 0 {
		lastStorageTokens = profile.StorageBreakpoints[len(profile.StorageBreakpoints)-1].CumulativeTokens
	}
	logger.Debugf("[PromptCache] build_profile model=%s total_input=%d breakpoints=%d last_breakpoint_tokens=%d storage_breakpoints=%d last_storage_tokens=%d",
		req.Model, totalInputTokens, len(profile.Breakpoints), lastTokens, len(profile.StorageBreakpoints), lastStorageTokens)
	return profile
}

func (t *promptCacheTracker) Compute(accountID string, profile *promptCacheProfile) promptCacheUsage {
	if t == nil || profile == nil || len(profile.StorageBreakpoints) == 0 || accountID == "" {
		logger.Debugf("[PromptCache] compute namespace=%q skipped profile_nil=%t breakpoints=%d storage_breakpoints=%d", accountID, profile == nil, func() int {
			if profile == nil {
				return 0
			}
			return len(profile.Breakpoints)
		}(), func() int {
			if profile == nil {
				return 0
			}
			return len(profile.StorageBreakpoints)
		}())
		return promptCacheUsage{}
	}

	minTokens := minCacheableTokensForModel(profile.Model)
	lastTokens := 0
	if len(profile.Breakpoints) > 0 {
		last := profile.Breakpoints[len(profile.Breakpoints)-1]
		lastTokens = minInt(last.CumulativeTokens, profile.TotalInputTokens)
	}
	now := time.Now()

	t.mu.Lock()
	defer t.mu.Unlock()
	t.pruneExpiredLocked(now)

	entries := t.entriesByAccount[accountID]
	if len(entries) == 0 {
		// First request for this account: report creation only if above threshold.
		effectiveCreation := lastTokens
		if effectiveCreation < minTokens {
			effectiveCreation = 0
		}
		cache5m, cache1h := computePromptCacheTTLBreakdown(profile, 0)
		usage := promptCacheUsage{
			CacheCreationInputTokens:   effectiveCreation,
			CacheReadInputTokens:       0,
			CacheCreation5mInputTokens: cache5m,
			CacheCreation1hInputTokens: cache1h,
			CacheCoveredEstimate:       effectiveCreation,
			PromptTotalEstimate:        profile.TotalInputTokens,
		}
		logger.Debugf("[PromptCache] compute namespace=%q entries=0 total_input=%d min_tokens=%d last_tokens=%d result={creation:%d read:%d covered_est:%d prompt_total_est:%d}",
			accountID, profile.TotalInputTokens, minTokens, lastTokens, usage.CacheCreationInputTokens, usage.CacheReadInputTokens, usage.CacheCoveredEstimate, usage.PromptTotalEstimate)
		return usage
	}

	matchedTokens := 0
	for i := len(profile.StorageBreakpoints) - 1; i >= 0; i-- {
		breakpoint := profile.StorageBreakpoints[i]
		// Skip breakpoints below the minimum cacheable token threshold.
		if breakpoint.CumulativeTokens < minTokens {
			continue
		}
		entry, ok := entries[breakpoint.Fingerprint]
		if !ok || entry.ExpiresAt.Before(now) {
			continue
		}
		matchedTokens = minInt(breakpoint.CumulativeTokens, profile.TotalInputTokens)
		break
	}

	creation := maxInt(lastTokens-matchedTokens, 0)
	coveredEstimate := maxInt(lastTokens, matchedTokens)
	cache5m, cache1h := computePromptCacheTTLBreakdown(profile, matchedTokens)
	usage := promptCacheUsage{
		CacheCreationInputTokens:   creation,
		CacheReadInputTokens:       matchedTokens,
		CacheCreation5mInputTokens: cache5m,
		CacheCreation1hInputTokens: cache1h,
		CacheCoveredEstimate:       coveredEstimate,
		PromptTotalEstimate:        profile.TotalInputTokens,
	}
	logger.Debugf("[PromptCache] compute namespace=%q entries=%d total_input=%d min_tokens=%d last_tokens=%d matched_tokens=%d breakpoints=%d storage_breakpoints=%d result={creation:%d read:%d covered_est:%d prompt_total_est:%d}",
		accountID, len(entries), profile.TotalInputTokens, minTokens, lastTokens, matchedTokens, len(profile.Breakpoints), len(profile.StorageBreakpoints), usage.CacheCreationInputTokens, usage.CacheReadInputTokens, usage.CacheCoveredEstimate, usage.PromptTotalEstimate)
	return usage
}

func (t *promptCacheTracker) Update(accountID string, profile *promptCacheProfile) {
	if t == nil || profile == nil || len(profile.StorageBreakpoints) == 0 || accountID == "" {
		return
	}

	minTokens := minCacheableTokensForModel(profile.Model)
	now := time.Now()
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pruneExpiredLocked(now)

	entries := t.entriesByAccount[accountID]
	if entries == nil {
		entries = make(map[[32]byte]promptCacheEntry)
		t.entriesByAccount[accountID] = entries
	}

	for _, breakpoint := range profile.StorageBreakpoints {
		// Skip breakpoints below the minimum cacheable token threshold.
		if breakpoint.CumulativeTokens < minTokens {
			continue
		}
		entries[breakpoint.Fingerprint] = promptCacheEntry{
			ExpiresAt: now.Add(breakpoint.TTL),
			TTL:       breakpoint.TTL,
		}
	}
}

func (t *promptCacheTracker) pruneExpiredLocked(now time.Time) {
	for accountID, entries := range t.entriesByAccount {
		for fingerprint, entry := range entries {
			if !entry.ExpiresAt.After(now) {
				delete(entries, fingerprint)
			}
		}
		if len(entries) == 0 {
			delete(t.entriesByAccount, accountID)
		}
	}
}

type cacheablePromptBlock struct {
	Value        interface{}
	Tokens       int
	TTL          time.Duration
	IsMessageEnd bool
}

func flattenClaudeCacheBlocks(req *ClaudeRequest) []cacheablePromptBlock {
	blocks := make([]cacheablePromptBlock, 0)
	blocks = append(blocks, buildCachePreludeBlock(req))

	for toolIndex, tool := range req.Tools {
		toolValue := map[string]interface{}{
			"kind":         "tool",
			"tool_index":   toolIndex,
			"name":         tool.Name,
			"description":  tool.Description,
			"input_schema": tool.InputSchema,
		}
		fingerprintValue := stripCachePositionKeys(toolValue)
		blocks = append(blocks, cacheablePromptBlock{
			Value:  fingerprintValue,
			Tokens: estimateApproxTokens(canonicalizeCacheValue(fingerprintValue)),
			TTL:    normalizePromptCacheTTL(extractPromptCacheTTL(tool)),
		})
	}

	appendSystemCacheBlocks(&blocks, req.System)

	for messageIndex, msg := range req.Messages {
		appendMessageCacheBlocks(&blocks, messageIndex, msg)
	}

	return blocks
}

func buildCachePreludeBlock(req *ClaudeRequest) cacheablePromptBlock {
	prelude := map[string]interface{}{
		"kind":        "request_prelude",
		"model":       req.Model,
		"tool_choice": req.ToolChoice,
	}
	return cacheablePromptBlock{
		Value:  prelude,
		Tokens: estimateApproxTokens(canonicalizeCacheValue(prelude)),
	}
}

func appendSystemCacheBlocks(blocks *[]cacheablePromptBlock, system interface{}) {
	switch v := system.(type) {
	case string:
		appendPromptBlock(blocks, map[string]interface{}{
			"kind":         "system",
			"system_index": 0,
			"block": map[string]interface{}{
				"type": "text",
				"text": v,
			},
		}, false)
	case []interface{}:
		startIndex := firstExplicitPromptCacheBlockIndex(v)
		for i, block := range v {
			if i < startIndex {
				continue
			}
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":         "system",
				"system_index": i,
				"block":        block,
			}, false)
		}
	case []string:
		for i, block := range v {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":         "system",
				"system_index": i,
				"block": map[string]interface{}{
					"type": "text",
					"text": block,
				},
			}, false)
		}
	}
}

func firstExplicitPromptCacheBlockIndex(blocks []interface{}) int {
	for i, block := range blocks {
		if normalizePromptCacheTTL(extractPromptCacheTTL(block)) > 0 {
			return i
		}
	}
	return 0
}

func appendMessageCacheBlocks(blocks *[]cacheablePromptBlock, messageIndex int, msg ClaudeMessage) {
	role := msg.Role
	switch content := msg.Content.(type) {
	case string:
		appendPromptBlock(blocks, map[string]interface{}{
			"kind":          "message",
			"message_index": messageIndex,
			"role":          role,
			"block_index":   0,
			"block": map[string]interface{}{
				"type": "text",
				"text": content,
			},
		}, true)
	case []interface{}:
		lastIdx := len(content) - 1
		for blockIndex, block := range content {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":          "message",
				"message_index": messageIndex,
				"role":          role,
				"block_index":   blockIndex,
				"block":         block,
			}, blockIndex == lastIdx)
		}
	default:
		if content != nil {
			appendPromptBlock(blocks, map[string]interface{}{
				"kind":          "message",
				"message_index": messageIndex,
				"role":          role,
				"block_index":   0,
				"block":         content,
			}, true)
		}
	}
}

func appendPromptBlock(blocks *[]cacheablePromptBlock, wrapper map[string]interface{}, isMessageEnd bool) {
	blockValue := wrapper["block"]
	ttl := normalizePromptCacheTTL(extractPromptCacheTTL(blockValue))

	// Drop volatile billing metadata from the cache fingerprint. Claude Code's
	// x-anthropic-billing-header can drift, appear, or disappear across
	// otherwise identical requests, and it does not change model semantics.
	if isAnthropicBillingHeaderBlock(blockValue) {
		return
	}

	fingerprintValue := stripCachePositionKeys(wrapper)
	canonical := canonicalizeCacheValue(fingerprintValue)
	*blocks = append(*blocks, cacheablePromptBlock{
		Value:        fingerprintValue,
		Tokens:       estimateApproxTokens(canonical),
		TTL:          ttl,
		IsMessageEnd: isMessageEnd,
	})
}

func stripCachePositionKeys(value map[string]interface{}) map[string]interface{} {
	cloned := make(map[string]interface{}, len(value))
	for key, item := range value {
		if isCachePositionKey(key) {
			continue
		}
		cloned[key] = item
	}
	return cloned
}

func isAnthropicBillingHeaderBlock(value interface{}) bool {
	blockMap, ok := value.(map[string]interface{})
	if !ok {
		return false
	}

	// Only normalize text blocks (or blocks without an explicit type but containing text).
	if t, ok := blockMap["type"].(string); ok && t != "" && t != "text" {
		return false
	}

	text, ok := blockMap["text"].(string)
	if !ok {
		return false
	}

	trimmed := strings.TrimLeft(text, " \t\r\n")
	return strings.HasPrefix(strings.ToLower(trimmed), "x-anthropic-billing-header:")
}

func extractPromptCacheTTL(value interface{}) time.Duration {
	block, ok := value.(map[string]interface{})
	if !ok {
		if raw, err := json.Marshal(value); err == nil {
			var decoded map[string]interface{}
			if json.Unmarshal(raw, &decoded) == nil {
				block = decoded
				ok = true
			}
		}
	}
	if !ok {
		return 0
	}

	rawCache, ok := block["cache_control"]
	if !ok {
		return 0
	}
	cacheControl, ok := rawCache.(map[string]interface{})
	if !ok {
		return 0
	}
	cacheType, _ := cacheControl["type"].(string)
	if !strings.EqualFold(cacheType, "ephemeral") {
		return 0
	}

	if ttl, ok := parsePromptCacheTTLValue(cacheControl["ttl"]); ok {
		return ttl
	}
	return defaultPromptCacheTTL
}

func parsePromptCacheTTLValue(value interface{}) (time.Duration, bool) {
	switch v := value.(type) {
	case string:
		trimmed := strings.TrimSpace(strings.ToLower(v))
		if trimmed == "" {
			return 0, false
		}
		if d, err := time.ParseDuration(trimmed); err == nil {
			return d, true
		}
		if seconds, err := strconv.Atoi(trimmed); err == nil {
			return time.Duration(seconds) * time.Second, true
		}
	case float64:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	case int:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	case int64:
		if v > 0 {
			return time.Duration(v) * time.Second, true
		}
	}
	return 0, false
}

func normalizePromptCacheTTL(ttl time.Duration) time.Duration {
	if ttl <= 0 {
		return 0
	}
	if ttl > time.Hour {
		return time.Hour
	}
	if ttl > defaultPromptCacheTTL {
		return time.Hour
	}
	return defaultPromptCacheTTL
}

func computePromptCacheTTLBreakdown(profile *promptCacheProfile, matchedTokens int) (int, int) {
	if profile == nil || len(profile.Breakpoints) == 0 {
		return 0, 0
	}

	cache5m := 0
	cache1h := 0
	previous := matchedTokens
	for _, breakpoint := range profile.Breakpoints {
		current := minInt(breakpoint.CumulativeTokens, profile.TotalInputTokens)
		if current <= previous {
			continue
		}
		delta := current - previous
		if breakpoint.TTL >= time.Hour {
			cache1h += delta
		} else {
			cache5m += delta
		}
		previous = current
	}
	return cache5m, cache1h
}

func buildClaudeUsageMap(inputTokens, outputTokens int, usage promptCacheUsage, includeCache bool) map[string]interface{} {
	visibleInputTokens, usage := claudeUsageBreakdown(inputTokens, usage, includeCache)
	result := map[string]interface{}{
		"input_tokens":  visibleInputTokens,
		"output_tokens": outputTokens,
	}
	if !includeCache {
		return result
	}
	result["cache_creation_input_tokens"] = usage.CacheCreationInputTokens
	result["cache_read_input_tokens"] = usage.CacheReadInputTokens
	result["cache_creation"] = map[string]int{
		"ephemeral_5m_input_tokens": usage.CacheCreation5mInputTokens,
		"ephemeral_1h_input_tokens": usage.CacheCreation1hInputTokens,
	}
	return result
}

func buildClaudeBillableUsageMap(inputTokens, outputTokens int, usage promptCacheUsage, includeCache bool) map[string]interface{} {
	if inputTokens < 0 {
		inputTokens = 0
	}
	result := map[string]interface{}{
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
	}
	if !includeCache {
		return result
	}
	result["cache_creation_input_tokens"] = maxInt(usage.CacheCreationInputTokens, 0)
	result["cache_read_input_tokens"] = maxInt(usage.CacheReadInputTokens, 0)
	result["cache_creation"] = map[string]int{
		"ephemeral_5m_input_tokens": maxInt(usage.CacheCreation5mInputTokens, 0),
		"ephemeral_1h_input_tokens": maxInt(usage.CacheCreation1hInputTokens, 0),
	}
	return result
}

func claudeUsageBreakdown(totalInputTokens int, usage promptCacheUsage, includeCache bool) (int, promptCacheUsage) {
	if totalInputTokens < 0 {
		totalInputTokens = 0
	}
	if !includeCache {
		return totalInputTokens, promptCacheUsage{}
	}

	usage = rebalancePromptCacheUsage(totalInputTokens, usage)
	cachedInputTokens := usage.CacheReadInputTokens + usage.CacheCreationInputTokens
	if cachedInputTokens <= 0 {
		return totalInputTokens, usage
	}
	if cachedInputTokens > totalInputTokens {
		cachedInputTokens = totalInputTokens
	}
	visibleInputTokens := totalInputTokens - cachedInputTokens
	visibleInputTokens, usage = applyClaudeUsageEnvelopeFloor(totalInputTokens, visibleInputTokens, usage)
	return visibleInputTokens, usage
}

func rebalancePromptCacheUsage(totalInputTokens int, usage promptCacheUsage) promptCacheUsage {
	if totalInputTokens <= 0 {
		return promptCacheUsage{}
	}

	readTokens := maxInt(usage.CacheReadInputTokens, 0)
	creationTokens := maxInt(usage.CacheCreationInputTokens, 0)
	cachedInputTokens := readTokens + creationTokens
	if cachedInputTokens <= 0 {
		usage.CacheReadInputTokens = 0
		usage.CacheCreationInputTokens = 0
		usage.CacheCreation5mInputTokens = 0
		usage.CacheCreation1hInputTokens = 0
		usage.CacheCoveredEstimate = 0
		return usage
	}

	scaledCacheTotal := cachedInputTokens
	rawCovered := maxInt(usage.CacheCoveredEstimate, 0)
	rawPromptTotal := maxInt(usage.PromptTotalEstimate, 0)
	if rawCovered > 0 && rawPromptTotal > 0 {
		scaledCacheTotal = proportionalInt(totalInputTokens, rawCovered, rawPromptTotal)
	}
	if scaledCacheTotal > totalInputTokens {
		scaledCacheTotal = totalInputTokens
	}

	readTokens = proportionalInt(scaledCacheTotal, readTokens, cachedInputTokens)
	creationTokens = scaledCacheTotal - readTokens
	cache5m, cache1h := rebalanceCacheCreationTTL(creationTokens, usage)
	return promptCacheUsage{
		CacheCreationInputTokens:   creationTokens,
		CacheReadInputTokens:       readTokens,
		CacheCreation5mInputTokens: cache5m,
		CacheCreation1hInputTokens: cache1h,
		CacheCoveredEstimate:       scaledCacheTotal,
		PromptTotalEstimate:        totalInputTokens,
	}
}

func rebalanceCacheCreationTTL(creationTokens int, usage promptCacheUsage) (int, int) {
	if creationTokens <= 0 {
		return 0, 0
	}

	raw5m := maxInt(usage.CacheCreation5mInputTokens, 0)
	raw1h := maxInt(usage.CacheCreation1hInputTokens, 0)
	rawTTL := raw5m + raw1h
	if rawTTL <= 0 {
		return creationTokens, 0
	}

	cache5m := proportionalInt(creationTokens, raw5m, rawTTL)
	return cache5m, creationTokens - cache5m
}

func proportionalInt(total, numerator, denominator int) int {
	if total <= 0 || numerator <= 0 || denominator <= 0 {
		return 0
	}
	value := (int64(total)*int64(numerator) + int64(denominator)/2) / int64(denominator)
	if value > int64(total) {
		return total
	}
	return int(value)
}

func applyClaudeUsageEnvelopeFloor(totalInputTokens, visibleInputTokens int, usage promptCacheUsage) (int, promptCacheUsage) {
	if totalInputTokens <= 0 {
		return 0, usage
	}
	if visibleInputTokens >= claudeUsageEnvelopeMinTokens {
		return visibleInputTokens, usage
	}

	desiredVisibleTokens := minInt(claudeUsageEnvelopeMinTokens, totalInputTokens)
	cacheBudget := totalInputTokens - desiredVisibleTokens
	if cacheBudget < 0 {
		cacheBudget = 0
	}

	cachedInputTokens := maxInt(usage.CacheReadInputTokens, 0) + maxInt(usage.CacheCreationInputTokens, 0)
	if cachedInputTokens <= 0 {
		return desiredVisibleTokens, usage
	}

	readTokens := proportionalInt(cacheBudget, maxInt(usage.CacheReadInputTokens, 0), cachedInputTokens)
	creationTokens := cacheBudget - readTokens
	cache5m, cache1h := rebalanceCacheCreationTTL(creationTokens, usage)
	return desiredVisibleTokens, promptCacheUsage{
		CacheCreationInputTokens:   creationTokens,
		CacheReadInputTokens:       readTokens,
		CacheCreation5mInputTokens: cache5m,
		CacheCreation1hInputTokens: cache1h,
		CacheCoveredEstimate:       cacheBudget,
		PromptTotalEstimate:        totalInputTokens,
	}
}

func canonicalizeCacheValue(value interface{}) string {
	var buf bytes.Buffer
	writeCanonicalJSON(&buf, value)
	return buf.String()
}

func writeCanonicalJSON(buf *bytes.Buffer, value interface{}) {
	switch v := value.(type) {
	case nil:
		buf.WriteString("null")
	case string:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	case bool:
		if v {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
	case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, json.Number:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	case []interface{}:
		buf.WriteByte('[')
		for i, item := range v {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeCanonicalJSON(buf, item)
		}
		buf.WriteByte(']')
	case map[string]interface{}:
		buf.WriteByte('{')
		keys := make([]string, 0, len(v))
		for key := range v {
			if key == "cache_control" {
				continue
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for i, key := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			encoded, _ := json.Marshal(key)
			buf.Write(encoded)
			buf.WriteByte(':')
			writeCanonicalJSON(buf, v[key])
		}
		buf.WriteByte('}')
	default:
		encoded, _ := json.Marshal(v)
		buf.Write(encoded)
	}
}

func isCachePositionKey(key string) bool {
	switch key {
	case "tool_index", "system_index", "message_index", "block_index":
		return true
	default:
		return false
	}
}

func writeHashChunk(hasher hashWriter, chunk string) {
	length := strconv.Itoa(len(chunk))
	hasher.Write([]byte(length))
	hasher.Write([]byte{0})
	hasher.Write([]byte(chunk))
	hasher.Write([]byte{0})
}

type hashWriter interface {
	Write([]byte) (int, error)
	Sum([]byte) []byte
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
