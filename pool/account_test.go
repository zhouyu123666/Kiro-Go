package pool

import (
	"errors"
	"kiro-go/config"
	"path/filepath"
	"testing"
	"time"
)

func TestOverLimitAccountsAreSkippedByDefault(t *testing.T) {
	p := &AccountPool{}
	normal := config.Account{ID: "normal"}
	overLimit := config.Account{ID: "over", UsageCurrent: 10, UsageLimit: 10}

	p.accounts = []config.Account{normal, overLimit}

	for i := 0; i < 5; i++ {
		acc := p.GetNext()
		if acc == nil {
			t.Fatalf("expected an account")
		}
		if acc.ID == "over" {
			t.Fatalf("expected over-limit account to be skipped when upstream OverageStatus is empty")
		}
	}
}

func TestOverLimitAccountsCanBeSelectedWhenUpstreamOverageEnabled(t *testing.T) {
	p := &AccountPool{}
	overLimit := config.Account{
		ID:            "over",
		UsageCurrent:  10,
		UsageLimit:    10,
		OverageStatus: "ENABLED",
	}

	p.accounts = []config.Account{overLimit}

	acc := p.GetNext()
	if acc == nil {
		t.Fatalf("expected upstream-enabled overage account to be selectable")
	}
	if acc.ID != "over" {
		t.Fatalf("expected overage account, got %q", acc.ID)
	}
}

func TestOverLimitAccountsRemainSkippedWhenUpstreamOverageDisabled(t *testing.T) {
	p := &AccountPool{}
	overLimit := config.Account{
		ID:            "over",
		UsageCurrent:  10,
		UsageLimit:    10,
		OverageStatus: "DISABLED",
	}

	p.accounts = []config.Account{overLimit}

	if acc := p.GetNext(); acc != nil {
		t.Fatalf("expected nil when upstream OverageStatus=DISABLED, got %q", acc.ID)
	}
}

func TestGetNextKeepsFiveMinuteTokenAvailable(t *testing.T) {
	p := &AccountPool{}
	account := config.Account{
		ID:          "acct-1",
		AccessToken: "access-token",
		ExpiresAt:   time.Now().Unix() + 300,
	}

	p.accounts = []config.Account{account}

	got := p.GetNext()
	if got == nil {
		t.Fatalf("expected five-minute token to be available")
	}
	if got.ID != account.ID {
		t.Fatalf("expected account %q, got %q", account.ID, got.ID)
	}
}

func TestNormalizeSubscriptionTier(t *testing.T) {
	tests := []struct {
		name  string
		typ   string
		title string
		want  string
	}{
		{name: "prefer explicit type", typ: "pro_plus", title: "free plan", want: "PRO_PLUS"},
		{name: "infer pro from title", typ: "", title: "Kiro Pro Monthly", want: "PRO"},
		{name: "infer free from title", typ: "", title: "free tier", want: "FREE"},
		{name: "unknown title", typ: "", title: "starter", want: ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeSubscriptionTier(tt.typ, tt.title); got != tt.want {
				t.Fatalf("normalizeSubscriptionTier(%q, %q) = %q, want %q", tt.typ, tt.title, got, tt.want)
			}
		})
	}
}

func TestSubscriptionLikelySupportsModel(t *testing.T) {
	tests := []struct {
		name  string
		typ   string
		title string
		model string
		want  bool
	}{
		{
			name:  "non opus model always allowed",
			typ:   "FREE",
			title: "Free",
			model: "claude-sonnet-4.5",
			want:  true,
		},
		{
			name:  "opus model requires pro-like tier",
			typ:   "FREE",
			title: "Free",
			model: "claude-opus-4.1",
			want:  false,
		},
		{
			name:  "opus model allowed on pro tier",
			typ:   "PRO",
			title: "Kiro Pro",
			model: "claude-opus-4.1",
			want:  true,
		},
		{
			name:  "unknown tier is optimistic",
			typ:   "",
			title: "",
			model: "claude-opus-4.1",
			want:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := subscriptionLikelySupportsModel(tt.typ, tt.title, tt.model); got != tt.want {
				t.Fatalf("subscriptionLikelySupportsModel(%q, %q, %q) = %v, want %v", tt.typ, tt.title, tt.model, got, tt.want)
			}
		})
	}
}

func TestAccountSupportsModelPrefersExplicitModelList(t *testing.T) {
	p := newTestPool()
	acc := &config.Account{
		ID:                "acct-1",
		SubscriptionType:  "PRO",
		SubscriptionTitle: "Kiro Pro",
	}

	// Even with PRO subscription, explicit model list should be authoritative.
	p.SetModelList("acct-1", []string{"claude-sonnet-4.5"})

	if p.accountSupportsModel(acc, "claude-opus-4.1") {
		t.Fatal("expected opus model to be rejected when not in explicit model list")
	}
	if !p.accountSupportsModel(acc, "claude-sonnet-4.5") {
		t.Fatal("expected listed model to be accepted")
	}
}

func TestAccountSupportsModelFallsBackToSubscriptionTier(t *testing.T) {
	p := newTestPool()
	free := &config.Account{
		ID:                "free-acct",
		SubscriptionType:  "FREE",
		SubscriptionTitle: "Free",
	}
	pro := &config.Account{
		ID:                "pro-acct",
		SubscriptionType:  "PRO",
		SubscriptionTitle: "Kiro Pro",
	}

	if p.accountSupportsModel(free, "claude-opus-4.1") {
		t.Fatal("expected FREE account to reject opus model when model list is absent")
	}
	if !p.accountSupportsModel(pro, "claude-opus-4.1") {
		t.Fatal("expected PRO account to allow opus model when model list is absent")
	}
}

// ---------------------------------------------------------------------------
// IsAuthFailure
// ---------------------------------------------------------------------------

func TestIsAuthFailureRecognizes401And403(t *testing.T) {
	positives := []string{
		"HTTP 401 from server",
		"received 403 Forbidden",
		"bad credentials",
		"invalid_grant",
		"invalid_token",
		"token expired",
		"token has expired",
		"unauthorized",
	}
	for _, msg := range positives {
		if !IsAuthFailure(errors.New(msg)) {
			t.Errorf("IsAuthFailure(%q) = false, want true", msg)
		}
	}
}

func TestIsAuthFailureIgnoresFalsePositives(t *testing.T) {
	// hasStatusToken only excludes digit boundaries; e.g. "4011" contains "401"
	// but the trailing '1' is a digit so it does NOT match.
	negatives := []string{
		"status code 4011 found", // digit immediately after 401 → not a standalone token
		"error 14013 exceeded",   // digit before and after 401
		"some random error",
		"status 200 OK",
	}
	for _, msg := range negatives {
		if IsAuthFailure(errors.New(msg)) {
			t.Errorf("IsAuthFailure(%q) = true, want false", msg)
		}
	}
}

func TestIsAuthFailureNilError(t *testing.T) {
	if IsAuthFailure(nil) {
		t.Fatal("IsAuthFailure(nil) = true, want false")
	}
}

// ---------------------------------------------------------------------------
// IsSuspensionError
// ---------------------------------------------------------------------------

func TestIsSuspensionErrorDetectsKnownMessages(t *testing.T) {
	positives := []string{
		"account temporarily_suspended",
		"account temporarily suspended",
		"no available kiro profile",
		"No Available Kiro Profile", // case-insensitive
	}
	for _, msg := range positives {
		if !IsSuspensionError(errors.New(msg)) {
			t.Errorf("IsSuspensionError(%q) = false, want true", msg)
		}
	}
}

func TestIsSuspensionErrorIgnoresUnrelatedErrors(t *testing.T) {
	negatives := []string{
		"some other error",
		"unauthorized",
		"429 too many requests",
	}
	for _, msg := range negatives {
		if IsSuspensionError(errors.New(msg)) {
			t.Errorf("IsSuspensionError(%q) = true, want false", msg)
		}
	}
}

func TestIsSuspensionErrorNilError(t *testing.T) {
	if IsSuspensionError(nil) {
		t.Fatal("IsSuspensionError(nil) = true, want false")
	}
}

// ---------------------------------------------------------------------------
// GetNextForModelExcluding
// ---------------------------------------------------------------------------

func newTestPool(accounts ...config.Account) *AccountPool {
	p := &AccountPool{
		cooldowns:   make(map[string]time.Time),
		errorCounts: make(map[string]int),
		modelLists:  make(map[string]map[string]bool),
	}
	p.accounts = accounts
	return p
}

func TestGetNextForModelExcludingSkipsExcludedAccounts(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a"},
		config.Account{ID: "b"},
	)
	excluded := map[string]bool{"a": true}
	for i := 0; i < 5; i++ {
		acc := p.GetNextForModelExcluding("model", excluded)
		if acc == nil {
			t.Fatal("expected account b, got nil")
		}
		if acc.ID == "a" {
			t.Fatalf("excluded account a was returned on iteration %d", i)
		}
	}
}

func TestGetNextForModelExcludingReturnsNilWhenAllExcluded(t *testing.T) {
	p := newTestPool(config.Account{ID: "only"})
	acc := p.GetNextForModelExcluding("model", map[string]bool{"only": true})
	if acc != nil {
		t.Fatalf("expected nil when only account is excluded, got %q", acc.ID)
	}
}

func TestGetNextForModelExcludingReturnsNilOnEmptyPool(t *testing.T) {
	p := newTestPool()
	acc := p.GetNextForModelExcluding("model", map[string]bool{})
	if acc != nil {
		t.Fatalf("expected nil for empty pool, got %q", acc.ID)
	}
}

// ---------------------------------------------------------------------------
// DisableAccount
// ---------------------------------------------------------------------------

func TestDisableAccountSetsCooldown(t *testing.T) {
	// Initialize a temporary config so SetAccountBanStatus can persist safely.
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}

	p := newTestPool()
	p.DisableAccount("test-id", "test reason")

	p.mu.RLock()
	cooldown, ok := p.cooldowns["test-id"]
	p.mu.RUnlock()

	if !ok {
		t.Fatal("expected cooldown to be set after DisableAccount")
	}
	// Safety-net cooldown must be at least 23 hours from now.
	minExpected := time.Now().Add(23 * time.Hour)
	if cooldown.Before(minExpected) {
		t.Fatalf("expected cooldown >= 23h in future, got %v", cooldown)
	}
}

func TestGetNextExcludingSkipsExcludedAccount(t *testing.T) {
	p := &AccountPool{
		accounts: []config.Account{
			{ID: "a", Enabled: true},
			{ID: "b", Enabled: true},
		},
		cooldowns:    make(map[string]time.Time),
		errorCounts:  make(map[string]int),
		modelLists:   make(map[string]map[string]bool),
		currentIndex: ^uint64(0),
	}

	acc := p.GetNextExcluding(map[string]bool{"a": true})
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected account b, got %#v", acc)
	}
}

func TestGetNextForModelExcludingSkipsExcludedAccount(t *testing.T) {
	p := &AccountPool{
		accounts: []config.Account{
			{ID: "a", Enabled: true},
			{ID: "b", Enabled: true},
		},
		cooldowns:    make(map[string]time.Time),
		errorCounts:  make(map[string]int),
		modelLists:   make(map[string]map[string]bool),
		currentIndex: ^uint64(0),
	}
	p.SetModelList("a", []string{"claude-sonnet-4.5"})
	p.SetModelList("b", []string{"claude-sonnet-4.5"})

	acc := p.GetNextForModelExcluding("claude-sonnet-4.5", map[string]bool{"a": true})
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected account b, got %#v", acc)
	}
}

func TestGetNextForModelWithAffinityPrefersLastSuccessfulAccount(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a", Enabled: true},
		config.Account{ID: "b", Enabled: true},
	)
	p.currentIndex = 0 // round-robin would choose b first
	p.RecordAffinitySuccess("session-1", "a")

	acc := p.GetNextForModelWithAffinity("claude-sonnet-4.5", "session-1", nil)
	if acc == nil || acc.ID != "a" {
		t.Fatalf("expected sticky account a, got %#v", acc)
	}
}

func TestGetNextForModelWithAffinityFallsBackWhenStickyAccountExcluded(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a", Enabled: true},
		config.Account{ID: "b", Enabled: true},
	)
	p.currentIndex = ^uint64(0)
	p.RecordAffinitySuccess("session-1", "a")

	acc := p.GetNextForModelWithAffinity("claude-sonnet-4.5", "session-1", map[string]bool{"a": true})
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected fallback account b, got %#v", acc)
	}
}

func TestGetNextForModelWithAffinityFallsBackWhenStickyAccountLacksModel(t *testing.T) {
	p := newTestPool(
		config.Account{ID: "a", Enabled: true},
		config.Account{ID: "b", Enabled: true},
	)
	p.SetModelList("a", []string{"claude-haiku-4.5"})
	p.SetModelList("b", []string{"claude-sonnet-4.5"})
	p.currentIndex = ^uint64(0)
	p.RecordAffinitySuccess("session-1", "a")

	acc := p.GetNextForModelWithAffinity("claude-sonnet-4.5", "session-1", nil)
	if acc == nil || acc.ID != "b" {
		t.Fatalf("expected fallback account b, got %#v", acc)
	}
}

// ---------------------------------------------------------------------------
// Reload over-usage filtering
// ---------------------------------------------------------------------------

func TestReloadKeepsOverQuotaAccountWhenAllowOverUsage(t *testing.T) {
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:           "over",
		Enabled:      true,
		UsageCurrent: 10,
		UsageLimit:   10,
	}); err != nil {
		t.Fatalf("AddAccount: %v", err)
	}
	if err := config.UpdateAllowOverUsage(true); err != nil {
		t.Fatalf("UpdateAllowOverUsage: %v", err)
	}

	p := newTestPool()
	p.Reload()

	if got := p.GetNext(); got == nil || got.ID != "over" {
		t.Fatalf("expected over-quota account to remain routable when allowOverUsage=true, got %#v", got)
	}
}

func TestReloadDropsOverQuotaAccountWhenAllowOverUsageDisabled(t *testing.T) {
	cfgFile := filepath.Join(t.TempDir(), "config.json")
	if err := config.Init(cfgFile); err != nil {
		t.Fatalf("config.Init: %v", err)
	}
	if err := config.AddAccount(config.Account{
		ID:           "over",
		Enabled:      true,
		UsageCurrent: 10,
		UsageLimit:   10,
	}); err != nil {
		t.Fatalf("AddAccount: %v", err)
	}

	p := newTestPool()
	p.Reload()

	if got := p.GetNext(); got != nil {
		t.Fatalf("expected over-quota account to be dropped, got %q", got.ID)
	}
}
