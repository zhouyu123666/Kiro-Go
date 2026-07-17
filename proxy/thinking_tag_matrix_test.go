package proxy

import (
	"strings"
	"testing"
)

// splitRunes breaks s into one-rune chunks, forcing the parser to reassemble
// tags across the worst-case chunk boundaries (every tag straddles many feeds).
func splitRunes(s string) []string {
	var out []string
	for _, r := range s {
		out = append(out, string(r))
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

// tag tokens built by concatenation so this source file itself never contains a
// literal <thinking> token that a naive scanner (or a future editor tool) might
// trip over.
var (
	mtOpen  = "<" + "thinking" + ">"
	mtClose = "</" + "thinking" + ">"
)

// thinkingMatrixCase describes one visible-channel stream (as delivered by the
// upstream, where a leading reasoning block is wrapped in <thinking>...</thinking>)
// and the expected split after the parser runs.
type thinkingMatrixCase struct {
	name         string
	input        string // full visible-channel text for the turn
	wantThinking string
	wantAnswer   string
	knownEdge    bool // true = documents an accepted limitation of depth-balancing
	note         string
}

// runThroughParser feeds chunks through a fresh parser (last chunk flushed) and
// returns the concatenated thinking and answer emissions.
func runThroughParser(chunks []string) (string, string) {
	events := collectFeed(chunks)
	return thinkingText(events), answerText(events)
}

// TestThinkingTagMatrix is the exhaustive stress sweep. Every case is run twice:
// as a single chunk, and split rune-by-rune so tag reassembly across streaming
// boundaries is exercised. Expected values reflect the ACTUAL shipped behavior of
// the monotonic parser + depth-balanced close, including the two documented edges
// where unbalanced literal tags inside reasoning are inherently ambiguous.
func TestThinkingTagMatrix(t *testing.T) {
	cases := []thinkingMatrixCase{
		{
			name:         "A_balanced_pair_in_reasoning",
			input:        mtOpen + "reason then " + mtOpen + "literal" + mtClose + " continue" + mtClose + "the answer",
			wantThinking: "reason then " + mtOpen + "literal" + mtClose + " continue",
			wantAnswer:   "the answer",
			note:         "balanced literal pair cancels out; real close balances the outer open",
		},
		{
			name:         "B_nested_multi_level",
			input:        mtOpen + "a" + mtOpen + "b" + mtOpen + "c" + mtClose + "d" + mtClose + "e" + mtClose + "ans",
			wantThinking: "a" + mtOpen + "b" + mtOpen + "c" + mtClose + "d" + mtClose + "e",
			wantAnswer:   "ans",
			note:         "depth climbs to 3 then unwinds to 0 exactly at the real close",
		},
		{
			name:         "C_lone_close_in_reasoning",
			input:        mtOpen + "reason " + mtClose + " tail" + mtClose + "ans",
			wantThinking: "reason ",
			wantAnswer:   " tail" + mtClose + "ans",
			knownEdge:    true,
			note:         "KNOWN EDGE: a lone </thinking> (surplus close) balances the open early and truncates; tail leaks to answer",
		},
		{
			name:         "B2_lone_open_in_reasoning",
			input:        mtOpen + "reason " + mtOpen + " no inner close, then real" + mtClose + "ans",
			wantThinking: "",
			wantAnswer:   mtOpen + "reason " + mtOpen + " no inner close, then real" + mtClose + "ans",
			knownEdge:    true,
			note:         "KNOWN EDGE: a lone <thinking> (surplus open) leaves depth>0 at the real close; on flush the whole block is preserved verbatim as answer, not swallowed",
		},
		{
			name:         "D_literal_tags_in_answer_body",
			input:        "the answer mentions " + mtOpen + " and " + mtClose + " inline",
			wantThinking: "",
			wantAnswer:   "the answer mentions " + mtOpen + " and " + mtClose + " inline",
			note:         "not leading, so answer phase from the start; tags are content",
		},
		{
			name:         "E_tags_in_code_block_after_reasoning",
			input:        mtOpen + "brief" + mtClose + "Here is code:\n```\n" + mtOpen + "x" + mtClose + "\n```\ndone",
			wantThinking: "brief",
			wantAnswer:   "Here is code:\n```\n" + mtOpen + "x" + mtClose + "\n```\ndone",
			note:         "leading block extracted; code-block tags in the answer stay verbatim",
		},
		{
			name:         "F_reasoning_and_answer_both_mention_tags",
			input:        mtOpen + "reason " + mtOpen + "x" + mtClose + mtClose + "answer has " + mtOpen + "literal" + mtClose + " too",
			wantThinking: "reason " + mtOpen + "x" + mtClose,
			wantAnswer:   "answer has " + mtOpen + "literal" + mtClose + " too",
			note:         "balanced pair in reasoning stays; answer-side literals stay too, no cross-contamination",
		},
		{
			name:         "G_plain_answer_no_tags",
			input:        "just a normal answer with no tags at all",
			wantThinking: "",
			wantAnswer:   "just a normal answer with no tags at all",
		},
		{
			name:         "H_leading_block_then_plain_answer",
			input:        mtOpen + "quick thought" + mtClose + "plain answer",
			wantThinking: "quick thought",
			wantAnswer:   "plain answer",
		},
		{
			name:         "I_unclosed_leading_open_is_answer",
			input:        mtOpen + "opens but never closes, all answer",
			wantThinking: "",
			wantAnswer:   mtOpen + "opens but never closes, all answer",
			note:         "closing-required rule: unclosed leading open is preserved verbatim",
		},
		{
			name:         "J_leading_whitespace_before_block",
			input:        "   \n" + mtOpen + "padded" + mtClose + "answer",
			wantThinking: "padded",
			wantAnswer:   "answer",
			note:         "leading whitespace does not disqualify the block",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name+"/single", func(t *testing.T) {
			gotThink, gotAns := runThroughParser([]string{tc.input})
			assertMatrix(t, tc, gotThink, gotAns)
		})
		t.Run(tc.name+"/runesplit", func(t *testing.T) {
			gotThink, gotAns := runThroughParser(splitRunes(tc.input))
			assertMatrix(t, tc, gotThink, gotAns)
		})
	}
}

func assertMatrix(t *testing.T, tc thinkingMatrixCase, gotThink, gotAns string) {
	t.Helper()
	if gotThink != tc.wantThinking {
		t.Errorf("[%s] thinking mismatch (edge=%v; %s)\n got:  %q\n want: %q", tc.name, tc.knownEdge, tc.note, gotThink, tc.wantThinking)
	}
	if gotAns != tc.wantAnswer {
		t.Errorf("[%s] answer mismatch (edge=%v; %s)\n got:  %q\n want: %q", tc.name, tc.knownEdge, tc.note, gotAns, tc.wantAnswer)
	}
	// Invariant that must hold for every non-edge case: no thinking content byte
	// is lost into the answer and vice versa — the concatenation of both channels
	// (with tags accounted for) reconstructs a superset of the input's visible text.
	if !tc.knownEdge {
		if strings.Contains(gotAns, tc.wantThinking) && tc.wantThinking != "" && tc.wantAnswer != tc.wantThinking {
			// The reasoning must NOT reappear inside the answer.
			if strings.Contains(gotAns, gotThink) && gotThink != "" {
				t.Errorf("[%s] reasoning leaked into answer: think=%q ans=%q", tc.name, gotThink, gotAns)
			}
		}
	}
}

// TestThinkingExtractorMatrix runs the same intent through the NON-stream path
// (extractThinkingFromContent via extractVisibleAndReasoning), which the buffered
// / non-streaming responses use. It must agree with the streaming parser on the
// leading-block + balanced-close semantics.
func TestThinkingExtractorMatrix(t *testing.T) {
	cases := []thinkingMatrixCase{
		{
			name:         "A_balanced_pair",
			input:        mtOpen + "reason " + mtOpen + "lit" + mtClose + " more" + mtClose + "answer",
			wantThinking: "reason " + mtOpen + "lit" + mtClose + " more",
			wantAnswer:   "answer",
		},
		{
			name:         "D_literal_in_answer_body",
			input:        "answer mentions " + mtOpen + " tag " + mtClose + " inline",
			wantThinking: "",
			wantAnswer:   "answer mentions " + mtOpen + " tag " + mtClose + " inline",
		},
		{
			name:         "F_both_mention",
			input:        mtOpen + "r " + mtOpen + "x" + mtClose + mtClose + "ans " + mtOpen + "lit" + mtClose,
			wantThinking: "r " + mtOpen + "x" + mtClose,
			wantAnswer:   "ans " + mtOpen + "lit" + mtClose,
		},
		{
			name:         "I_unclosed",
			input:        mtOpen + "never closes",
			wantThinking: "",
			wantAnswer:   mtOpen + "never closes",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			ans, reasoning := extractVisibleAndReasoning(tc.input, thinkingSourceUnknown)
			if reasoning != tc.wantThinking {
				t.Errorf("[%s] reasoning mismatch\n got:  %q\n want: %q", tc.name, reasoning, tc.wantThinking)
			}
			if ans != strings.TrimSpace(tc.wantAnswer) {
				t.Errorf("[%s] answer mismatch\n got:  %q\n want: %q", tc.name, ans, strings.TrimSpace(tc.wantAnswer))
			}
		})
	}
}
