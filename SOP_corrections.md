# Exact wording to change in FINAL_SOP.pdf (so the SOP and the code say the same thing)

## Why the SOP cannot say "all four branches learn a span"
ECERC's four Evidence-Cause Interaction branches use these attention masks (paper, Fig. 3):

| branch | mask | which earlier utterances can it see? | can a learned *span* change it? |
|---|---|---|---|
| self-emotion (contagion) | Smask | earlier utterances of the same speaker | **yes** |
| cross-event | Cmask | earlier utterances of other speakers | **yes** |
| cross-emotion | Cmask | earlier utterances of other speakers | **yes** |
| **self-event** | **Imask (identity)** | **only the target utterance itself (key distance = 0)** | **no - by the paper's own design** |

A span limits how far *back* a branch attends. The self-event branch never looks back (the paper defines this cause as the event
described in the target utterance itself), so its distance is always 0 and any span >= 0 keeps it unchanged. This is not a bug in our
implementation: it is verified by unit tests (`tests/run_tests.py::test_self_event_branch_is_provably_inactive`: the output is bit-identical
even if that span is set to 0, and its gradient is exactly 0). The code keeps a span module on that branch only for uniformity and reports it as
**inactive** (`status.json`/`final_results.json` -> `span.inactive_branches = ["self_event"]`, `final_report.md` section 6).

Making it "work" would require letting the self-event branch attend to the same speaker's *earlier* events. That changes ECERC's architecture
(and cannot start from the published baseline), which contradicts the SOP's own scope ("we do not alter the core architecture"), so it was deliberately
**not** implemented. If your supervisor wants it as an extra, it should be proposed as a separate, clearly labelled extension.

## Replacement text (copy into the SOP)

**I. Project details, Member 1 bullet** - replace
"...integrated across all four Evidence-Cause Interaction branches (self-event, cross-event, self-emotion/contagion, cross-emotion)."
with
"...integrated across the three history-attending Evidence-Cause Interaction branches (cross-event, self-emotion/contagion, cross-emotion). The self-event branch attends only to the target utterance (identity mask, key distance 0), so a span cannot affect it; it is kept for uniformity and reported as inactive."

**III. Methodology, Limitation 4 paragraph** - replace
"We adapt this idea to ECERC's four Evidence-Cause Interaction branches: each branch learns a soft span parameter determining how far back into the conversation it attends, replacing the model's current unbounded causal mask over the entire dialogue history."
with
"We adapt this idea to the three Evidence-Cause Interaction branches that attend to conversation history (cross-event, self-emotion/contagion, cross-emotion): each learns a soft span parameter (one per attention head) determining how far back it attends, replacing the unbounded causal mask over the dialogue history. The span starts at the longest dialogue length, i.e. exactly at the full-history baseline. The fourth branch, self-event, attends only to the target utterance (distance 0) and is therefore unaffected by a span; it is reported as inactive."

**III. Methodology, Member 1 sub-bullet of Limitation 4** - replace "...integrates it across all four Evidence-Cause Interaction branches." with "...integrates it across the three history-attending Evidence-Cause Interaction branches."

**Fig. 1** - change the box text "implements the adaptive-span masking/gating logic (all 4 EC-Interaction branches)" to "(3 history-attending EC-Interaction branches)".

**II. Problem statement and V. item 4** - replace "unbounded, full-history attention mechanism with no scalability control" / "Full-history, unbounded (O(L^2)) attention with no scalability control" with
"an attention mechanism that always uses the full dialogue history, with no learned or tunable limit on how far back each cause branch looks". (The span masks attention weights after they are computed, so it limits the *effective context* but does not reduce computation or memory; do not claim scalability.)

**VI. Expected outcomes, Limitation 4 bullet** - append: "Spans are reported per branch and per head for the three active branches; the self-event branch is reported as inactive by construction."

**V. item 1 (Limitation 1)** - replace "the focal-loss gamma hyperparameter the codebase already supports is never turned on for either dataset" with
"MELD's script uses no class-frequency weighting; IEMOCAP uses un-normalised inverse-frequency class weights and no focal term; neither uses effective-number class-balanced weights."
(On MELD the authors' loss already defaults to focal gamma = 1.)
