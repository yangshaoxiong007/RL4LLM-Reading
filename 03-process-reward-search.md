<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:22D3EE,100:5B8CFF&height=140&section=header&text=03%20%C2%B7%20Process%20Reward%20for%20Search&fontSize=34&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

## 1 · Why it matters

A **process reward** is a *step-level, checkable* signal $r_t$ along a reasoning/search trajectory, as opposed to the *outcome reward* $r_H$ at the end. It is the concrete mechanism that resolves *both* earlier threads at once:

- vs. doc 01: dense $r_t$ shrinks the temporal horizon of the credit problem from $H$ to ~1, because each step is rewarded locally;
- vs. doc 02: per-step / per-turn rewards *are* a credit signal — they decompose the team outcome into attributable pieces without a centralized $Q$.

**Search is the domain where process rewards are most natural** because each retrieval / reasoning step has a *checkable intermediate state*: does the fetched evidence support the partial answer? does the query retrieve *novel* useful information? is the reasoning step locally sound? Search turns "process reward" from a hand-labeled luxury (PRM800K) into something partially *machine-verifiable*, which is why Search-R1-class work can train with far less human annotation than math-PRM work.

## 2 · Formalization

Model search as an MDP:

$$\mathcal M=(\mathcal S,\mathcal A,P,r,\gamma),\qquad s_t=\text{accumulated context},\quad a_t\in\{\text{query},\text{tool-call},\text{reason-step},\text{terminate}\}$$

Outcome RL uses $r=\sum_t r_t$ with $r_t=0$ for $t<H$, $r_H=\text{verifier}(\text{answer})$. **Process-rewarded RL** replaces this with $r_t$ defined (or *learned* from rollouts) per step. Three forms of $r_t$:

1. **Human-annotated PRM** — a classifier $\rho_\phi(s_t,a_t)\to[0,1]$ trained on step labels (correct/incorrect/useful) — PRM800K.
2. **Auto-labeled PRM (Monte-Carlo)** — estimate $\rho_\phi$ by the empirical probability of *eventual success* conditioned on reaching step $(s_t,a_t)$, via sampled continuations — Math-Shepherd; the trick that makes PRMs cheap.
3. **Implicit / verifiable process reward** — no separate $\rho_\phi$; instead *shape* the outcome reward onto steps through an auxiliary objective (e.g. token-level advantage from a value/VLM, or step-level verifier as in Search-R1's per-search-step rule).

The estimator that turns $r_t$ into a gradient is the same as doc 01's: a leave-one-out / group advantage **applied at step granularity** — which is exactly why process reward and GRPO compose, and exactly where the search-RL trio lives.

## 3 · Method taxonomy

| Family | What provides $r_t$ | Faithfulness | Cost | Representative |
|---|---|---|---|---|
| **Human PRM** | labeled step correctness | High but bounded by labeler quality | Very high (PRM800K ≈ 800k human labels) | Let's Verify (2305.20050) |
| **MC-auto PRM** | empirical eventual-success per step | Self-consistent; biased by rollout policy | Moderate (rollouts at labeling time) | Math-Shepherd (2312.08935) |
| **Implicit PRM / rechecking** | derive step reward from outcome + value | No annotation; inherits value-bias noise | Low | Implicit PRM / rechecked reasoning (arXiv ID unconfirmed) |
| **Search-step verifier PRM** | rule-based per retrieval step (evidence sufficiency, novelty) | Faithful for factual search | Low (heuristic + verifier) | Search-R1 (arXiv:2503.09516); R1-Searcher (arXiv:2503.05592); ReSearch (arXiv:2503.19470) |
| **Outcome-RL + PRM-outcome hybrid** | train PRM from outcome rollouts, then RL on PRM | Combines 2+4; PRM can drift from outcome | Moderate | autumo-PRM / Q*-style reward shaping (arXiv ID unconfirmed) |
| **Step-level outcome decomposition (the author's setting)** | decompose team outcome onto role-steps | Bridges doc 02 ⇆ 03 | Verifier-bound | Multi-Agent Search + RL (ACL 2026) |

## 4 · Paper dissection (expert)

### PRM800K / Let's Verify Step by Step (arXiv 2305.20050) ✓
**Contribution.** Scale human *step-level* labels (correct/incorrect per CoT step) to ~800k annotations; train a process reward model $\rho_\phi$ and show that **best-of-N with $\rho_\phi$ beats outcome-reward best-of-N** on math. The empirical founding result: *process supervision helps*.
**The expert caveat.** The gain is largest where (i) the task decomposes into checkable steps and (ii) humans can label those steps reliably. Research/search only partially satisfies (ii); the legacy of this paper is not "annotate everything" but **"find the cheap verifier"** — which the search-RL trio does by replacing human labelers with retrieval-time heuristics.
**Limit.** Labeler disagreement on ambiguous steps; label cost makes it non-reproducible at scale.

### Math-Shepherd (arXiv 2312.08935) ✓
**Contribution.** **Auto-label** step rewards by Monte-Carlo: for each partial solution, sample completions; label the step "hard" (+) if completions frequently reach the correct *final* answer, "weak" otherwise. This removes the human labeler — PRMs become data-generated — and is the practical pivot that made process-rewarded RL deployable.
**The subtle point.** Math-Shepherd's auto-label is **consistent with the labeling-time policy** — if the rollout policy improves, the labels change. This is a *feature* (self-improving supervision) and a *bug* (PRM lag, distribution shift between labeler and learner). The same tension is the reason R1-style outcome RL can overtake PRM-RL in some regimes: outcome RL needs no policy-consistent labeler.
**Read across docs.** Math-Shepherd's MC labeling *is* a form of credit assignment (doc 02): it estimates, per step, the counterfactual "success probability if we reach here" — i.e. the step's marginal credit for the outcome. PRMs and COMA are estimating cousins of the same object.

### Search-RL trio: Search-R1 / R1-Searcher / ReSearch (IDs ⟨id→?⟩ — verify)
**Common contribution.** Train LLM agents that **interleave reasoning and retrieval** under RL, using a **per-search-step process signal** (e.g. evidence-sufficiency / answer-grounding) alongside the final-answer outcome reward. This is where process reward meets long horizon (doc 01) *and* the search-as-MDP framing (§2).
- **Search-R1-style** treats each retrieval call as an action and shapes reward with whether retrieved documents improve the partial answer — i.e. process reward = marginal information gain of the query.
- **R1-Searcher-style** learns *when to search* and *what to query* as policy, process-rewarded by retrieval quality — closest to "verifiable process reward" in the open-ended setting.
- **ReSearch-style** separates reasoning from search state, using process reward to align the reasoning transition with the retrieved evidence — a clean signal because the transition is locally checkable.
**Expert synthesis (this is the crux of all three notes).** Search-RL is the existence proof that **process rewards can be made cheaper than outcome verifiers when the action space is search**, because retrieval steps have intermediate checkable states. This is precisely why *the author's multi-agent-search + RL line* lives here: role-specialized search agents share an outcome, and the only affordable credit signal is the search-step process reward — making doc 02's between-group credit problem *solvable from data* (the open problem stated in doc 02 §6).

## 5 · Critique

- **"PRM > ORM, always."** Only under cheap step-verifiability. For research/search the PRM is either heuristic (drift) or learned (Math-Shepherd lag); the gain over outcome RL is regime-dependent, not universal. Recent outcome-RL reproduction successes (R1-class) are quietly evidence *against* universal PRM superiority.
- **Auto-PRM policy-consistency is under-theorized.** The MC label is policy-defined; few papers analyze the PRM-learner feedback loop. It behaves like a self-distillation, with the usual collapse/stagnation risks — unverified in practice.
- **Search reward shape is brittle hand-design.** "Evidence sufficiency", "novelty", "grounding" are defined per paper, differ across the trio, and rarely ablated together. A shared, verifiable search-step reward benchmark is missing.

## 6 · Open problems (→ the author's line)

1. **A unified, verifiable search-process reward.** Converge the trio's ad-hoc step rewards into one checkable surrogate (the search analog of a math verifier).
2. **Auto-PRM under multi-agent rollouts.** Math-Shepherd assumes a *single* continuation policy; in heterogeneous-agent search the labeled "eventual success" mixes role contributions. Credit-aware auto-PRM = the open joint of docs 02 ⇆ 03.
3. **PRM-as-critic vs. PRM-as-reward.** Use a PRM as the centralized critic for credit (COMA-style, doc 02) rather than merely as a reward bonus — barely explored for LLM search agents.
4. **Verifier for research.** The terminal constraint: without a research verifier, even perfect process-reward machinery caps at where the verifier is trustworthy.

## References (subject to § verification)

- Let's Verify Step by Step (PRM/PRM800K) — **arXiv:2305.20050** ✓
- Math-Shepherd — **arXiv:2312.08935** ✓
- Search-R1 — **arXiv:2503.09516** ✓
- R1-Searcher — **arXiv:2503.05592** ✓
- ReSearch — **arXiv:2503.19470** ✓
- PRMBench — **arXiv:2501.03124** ✓
- ProcessBench, Implicit PRM/rechecking — arXiv ID unconfirmed; cite by title
- GRPO/Dr.GRPO/DAPO estimator context — doc 01

<p align="center"><sub>03 / 03 · Process Reward for Search · cross-refs 01 (horizon), 02 (credit)</sub></p>
