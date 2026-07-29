<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:22D3EE,100:5B8CFF&height=140&section=header&text=03%20%C2%B7%20Process%20Reward%20for%20Search&fontSize=30&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

<p align="center"><sub>🇬🇧 English &nbsp;·&nbsp; <a href="./03-process-reward-search.zh.md">🇨🇳 中文</a> &nbsp;·&nbsp; cross-refs <a href="./01-long-horizon-rl.md">01 Long-Horizon</a> · <a href="./02-credit-assignment-rl.md">02 Credit</a></sub></p>

## 1 · Why it matters

A **process reward** is a *step-level, checkable* signal $r_t$ along a trajectory, vs. the terminal *outcome reward* $r_H$. It resolves both earlier threads at once:
- vs. doc 01: dense $r_t$ shrinks the temporal horizon of credit from $H$ to ~1;
- vs. doc 02: per-step/turn rewards *are* a credit signal — decomposing the team outcome into attributable pieces without a centralized $Q$.

**Search is the domain where process rewards are most natural** because each retrieval/reasoning step has a *checkable intermediate state*: does fetched evidence support the partial answer? does the query retrieve *novel* useful info? Search turns "process reward" from a hand-labeled luxury (PRM800K) into something *partly machine-verifiable* — why Search-R1-class work trains with far less human annotation.

## 2 · Formalization

Search as MDP: $s_t=$ accumulated context, $a_t\in\{$query, tool-call, reason-step, terminate$\}$. Outcome RL: $r_t=0(t<H), r_H=\text{verifier(answer)}$. Process-rewarded RL replaces it with $r_t$ (defined or *learned*). Three forms:
1. **Human PRM** — classifier $\rho_\phi$ on step labels (PRM800K).
2. **MC-auto PRM** — $\rho_\phi$ = empirical eventual-success conditioned on reaching the step, via sampled continuations (Math-Shepherd); the trick that makes PRMs cheap.
3. **Implicit / verifiable process reward** — no separate $\rho_\phi$; shape outcome onto steps via an auxiliary objective (token-level advantage, or Search-R1's per-search-step rules).

The estimator turning $r_t$ into a gradient is the same leave-one-out/group advantage as doc 01, applied at step granularity — why process reward and GRPO compose.

## 3 · Method taxonomy (with precise 2025–2026 search-RL classification)

The table fuses classic PRMs and recent search RL, annotated [reward type / scope / algorithm] — cross-checked against public awesome-lists and arXiv title verification.

| Work | $r_t$ source | Reward type | Scope | Algorithm | arXiv |
|---|---|---|---|---|---|
| PRM800K / Let's Verify | human step-correctness | PRM | step | (BoN eval) | **2305.20050** ✓ |
| Math-Shepherd | MC auto-label | PRM | step | RL on PRM | **2312.08935** ✓ |
| Search-R1 | per-retrieval rule | rule PRM | step | GRPO | **2503.09516** ✓ |
| R1-Searcher | per-retrieval rule | PRM | step | GRPO | **2503.05592** ✓ |
| ReSearch | reasoning-evidence align | PRM | step | GRPO | **2503.19470** ✓ |
| ReasonRAG | shortest-path (MCTS) | PRM | step | DPO | **2505.14069** ✓ |
| LeTS | process+outcome hybrid | ORM+PRM | step | RL | **2505.17447** ✓ |
| E-GRPO | entity-match-rate partial | PRM | step | GRPO | **2510.24694** ✓ |
| StepSearch | info-gain/redundancy | ORM+PRM | step | PPO | **2505.15107** ✓ |
| IGPO / IG-Search | info-gain pseudo | ORM+PRM | step | IGPO/GRPO | **2510.14967** ✓ / **2604.15148** ✓ |
| CW-GRPO | LLM-judge contribution | ORM+PRM | step | CW-GRPO | **2604.14267** ✓ (ACL26) |
| SLATE | truncated step + decomposed | PRM | step | step-GRPO | **2602.23440** ✓ |
| C-GRPO | citation rubric | PRM | step | GRPO | **2601.06021** ✓ |
| Search-R2 | actor-refiner dense | PRM | single | — | **2602.03647** ✓ |
| PRMBench | PRM benchmark | (eval) | — | — | **2501.03124** ✓ |

> Spectrum read: 2025–2026 search RL is **almost uniformly step-level + PRM** (or ORM+PRM hybrid); pure outcome RL is now a minority — double-sided evidence for doc 03 §5's critique ("PRM universally better" accepted in practice; "judge/heuristic brittleness" inherited in practice).

## 4 · Paper dissection

### PRM800K / Let's Verify (arXiv:2305.20050✓)
Scale human *step-level* labels to ~800k; train $\rho_\phi$; show **best-of-N with $\rho_\phi$ beats outcome-reward best-of-N** on math. Founding result: *process supervision helps*.
**Caveat.** Gain is largest where the task decomposes into checkable steps that humans label reliably. Research/search only partially satisfies the latter; the legacy is not "annotate everything" but **"find the cheap verifier"** — which the search-RL trio does by replacing human labelers with retrieval-time heuristics.

### Math-Shepherd (arXiv:2312.08935✓)
**Auto-label** step rewards by Monte-Carlo: + if continuations frequently reach the correct final answer, − otherwise. Removes the human labeler; PRMs become data-generated — the practical pivot that made process-rewarded RL deployable.
**Subtle point.** The auto-label is **consistent with the labeling-time policy** — if the policy improves, labels change. Feature (self-improving supervision) and bug (PRM lag, labeler/learner distribution shift). Same tension is why R1-style outcome RL can overtake PRM-RL in some regimes: it needs no policy-consistent labeler.
**Cross-doc read.** MC labeling *is* a form of credit assignment (doc 02): it estimates, per step, the "success probability if we reach here" — the step's marginal credit. PRMs and COMA estimate cousins of the same object.

### Search-RL trio: Search-R1 / R1-Searcher / ReSearch
Train LLM agents that **interleave reasoning and retrieval** under RL, with a **per-search-step process signal** (evidence sufficiency / answer grounding) alongside the final-answer reward. Process reward here meets long horizon (doc 01) *and* search-as-MDP (§2).
- **Search-R1** (arXiv:2503.09516✓): each retrieval is an action; reward shaped by whether retrieval improves the partial answer → process reward = the query's marginal information gain.
- **R1-Searcher** (arXiv:2503.05592✓): learns *when to search, what to query*, rewarded by retrieval quality — closest to "verifiable process reward" in the open-ended setting.
- **ReSearch** (arXiv:2503.19470✓): separates reasoning from search state, uses process reward to align the reasoning transition with retrieved evidence — clean because the transition is locally checkable.
Search-RL is the existence proof that **process rewards can be cheaper than outcome verifiers when the action space is search**, because retrieval steps have checkable intermediate states. This is exactly why *the author's multi-agent-search + RL line* lives here: role-specialized search agents share an outcome, and the only affordable credit signal is the search-step process reward — making doc 02's between-group credit problem *solvable from data* (doc 02 §6).

### The credit-via-PRM quartet: ReasonRAG / CW-GRPO / SLATE / E-GRPO
These four weld process reward to credit assignment (doc 02) — the clearest 2025–2026 "PRM-as-credit" spectrum:
- **ReasonRAG** (arXiv:2505.14069✓): systematically shows fine-grained process reward **beats** outcome-only; **5K samples ≈ Search-R1's 90K (18× data efficiency)** — process reward is not just better, it's cheaper.
- **E-GRPO** (arXiv:2510.24694✓): standard GRPO wastes entity info; assigns partial reward proportional to entity-match rate to error samples, learning from "approximately correct" trajectories.
- **SLATE** (arXiv:2602.23440✓): truncated step-level sampling + decomposed process reward (reasoning + query + answer quality); 7B +7.0%, 3B +30.7% vs. Search-R1.
- **CW-GRPO** (arXiv:2604.14267✓, ACL26): LLM judge estimates per-turn contribution and rescales the outcome advantage — fine-grained credit; 8B +5.0%, 1.7B +6.3%.
**Joint read.** This spectrum confirms doc 02's critique: **GRPO's exchangeability breaks on multi-step search, spawning an E-GRPO→CW-GRPO→SLATE→C-GRPO mainstream of "contribution/step-level credit"; but all rely on heuristics or an external judge with no precise causal attribution** — the unresolved junction of between-group credit + auto-PRM.

## 5 · Critique

- **"PRM > ORM, always."** Only under cheap step-verifiability. For research/search the PRM is heuristic (drift) or learned (Math-Shepherd lag); the gain over outcome RL is regime-dependent, not universal. Recent outcome-RL reproduction successes (R1-class) are quietly evidence *against* universal PRM superiority.
- **Auto-PRM policy-consistency under-theorized.** The MC label is policy-defined; few papers analyze the PRM-learner feedback loop. It behaves like self-distillation, with collapse/stagnation risks — unverified in practice.
- **Search reward shape is brittle hand-design.** "Evidence sufficiency / novelty / grounding" differ per paper, rarely ablated together across the trio. No shared, verifiable search-step reward benchmark (PRMBench only evaluates PRMs, not search-step rewards).

## 6 · Open problems

1. **A unified, verifiable search-process reward.** Converge the trio's ad-hoc step rewards into one checkable surrogate (the search analog of a math verifier).
2. **Auto-PRM under multi-agent rollouts.** Math-Shepherd assumes a *single* continuation policy; in heterogeneous search, labeled "eventual success" mixes role contributions. Credit-aware auto-PRM = the open joint of docs 02 ⇆ 03.
3. **PRM-as-critic vs. PRM-as-reward.** Use a PRM as the centralized credit critic (COMA-style, doc 02) rather than merely a reward bonus — barely explored for LLM search agents.
4. **Verifier for research.** Terminal constraint: without a research verifier, even perfect process-reward machinery caps where the verifier is trustworthy.

## References

Let's Verify **2305.20050**✓; Math-Shepherd **2312.08935**✓; Search-R1 **2503.09516**✓; R1-Searcher **2503.05592**✓; ReSearch **2503.19470**✓; ReasonRAG **2505.14069**✓; LeTS **2505.17447**✓; E-GRPO **2510.24694**✓; StepSearch **2505.15107**✓; IGPO **2510.14967**✓; IG-Search **2604.15148**✓; CW-GRPO **2604.14267**✓ (ACL26); SLATE **2602.23440**✓; C-GRPO **2601.06021**✓; Search-R2 **2602.03647**✓; PRMBench **2501.03124**✓. GRPO/Dr.GRPO/DAPO/LLDS/ASTER context — doc 01.

<p align="center"><sub>03 / 03 · Process Reward for Search · cross-refs 01 (horizon), 02 (credit)</sub></p>
