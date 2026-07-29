<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:5B8CFF,100:9B6BFF&height=140&section=header&text=01%20%C2%B7%20Long-Horizon%20RL%20for%20LLMs&fontSize=32&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

<p align="center"><sub>🇬🇧 English &nbsp;·&nbsp; <a href="./01-long-horizon-rl.zh.md">🇨🇳 中文</a> &nbsp;·&nbsp; cross-refs <a href="./02-credit-assignment-rl.md">02 Credit</a> · <a href="./03-process-reward-search.md">03 Process Reward</a></sub></p>

## 1 · Why it matters

A long-horizon task for an LLM agent is one where the action sequence that *decides* success is separated from the outcome signal by hundreds of tokens and many tool/search round-trips. Three failure modes follow deterministically:

1. **Sparse & delayed reward.** The verdict arrives only at the trajectory end; the SNR on any early decision is O(1/H) in horizon H.
2. **Step credit dilution.** Even with a correct outcome, the gradient cannot tell which step caused it. Outcome-only RL assumes a *uniform* credit prior — wrong for reasoning (one mis-step aborts) and doubly wrong for multi-step search.
3. **Train/inference length mismatch & reward hacking.** Optimizing terminal outcome over long rollouts rewards *length-as-strategy* and shortcuts (LLDS: probability mass displaced onto "easy" tokens rather than "key" ones; ASTER: regression to much internal reasoning, few tool calls).

The thread is *temporal credit attribution*: faithful reward propagation across a long LLM rollout.

## 2 · Formalization

Policy $\pi_\theta$ over tokens; a horizon-$H$ rollout $\tau=(s_0,a_0,\dots,s_H,a_H)$. RL surrogate (PPO leave-one-out / GRPO group baseline):

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\,\sum_{t=0}^{H} R(s_t,a_t) - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\Big]$$

Outcome reward collapses $R=\sum_t r_t$ to terminal $R(\tau)=r_H$. Three quantities long-horizon RL manipulates:
- **Baseline variance.** GRPO replaces a learned critic $V_\phi$ with a *group-relative* advantage $\hat A_i=(r_i-\mathrm{mean})/\mathrm{std}$ — removing a noisy long-horizon value head (the dominant pathology source).
- **Effective horizon.** Only near-$s_H$ states get informative advantage; early steps are *non-attributed* (process rewards, doc 03, fix this).
- **KL anchoring.** $\beta$ controls drift from the SFT prior; too small → hacking/length degeneracy, too large → no exploration. Dr. GRPO/DAPO *decouple* KL/length control from the advantage estimator.

## 3 · Method taxonomy

| Family | Core idea | Long-horizon relevance | Representative |
|---|---|---|---|
| **Outcome RL + group baseline** | Drop critic; baseline across $G$ samples | Removes noisy long-horizon value | GRPO (arXiv:2402.03300✓); DeepSeek-R1 (arXiv:2501.12948✓) |
| **Outcome RL + decoupled control** | Separate estimator from KL/length penalty | Stops length hacking while exploring | Dr. GRPO (arXiv:2503.20783✓); DAPO (arXiv:2503.14476✓) |
| **RLVR / RFT** | Reward = executable verifier | Cheap faithful terminal reward | R1 cold-start; ReFT (arXiv:2401.08967✓) |
| **Step/Process-supervised RL** | PRM rewards intermediate steps | Direct attack on horizon | Let's Verify (arXiv:2305.20050✓); Math-Shepherd (arXiv:2312.08935✓) → doc 03 |
| **Collapse diagnosis + regularization** | Analyze & fix policy degradation on long rollouts | Targets long-horizon pathology | LLDS (arXiv:2512.04220✓); ASTER (arXiv:2602.01204✓) |
| **Tool/search-integrated RL** | Actions include retrieval/tools; reward on final answer | Horizon *interleaved* with external state | Search-R1/R1-Searcher/ReSearch → doc 03 |

## 4 · Paper dissection (expert)

### GRPO — *DeepSeekMath* (arXiv:2402.03300✓)
Sample $G$ completions per prompt, standardized group reward, **one shared baseline, no learned $V_\phi$**. The single most important *engineering* choice for long-horizon LLM RL: a learned value head over hundred-token thinking-rollouts is the dominant variance/pathology source; dropping it is cheaper and stabler.
**Under-theorized.** The group baseline is unbiased *only* when the $G$ samples are exchangeable given the prompt — holds for verifiable math but **not** for *multi-agent/role-heterogeneous* rollouts (planner vs. searcher are asymmetric). This is the seam where credit assignment (doc 02) re-enters GRPO.
**Limit.** Still outcome-only; horizon mitigated by variance reduction, not *solved*.

### DeepSeek-R1 (arXiv:2501.12948✓)
Pure RLVR (no SFT reasoning data) elicits long productive CoT with emergent self-reflection/"aha". Shows a length prior *need not be taught* — RL grows horizon organically when rewards are verifiable and KL tuned.
**Load-bearing claim.** "Emergence" is *explained* by: the verifier gives dense-enough terminal signal and the SFT prior already contains latent skills; RL selects and extends them. Strip prior quality and the "aha moment" disappears — many reproduction failures are prior failures, not RL failures.
**Open wound.** Length generalization unreliable; reward hacking on unverifiable sub-parts persists.

### LLDS: GRPO Collapse (arXiv:2512.04220✓)
Diagnoses **Lazy Likelihood Displacement (LLD)**: the model gains probability on "easy" tokens, not "key" ones; three stages (stagnation → decay → collapse). Root cause: group-relative advantage is *unstable on long search trajectories*. LLDS regularization recovers ~45%.
**Expert read.** The overdue bill for GRPO's "drop the critic" engineering shortcut: its cost erupts as collapse on long search horizons. Not an isolated bug but the structural fragility of "long-horizon + outcome-only RL."

### ASTER: Interaction Collapse (arXiv:2602.01204✓)
A second collapse mode: in RL, internal reasoning has denser per-step reward than tool calls (whose feedback is sparse/delayed), so the model regresses to *much internal reasoning, few tool calls*. 4K interaction-dense cold start mitigates but lacks generality.
**Joint read with LLDS.** LLDS is length/probability-displacement collapse; ASTER is tool-vs-reasoning signal-density collapse — both point to one root: **on long horizons, outcome-only RL pushes the policy toward the densest-signal (not necessarily best) behavior.** This is exactly where process reward (doc 03) must intervene.

### Dr. GRPO / DAPO (arXiv:2503.20783✓ / 2503.14476✓)
GRPO's standardization and KL/length penalty are *confounded*: standardizing by group std couples effective step-size to reward scale, so length control behaves incoherently. Decoupling stabilizes long-horizon training and recovers more R1 behavior at small scale.
**Expert read.** Corrections to an *estimator*, not new paradigms — important because they reveal that much published "emergence" was sensitive to normalization/idiosyncrasy. Cross-group/horizon reproducibility is the real metric.

## 5 · Critique — where the field over-claims

- **"Verifiers solved sparsity"** — only where verification is cheap and exact (math, code, factual retrieval). For open-ended research, verification is itself open; outcome RL there inherits full sparsity. Hence process rewards (doc 03).
- **Horizon conflated with length, not depth.** A 5000-token ramble is long-horizon but decisionally shallow; credit is about **depth of causal dependence**, which token-count doesn't measure. Under-specified in the literature.
- **KL anchoring as a band-aid.** $\beta$ does the work *faithful credit attribution* should. Remove process structure and $\beta$ caps how far long-horizon competence can grow.

## 6 · Open problems (→ author's line)

1. **Faithful step credit without an external PRM.** Can heterogeneous-group structure (doc 02) *replace* a hand-shaped process reward?
2. **Search-as-long-horizon-MDP.** When the horizon is *search iterations*, state = growing evidence context; value estimation and reward shaping are essentially open.
3. **Verifiable rewards for research, not just math/code.** The bottleneck for long-horizon deep-research RL is the verifier.
4. **Multi-agent × long horizon.** GRPO's exchangeability breaks under role specialization; combining docs 02 ⇆ 01 ⇆ 03 is the unsolved core.

## References (§ verified)

GRPO/DeepSeekMath **2402.03300**✓; DeepSeek-R1 **2501.12948**✓; LLDS **2512.04220**✓; ASTER **2602.01204**✓; Dr. GRPO **2503.20783**✓; DAPO **2503.14476**✓; ReFT **2401.08967**✓; Let's Verify **2305.20050**✓; Math-Shepherd **2312.08935**✓.

<p align="center"><sub>01 / 03 · Long-Horizon RL · cross-refs 02 (credit), 03 (process reward)</sub></p>
