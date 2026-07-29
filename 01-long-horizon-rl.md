<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,60:5B8CFF,100:9B6BFF&height=140&section=header&text=01%20%C2%B7%20Long-Horizon%20RL%20for%20LLMs&fontSize=34&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

## 1 · Why it matters

A long-horizon task for an LLM agent is one where the action sequence that *decides* success is separated from the outcome signal by **dozens to hundreds** of generated tokens and, in agentic settings, many tool/search round-trips. Three failure modes follow deterministically:

1. **Sparse & delayed reward.** A math answer or a retrieved-evidence verdict arrives only at the trajectory end. Per-token policy gradients are then driven almost entirely by the value baseline; the signal-to-noise ratio on any *early* decision is O(1/H) in horizon H.
2. **Credit dilution across steps.** Even with a correct outcome, the gradient cannot tell which of the H steps caused it. Outcome-only RL implicitly assumes a *uniform* credit prior, which is wrong for reasoning (a single mis-step aborts the solution) and doubly wrong for multi-step search.
3. **Train/inference length mismatch & reward hacking.** Optimizing terminal outcome over long rollouts rewards *length-as-strategy* and enables shortcuts (repeating, "thinking" without progressing). Length-controlled RL is then needed, but the control (KL, length penalty) trades off against exploration and long-horizon generalization (cf. the "aha moment" and sudden competence jumps in DeepSeek-R1).

The thread is *temporal credit attribution*: how to make reward propagate faithfully across a long LLM rollout.

## 2 · Formalization

Treat the LLM as a policy $\pi_\theta$ over tokens; a reasoning/agent rollout of horizon $H$ is a trajectory $\tau=(s_0,a_0,\dots,s_H,a_H)$ where an "action" at coarse granularity is a reasoning step or tool call. The RL objective is the standard surrogate (PPO-style leave-one-out / GRPO-style group baseline):

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\Big[\,\sum_{t=0}^{H} R(s_t,a_t) - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})\Big]$$

with **outcome reward** $R=\sum_t r_t$ collapsed to a terminal $R(\tau)=r_H$ (sparse, delayed). The three quantities that *long-horizon RL for LLMs* manipulates:

- **Baseline variance.** GRPO replaces the learned critic $V_\phi$ with a **group-relative advantage** over $G$ sampled completions: $\hat A_i = (r_i-\mathrm{mean}(\{r_j\}))/\mathrm{std}(\{r_j\})$. This removes a noisy long-horizon value head — a structural choice dictated by the difficulty of fitting $V$ over $H\!\gg\!100$ steps (see §4).
- **Effective horizon.** Only states/actions near $s_H$ receive informative advantage; early steps are *non-attributed*. Process rewards (§ref, doc 03) are the standard remedy.
- **KL anchoring.** $\beta$ controls drift from the SFT prior; in long horizons too-small $\beta$ → reward hacking / degenerate long rollouts, too-large → no exploration. The recent trend (Dr. GRPO, DAPO) is to **decouple** the KL/length-control knobs from the advantage estimator itself.

## 3 · Method taxonomy

| Family | Core idea | Long-horizon relevance | Representative work |
|---|---|---|---|
| **Outcome RL, group baseline** | Drop the critic; baseline across G samples | Removes a noisy long-horizon value estimate | GRPO (DeepSeekMath, 2402.03300); DeepSeek-R1 (2501.12948) |
| **Outcome RL, decoupled control** | Separate advantage estimator from KL/length penalties | Stops length-hacking while keeping exploration | Dr. GRPO (arXiv:2503.20783); DAPO (arXiv:2503.14476) |
| **RL from verifiable rewards (RLVR / RFT)** | Reward = executable verifier (math judge, code tests, retrieval ground truth) | Makes terminal reward *cheap and faithful* even over long rollouts | R1 cold-start; ReFT (arXiv:2401.08967) |
| **Step/Process-supervised RL** | Reward intermediate steps via a PRM | **Direct attack on horizon** — dense, attributable | Let's Verify (2305.20050); Math-Shepherd (2312.08935) → see doc 03 |
| **Tool/search-integrated RL** | Actions include retrieval/tool calls; reward on final answer | Long horizon *is inter-leaved* with external state | Search-R1 / R1-Searcher / ReSearch → doc 03 |
| **Curriculum / staged horizon** | Grow H during training; cold-start then RL | Mitigates reward sparsity by length budgeting | R1 cold-start→RL pipeline; multi-stage RL |
| **Long-CoT post training** | SFT distilled long reasoning then RL | Bootstraps the *length prior* before RL explores it | R1 → distillation to small models |

## 4 · Paper dissection (expert)

### GRPO — *DeepSeekMath* (arXiv 2402.03300)
**Contribution.** Group-Relative Policy Optimization: sample $G$ completions per prompt, compute advantage as the standardized group reward, **share one baseline across a group, no learned $V_\phi$**. This is the single most important *engineering* choice for long-horizon LLM RL — a learned value head over thinking-rollouts of hundreds of tokens is the dominant source of variance and training pathology, and removing it is both cheaper and stabler.
**Where it's under-theorized.** The group baseline is unbiased *only* when the $G$ samples are exchangeable given the prompt; for reasoning with deterministic verifiers this is approximately true, but for **multi-agent / role-specialized** rollouts the group is *not* exchangeable — the same outcome from a planner vs. a searcher is not symmetric. This is exactly the seam where credit assignment (doc 02) re-enters GRPO.
**Limit.** Still outcome-only; horizon problem is mitigated by variance reduction, not *solved*.

### DeepSeek-R1 (arXiv 2501.12948)
**Contribution.** Pure RLVR (no SFT reasoning data) can elicit long, productive chain-of-thought — including emergent self-reflection / "aha" behavior — using GRPO + verifiable rewards. Demonstrates that **a length prior need not be taught**: RL can grow the horizon organically when rewards are verifiable and the KL anchor is tuned.
**The subtle, load-bearing claim.** "Emergent" long-horizon competence is really *explained* by the verifier providing dense-enough terminal signal and the SFT prior already containing the latent skills; RL *selects and extends* them. Strip the SFT prior quality and the "aha moment" does not appear. Many reproductions failing to show emergence are failing on the prior, not on RL.
**Open wound.** Length generalization (generalizing to harder/longer problems than trained) remains unreliable; reward hacking on unverifiable sub-parts persists.

### Implicit / decoupled length control (Dr. GRPO, DAPO — IDs pending)
**Contribution.** Show that GRPO's standardization and the KL/length penalty are **confounded**: standardizing by group std couples the effective step-size to reward scale, so length control behaves incoherently. Decoupling (Dr. GRPO removes the std normalization; DAPO reformulates clipping/length penalties) stabilizes long-horizon training and recovers more of R1's behavior at small scale.
**Expert read.** These are **corrections to an estimator**, not new paradigms — important because they reveal that much published "emergence" was sensitive to normalization/idiosyncrasies. Reproducibility across groups/horizons is the real metric.

## 5 · Critique — where the field over-claims

- **"Reward sparsity is solved by verifiers"** — only where verification is cheap and exact (math, code, factual retrieval). For *open-ended deep research* verification is itself an unsolved problem; outcome RL there inherits full sparsity. This is precisely why process rewards (doc 03) matter.
- **Horizon is silently length, not depth.** "Long horizon" is conflated with "many tokens." A 5000-token ramble has a *long* horizon but a *shallow* decision frontier; the credit problem is about **depth of causal dependence**, which token-count does not measure. The literature under-specifies this.
- **KL anchoring as a band-aid.** $\beta$ is doing the work that *faithful credit attribution* should. The moment you remove process structure, you rely on $\beta$ to prevent drift — which caps how far long-horizon competence can grow.

## 6 · Open problems (→ the author's line)

1. **Faithful per-step credit without an external PRM.** Can heterogeneous-group structure (doc 02) *replace* a hand-shaped process reward?
2. **Search-as-long-horizon-MDP.** When the horizon is *search iterations* (not just tokens), the state space is the growing evidence context; value estimation and reward shaping there are essentially open.
3. **Verifiable rewards for research, not just math/code.** The bottleneck for long-horizon *deep-research* RL is the verifier, not the algorithm.
4. **Multi-agent + long horizon interaction.** GRPO's exchangeability assumption breaks under role specialization; combining docs 02 ⇆ 01 ⇆ 03 is the unsolved core.

## References (subject to § verification)

- GRPO / DeepSeekMath — **arXiv:2402.03300** ✓
- DeepSeek-R1 — **arXiv:2501.12948** ✓
- Let's Verify Step by Step (PRM/PRM800K) — **arXiv:2305.20050** ✓ (cross-ref doc 03)
- Math-Shepherd — **arXiv:2312.08935** ✓ (cross-ref doc 03)
- Dr. GRPO — **arXiv:2503.20783** ✓ ("Understanding R1-Zero-Like Training")
- DAPO — **arXiv:2503.14476** ✓
- ReFT — **arXiv:2401.08967** ✓

<p align="center"><sub>01 / 03 · Long-Horizon RL · cross-refs 02 (credit), 03 (process reward)</sub></p>
