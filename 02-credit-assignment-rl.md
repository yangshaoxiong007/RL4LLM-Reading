<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:9B6BFF,100:22D3EE&height=140&section=header&text=02%20%C2%B7%20Credit%20Assignment%20in%20RL&fontSize=32&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

<p align="center"><sub>🇬🇧 English &nbsp;·&nbsp; <a href="./02-credit-assignment-rl.zh.md">🇨🇳 中文</a> &nbsp;·&nbsp; cross-refs <a href="./01-long-horizon-rl.md">01 Long-Horizon</a> · <a href="./03-process-reward-search.md">03 Process Reward</a></sub></p>

## 1 · Why it matters

Credit assignment asks: given **one shared team reward**, which agent / step / action deserves credit (or blame)? It is the structural twin of long-horizon attribution (doc 01): doc 01 is credit over *time*, this doc over *agents and roles* — same math, different axes. For LLM-agent teams it is newly acute because:

- the "agents" are **role-specialized LLMs** (planner / searcher / verifier / writer) co-producing one whole-trajectory reward;
- one role's failure rarely aborts the team reward deterministically, so the shared reward is *noisier per role*;
- and the natural estimator (GRPO's group baseline, doc 01) is **biased under role heterogeneity** because it assumes exchangeable samples.

A role-heterogeneous team sharing one sparse outcome is the worst case for *all three* threads: long horizon (many search steps), shared reward (team outcome), sparse verification (research, not math).

## 2 · Formalization

A cooperative MADP with $n$ agents: $\pi(\mathbf a\mid\mathbf s)=\prod_i\pi_i(a_i\mid\tau_i)$, shared reward $r$, joint advantage $A^{\text{team}}$. Agent $i$'s gradient:

$$\nabla_\theta J=\mathbb E\big[(A^{\text{team}})\,\nabla_\theta\log\pi_i(a_i\mid\tau_i)\big]$$

Every agent's gradient is scaled by the *same* $A^{\text{team}}$ — a useless and a pivotal agent get the same signal. Credit assignment = replace $A^{\text{team}}$ with a per-agent counterfactual:

$$\hat A_i(s,\mathbf a)\approx Q(s,\mathbf a)-Q(s,(a_{-i},\bar a_i))$$

i.e. "how much better the outcome was because agent $i$ played $a_i$ instead of its default $\bar a_i$." The field is **estimators of this counterfactual** under representational constraints.

## 3 · Method taxonomy

| Family | Counterfactual | Strength | Limit | Representative |
|---|---|---|---|---|
| **Difference rewards / COMA** | $D_i=Q(\mathbf a)-Q(\mathbf a_{-i},c_i)$ | Exact per-agent marginal | Needs learnable centralized $Q$ | COMA (arXiv:1705.08926✓) |
| **Value decomposition (VDN→QMIX→QPLEX)** | $Q_{\text{tot}}=f(Q_1,\dots,Q_n)$, monotone for IGM | Scalable, principled consistency | Monotonicity limits antagonistic roles | QMIX (arXiv:1803.11485✓); QPLEX (NeurIPS21*) |
| **CTDE+PPO (MAPPO)** | train shared $V_\phi$, local actors | Simple, scalable | $V_\phi$ shared → **no per-role credit** | MAPPO (arXiv:2103.01955✓) |
| **Role-aware / heterogeneous grouping** | partition into homogeneous *groups*, baseline within | Restores within-group exchangeability; isolates roles | Group definition = inductive bias | Heterogeneous-group RL (author, ACL26); MADDPG (arXiv:1706.02275✓) |
| **Latent-role discovery (MAVEN)** | learn $z_i$, value on $(z_i,\tau_i)$ | Emergent roles | $z$-conditioning unstable at token level | MAVEN (NeurIPS19*) |
| **Outcome-reweighted fine-grained credit** | rescale outcome advantage by per-turn contribution, no centralized $Q$ | Landable on LLM tokens | Still LLM-judge/heuristic, not causal | CW-GRPO (arXiv:2604.14267✓); SLATE (arXiv:2602.23440✓); E-GRPO (arXiv:2510.24694✓) |

> *QPLEX (NeurIPS21), MAVEN (NeurIPS19): arXiv id not title-matched this session — cited by venue, no fabricated numeric id.*

## 4 · Paper dissection

### COMA (arXiv:1705.08926✓)
Centralized critic computes expected $Q$ over agent $i$'s other actions using teammates' actual actions; counterfactual baseline $=\sum_{a'}\pi_i(a'\mid\tau_i)Q(s,(a_{-i},a'))$. The canonical per-agent signal: "credit only the deviation from what you'd usually do."
**For LLM agents.** COMA's idea — don't credit an agent for the part of the outcome it would have produced anyway — is what GRPO lacks under role heterogeneity. A searcher retrieving the obvious first-hit vs. one doing hard retrieval get the same $A^{\text{team}}$; COMA shrinks the first toward zero.
**Limit.** Needs centralized $Q$ over a combinatorial joint action — painful at LLM-token granularity.

### QMIX (arXiv:1803.11485✓) (+ VDN 1706.05296✓ → QPLEX NeurIPS21*)
Factor $Q_{\text{tot}}$ as a **monotone** hypernet of per-agent $Q_i$ (VDN additive; QMIX monotone; QPLEX duplex-dueling). Monotonicity guarantees **IGM**: $\arg\max_{\mathbf a}Q_{\text{tot}}=(\arg\max_{a_i}Q_i)_i$, so decentralized greedy = global greedy — a clean credit-consistency theorem.
IGM is the field's only rigorous consistency result, bought at a price: **monotonicity forbids representing roles whose marginal contributions are antagonistic.** LLM teams with overlapping roles or mutual correction (planner vs. verifier) are *non-monotone by construction* — QMIX is the wrong inductive bias, a precise, citable reason to prefer **heterogeneous grouping** (no monotonicity, restore within-group exchangeability) over QMIX-style decomposition.

### MAPPO (arXiv:2103.01955✓)
PPO + shared centralized $V_\phi(s)$ + per-agent actors. The decoy of multi-agent LLM RL: "just use MAPPO."
**Sharp critique.** MAPPO trains one shared team advantage; it solves *cross-agent stability*, **not** credit assignment. Mapping LLM roles onto MAPPO actors *silently reverts* to $A^{\text{team}}$ — every role rewarded alike. Its popularity masks that it makes credit *worse*.

### Heterogeneous-Group RL (author's line, ACL 2026)
Partition agents into role-homogeneous *groups*; enforce GRPO exchangeability *within* group (unbiased baseline), handle *between*-group credit via group structure. The bridge between COMA's per-agent counterfactual (expensive) and MAPPO's shared value (credit-blind): cheap unbiased within-group credit, sacrificing only fine-grained between-role differentiation — which grouping makes explicit.
**Why the right surgery on GRPO.** GRPO's bias is exchangeability violation; heterogeneous grouping *restores* it by construction. The open question it leaves — between-group credit without a centralized $Q$ — is where COMA/QMIX machinery could re-enter if tractable.

### CW-GRPO: Contribution-Weighted GRPO (arXiv:2604.14267✓, ACL 2026)
An LLM judge estimates each retrieval turn's *contribution score* and **rescales** the outcome advantage by per-turn contribution — landing COMA's counterfactual-marginal idea (via a judge) onto search-step granularity. Qwen3-8B +5.0%, Qwen3-1.7B +6.3%.
The first landable fine-grained credit on LLM tokens, sharing its estimate object with doc 03 (per-step marginal contribution to outcome). It aligns with the contribution-weighting direction at ACL 2026 (CW-GRPO).
**Unresolved.** Contribution comes from an LLM judge, not precise causal attribution; judge bias contaminates credit — the mirror of doc 03's "search reward shape is brittle hand-design," on the credit side.

## 5 · Critique

- **"CTDE solves credit assignment."** No. CTDE *enables* centralized value sharing; it does not assign credit. Most "multi-agent LLM PPO" papers do MAPPO but claim gains the estimator cannot deliver.
- **IGM/monotonicity is a liability for LLM roles.** The theorem everyone cites (QMIX) *forbids* the non-monotone role interactions that dominate real teams — rarely acknowledged.
- **Role labels as a free lunch.** Fixing roles makes grouping look clean, but in deep research the *specialization itself* should be learned. MAVEN-style latent roles are underexplored for LLM agents (token actions destabilize $z$-conditioning).
- **Judge-based credit circularity.** CW-GRPO/SLATE estimate contribution with an LLM judge co-trained with the policy — self-reinforcement bias risk, no theoretical guarantee (echoing doc 03's auto-PRM policy-consistency issue).

## 6 · Open problems (→ author's line)

1. **Between-group credit without centralized $Q$.** Can process rewards (doc 03) supply the between-group signal that COMA would, cheaply and from data?
2. **Learned, not hand-defined roles.** End-to-end role emergence under shared reward + long horizon.
3. **Non-monotone credit consistency.** A theorem weaker than IGM but stronger than $A^{\text{team}}$ — the missing formal object for LLM teams.
4. **Verifier as credit signal.** For deep research the only reliable per-step credit may be a verifier; making it cheap is doc 03's job.

## References (§ verified)

COMA **1705.08926**✓; QMIX **1803.11485**✓; VDN **1706.05296**✓; MAPPO **2103.01955**✓; MADDPG **1706.02275**✓; CW-GRPO **2604.14267**✓ (ACL26); SLATE **2602.23440**✓; E-GRPO **2510.24694**✓; QPLEX NeurIPS21 / MAVEN NeurIPS19 (by venue); Heterogeneous-group RL (author, ACL26, no arXiv).

<p align="center"><sub>02 / 03 · Credit Assignment · cross-refs 01 (horizon), 03 (process reward)</sub></p>
