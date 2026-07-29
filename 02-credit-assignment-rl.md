<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,60:9B6BFF,100:22D3EE&height=140&section=header&text=02%20%C2%B7%20Credit%20Assignment%20in%20RL&fontSize=34&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

## 1 · Why it matters

Credit assignment asks: given a **single shared team reward**, which agent / which step / which action deserves the credit (or blame)? It is the structural twin of long-horizon attribution (doc 01): doc 01 is credit over *time*, this doc is credit over *agents and roles*. They are the same mathematics in different axes. For LLM-agent systems the problem is newly acute because:

- The "agents" are **role-specialized LLMs** (planner / searcher / verifier / writer) that co-produce one whole-trajectory reward;
- one role's failure rarely aborts the team reward deterministically (unlike a single mis-step in math), so the shared reward is **noisier per role**;
- and the natural RL estimator (GRPO's group baseline, doc 01) is **biased under role heterogeneity** because it assumes exchangeable samples.

So a heterogeneous agent team receiving one sparse outcome is the worst case for *all three* threads at once: long horizon (many search steps), shared reward (team outcome), sparse verification (research, not math).

## 2 · Formalization

A cooperative MADP with $n$ agents: joint policy $\pi(\mathbf a\mid \mathbf s)=\prod_i \pi_i(a_i\mid \tau_i)$, shared reward $r(\mathbf s,\mathbf a)$, joint advantage $A^{\text{team}}$. Gradient of agent $i$:

$$\nabla_\theta J = \mathbb E\Big[\,(A^{\text{team}})\,\nabla_\theta \log\pi_i(a_i\mid\tau_i)\,\Big]$$

The pathology is explicit: **every** agent's gradient is scaled by the *same* $A^{\text{team}}$ — the team advantage. A useless agent gets the same learning signal as a pivotal one. Credit assignment = replacing $A^{\text{team}}$ with a per-agent counterfactual advantage $\hat A_i$ that isolates agent $i$'s marginal contribution:

$$\hat A_i(s,\mathbf a)\approx Q(s,\mathbf a)-Q(s, (a_{-i},\,\bar a_i))$$

i.e. *"how much better was the outcome because agent $i$ played $a_i$ rather than its default/average $\bar a_i$?"* The whole field is **estimators of this counterfactual** under different representational constraints.

## 3 · Method taxonomy

| Family | Counterfactual form | Strength | Limit | Representative |
|---|---|---|---|---|
| **Difference rewards / COMA** | $D_i=Q(\mathbf a)-Q(\mathbf a_{-i},c_i)$ at counterfactual $c_i$ | Exact per-agent marginal | Needs a learnable centralized $Q$ | COMA (1705.08926) |
| **Value decomposition (VDN→QMIX→QPLEX)** | Assume $Q_{\text{tot}}=f(Q_1,\dots,Q_n)$, monotone for credit-consistency | Scalable; principled "IGM" constraint | Monotonicity (QMIX) limits representation of antagonistic roles | VDN (1706.05296); QMIX (1803.11485); QPLEX (NeurIPS 2021) |
| **Centralized training, decentralized exec (CTDE), PPO** | Train value $V_\phi$ centralized; actors local | Simple, scales to many agents | $V_\phi$ shared → **no per-agent credit**, back to $A^{\text{team}}$ | MAPPO (2103.01955) |
| **Role-aware / heterogeneous grouping** | Partition agents into homogeneous *groups*, baseline within group | Restores exchangeability *within* group; isolates *between*-group roles | Group definition is a design/inductive-bias choice | Heterogeneous-group RL (author, ACL 2026); MADDPG lineage (1706.02275) |
| **Latent-role discovery (MAVEN)** | Learn latent role $z_i$, value conditioned on $(z_i,\tau_i)$ | Roles emerge, not hand-designed | Credit now depends on $z$ quality; training instability | MAVEN (NeurIPS 2019) |
| **LLM-agent turn-level credit** | Assign per-turn / per-tool-call credit via outcome decomposition | Brings PRM machinery (doc 03) to agents | Decomposition often hand-crafted | Search-R1/R1-Searcher/ReSearch (doc 03); emerging |

## 4 · Paper dissection (expert)

### COMA — Counterfactual MAC (arXiv 1705.08926) ✓
**Contribution.** Centralized critic computes the expected $Q$ over agent $i$'s *other* actions, using the actual actions of teammates; the counterfactual baseline is $\sum_{a'}\pi_i(a'\mid\tau_i)\,Q(s,(a_{-i},a'))$. This is the **canonical** per-agent credit signal: it is the policy-gradient equivalent of *"credit only the deviation from what you'd usually do."*
**Why it matters for LLM agents.** COMA's idea — don't reward an agent for the part of the outcome it would have produced anyway — is exactly what GRPO is missing under role heterogeneity. A searcher that retrieved the obvious first-hit gets the same $A^{\text{team}}$ as a searcher that did a hard retrieval; COMA would shrinks the first case's credit toward zero.
**Limit.** Needs a centralized $Q$ over a combinatorial joint action — tractable for discrete small $n$, painful for LLM-token-level actions.

### QMIX (arXiv 1803.11485) ✓ (and VDN 1706.05296 ✓ → QPLEX ⟨?⟩)
**Contribution.** Factor $Q_{\text{tot}}$ as a **monotone** hypernet of per-agent $Q_i$ (VDN is the additive special case; QMIX relaxes additivity to monotonicity; QPLEX further to duplex dueling). Monotonicity guarantees the **IGM (Individual-Global-Max)** condition: $\arg\max_{\mathbf a}Q_{\text{tot}}=(\arg\max_{a_i}Q_i)_i$, so decentralized greedy is globally greedy — a clean *credit-consistency* theorem.
**Expert read.** The IGM guarantee is the field's only rigorous credit-consistency result, and it is bought at a price: **monotonicity forbids representing roles whose marginal contributions are antagonistic** (one agent's gain is another's loss of credit). LLM-agent teams with overlapping roles (two searchers, planner vs. verifier that catch each other) are *non-monotone by construction* — QMIX is the wrong inductive bias for them, and this is a precise, citable reason to prefer **heterogeneous grouping** (no monotonicity claim, restore exchangeability *within group*) over QMIX-style decomposition.
**Limit.** Representation gap for non-monotone roles; QPLEX mitigates but at complexity cost.

### MAPPO (arXiv 2103.01955) ✓
**Contribution.** PPO with a shared centralized value $V_\phi(s)$ and per-agent actors. The decoy of multi-agent LLM RL: "just use MAPPO." 
**The sharp critique.** MAPPO trains **one shared team advantage** (or an agent-shared value); it solves *learning stability across agents*, **not credit assignment**. If you map LLM-agent roles onto MAPPO actors, you have *silently reverted to $A^{\text{team}}$* — every role rewarded alike. The popularity of MAPPO as "the multi-agent baseline" masks that it makes the credit problem *worse*, not solved.

### Heterogeneous-Group RL (the author's line, ACL 2026)
**Contribution / positioning.** Partition agents into role-homogeneous **groups**; enforce GRPO-style exchangeability **within** a group (so the group baseline is unbiased) and handle **between**-group credit via group-level structure. This is the bridge between COMA's per-agent counterfactual (expensive) and MAPPO's shared value (credit-blind): it buys *unbiased within-group* credit cheaply, sacrificing only the fine-grained *between*-role differentiation — which group structure makes explicit rather than implicit.
**Why this is the right surgery on GRPO.** GRPO's bias is exchangeability violation; heterogeneous grouping *restores* exchangeability by construction. The open question it leaves — between-group credit without a full centralized $Q$ — is precisely where COMA/QMIX machinery could be re-introduced if tractable.

## 5 · Critique

- **"CTDE solves credit assignment."** No. CTDE *enables* centralized value sharing; it does not assign credit. Most "multi-agent PPO for LLM" papers are doing MAPPO and claiming credit gains that the estimator cannot deliver.
- **IGM / monotonicity is a liability for LLM roles.** The theorem everyone cites (QMIX) is exactly what *forbids* the non-monotone role interactions that dominate real agent teams. It is rarely acknowledged that adopting QMIX-style decomposition imposes this constraint.
- **Role labels as a free lunch.** Treating roles as fixed/known makes heterogeneous grouping look clean — but in deep research the role *specialization itself* should be learned. MAVEN-style latent roles are underexplored for LLM agents because the action space (tokens) makes $z$-conditioned estimation unstable.

## 6 · Open problems (→ the author's line)

1. **Between-group credit without centralized $Q$.** The honest gap in heterogeneous-group RL: can process rewards (doc 03) provide the between-group signal that COMA would, but cheaply and from data?
2. **Learned, not hand-defined roles.** End-to-end role emergence under shared reward + long horizon.
3. **Non-monotone credit-consistency.** A theorem weaker than IGM but stronger than $A^{\text{team}}$ — the missing formal object for LLM agent teams.
4. **Verifier as credit signal.** For deep research the only reliable per-step credit may be a verifier; making that verifier cheap is doc 03's problem.

## References (subject to § verification)

- COMA — **arXiv:1705.08926** ✓
- QMIX — **arXiv:1803.11485** ✓
- VDN — **arXiv:1706.05296** ✓
- MAPPO — **arXiv:2103.01955** ✓
- MADDPG — **arXiv:1706.02275** ✓
- QPLEX — NeurIPS 2021 (arXiv ID unconfirmed; cite by venue)
- MAVEN — NeurIPS 2019 (arXiv ID unconfirmed; cite by venue)
- Heterogeneous-group RL for LLM multi-agent search — ACL 2026 (author), no arXiv

<p align="center"><sub>02 / 03 · Credit Assignment · cross-refs 01 (horizon), 03 (process reward)</sub></p>
