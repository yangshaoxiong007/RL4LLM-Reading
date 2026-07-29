<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:5B8CFF,40:9B6BFF,75:22D3EE,100:F5C451&height=200&section=header&text=Problem%20Diagnosis%20%C2%B7%20Innovation%20Directions&fontSize=40&fontColor=ffffff&fontAlignY=42&fontAlign=50&desc=12%20failure%20modes%20%C2%B7%207%20innovation%20directions&descSize=14&descAlignY=66&animation=fadeIn"/>

> **Note 04 · The synthesis chapter.** Collapses the failure modes surfaced across Notes 01/02/03 (long-horizon RL / credit assignment / process reward for search) into **12 structural problems**, then gives **7 innovation directions**—each reduced to a concrete method-combination transfer scheme.

<p align="center">
  <b>🇬🇧 <a href="./04-problems-and-innovations.md">English</a></b> &nbsp; <b>🇨🇳 <a href="./04-problems-and-innovations.zh.md">中文</a></b>
  &nbsp; · &nbsp; <a href="./README.md">← back to index</a>
</p>

---

## 1. Why this note

The first three notes each follow one axis: long-horizon RL pushes reward **tens of steps downstream** on the *time* axis; credit assignment asks *who* deserves credit along the *structure* axis (multi-agent / multi-step); process reward for search is the *mechanism* that makes both tractable—distilling a sparse terminal outcome into dense inspectable step signals. But when the three are stacked on one multi-agent search system, **the real bottleneck is not any single method but the seams between them**. This note enumerates the failure modes at those seams and assigns each a buildable innovation direction, so the three axes collapse into a single agenda—**multi-agent search + RL**.

Each problem row carries: **severity**, **layer**, **representative paper (verified)**, **quantified behaviour**, **best current fix and its limits**.

---

## 2. The twelve structural problems

| # | Problem | Layer | Severity | Rep. paper | Quantified behaviour | Best current fix |
|---|---------|-------|----------|------------|----------------------|------------------|
| 1 | GRPO training collapse | Training | 🔴 high | [LLDS](https://arxiv.org/abs/2512.04220) | root cause Lazy Likelihood Displacement; stall→decay→collapse | LLDS regularization **+45.2%**, symptomatic only |
| 2 | Interaction collapse (degrades to internal reasoning, drops tools) | Training | 🔴 high | [ASTER](https://arxiv.org/abs/2602.01204) | tool feedback sparse/late, internal-reasoning reward denser | 4K tool-dense cold start, not general |
| 3 | Credit assignment | Training | 🟠 mid-high | [CW-GRPO](https://arxiv.org/abs/2604.14267)· [SLATE](https://arxiv.org/abs/2602.23440)· [E-GRPO](https://arxiv.org/abs/2510.24694) | multi-round causal deps; outcome gives only group-level reward | contribution weighting / step sampling / entity partial reward—heuristics or external judge |
| 4 | Plan anchoring (clings to initial plan) | Strategy | 🟠 mid-high | [WebAnchor](https://arxiv.org/abs/2601.03164) | plan+exec in one model, early-success plan over-reinforced | two-stage plan/exec RL; BrowseComp 46.0% / GAIA 76.4% |
| 5 | Tunnel vision (noisy first retrieval biases direction) | Strategy | 🟠 mid-high | [SIGHT](https://arxiv.org/abs/2602.11551) | no explore-exploit balance, no "abandon & restart" | info-gain branching + self-evidence |
| 6 | Over-searching | Strategy | 🟡 mid | [beta-GRPO](https://arxiv.org/abs/2505.17281) (Search Wisely) | **27.7%** avoidable searches | confidence gating; 3B +4% EM |
| 7 | Context rot | Long-horizon | 🔴 high | [RE-TRAC](https://arxiv.org/abs/2602.02486)· [Pensieve/StateLM](https://arxiv.org/abs/2602.12108)· [ARC](https://arxiv.org/abs/2601.12030) | no state mgmt → BrowseComp-Plus **~5%**; with mgmt **52%** | recursive compress / active curate / reflection |
| 8 | Search-stopping timing | Long-horizon | 🟠 mid-high | [DeepSearchQA](https://arxiv.org/abs/2601.20975) | premature stop vs recall inflation coexist | no mature sufficiency-reasoning solution |
| 9 | Retriever–reasoner split | Retrieval | 🟠 mid-high | [CoSearch](https://arxiv.org/abs/2604.17555) | fixed retriever vs oracle gap **26.8% F1** | joint training |
| 10 | Weak multimodal search | Synthesis | 🔴 high | [VISOR](https://arxiv.org/abs/2604.09508)· [LMM-Searcher](https://arxiv.org/abs/2604.12890) | best systems ~**40%** on multimodal benchmarks | unified action space nascent, cross-modal reward immature |
| 11 | Hallucination / citation errors | Synthesis | 🟠 mid-high | [C-GRPO](https://arxiv.org/abs/2601.06021) | citation stuffing / brute-force / claim-evidence mismatch | citation reward / 3-layer verification |
| 12 | Fragmented evaluation | Evaluation | 🟡 mid | 15+ benchmarks | divergent axes, no joint long/credit/process eval | no unified standard |

> **Root-cause thread**: ①③ = instability of GRPO advantage estimation when a search agent is trained as one long trajectory; ⑦ = the bounded lossy channel of a fixed context window; ⑨ = decoupling retriever and reasoner at training. These three are where the seven directions actually push.

---

## 3. Problem → innovation map

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/problem-innovation-map.png" width="96%" alt="Problem→innovation map: 12 structural problems grouped by layer flow into 7 innovation directions"/></p>

<sub align="center">The 12 problems, grouped Training / Strategy / Long-horizon / Retrieval&Synthesis, flow into 7 innovations along edges that resolve *root causes* rather than *symptoms*; each edge direction = the root cause the innovation removes.</sub>

```
Training(①③) ──▶  I. Adaptive hierarchical reward
Training(②)   ───▶  VII. Self-evolving search world model
Strategy(④⑤) ──▶  IV. Counterfactual search planning
Strategy(⑥)   ───▶  V. Search-budget Pareto
Long-horizon(⑦⑧) ──▶  II. Markov search state machine
Retrieval(⑨)  ───▶  III. Agent-Retriever co-evolution
Synthesis(⑩⑫) ──▶  VI. Unified multimodal search RL
Synthesis(⑪)  ───▶  I. Adaptive hierarchical reward (fine signal)
```

---

## 4. Seven innovation directions (method-combination transfer)

> Each direction = **root causes removed** + **verified methods fused** + **mature field transferred from** + **scheme** + **expected breakthrough (quantified)**.

### Direction I · Adaptive hierarchical reward architecture

- **Releases**: ①GRPO long-trajectory collapse, ③credit assignment, ⑪citation errors
- **Fuses**: [CW-GRPO](https://arxiv.org/abs/2604.14267) (per-round contribution, ACL 2026) · [SLATE](https://arxiv.org/abs/2602.23440) (step-level truncated sampling + decomposed process reward, 7B +7.0% / 3B +30.7%) · [LeTS](https://arxiv.org/abs/2505.17447) (annotation-free process reward) · [IG-Search](https://arxiv.org/abs/2604.15148) (info gain)
- **Transferred from**: cybernetics—**hierarchical feedback control**
- **Scheme**: three reward tiers—*micro token-level* (marginal info gain) → *meso step-level* (per search-reason cycle) → *macro trajectory-level* (outcome + efficiency penalty); weights shift by training stage (early→micro, late→macro).
- **Expected**: stabilizes GRPO long-trajectory training with no process-annotation cost, replicating ReasonRAG-grade **18× data efficiency** (5K≈90K, [ReasonRAG](https://arxiv.org/abs/2505.14069)).

### Direction II · Markov search state machine + recursive context compression

- **Releases**: ⑦context rot, ⑧stopping timing
- **Fuses**: [AoT](https://arxiv.org/abs/2502.12018) (Markov reasoning, NeurIPS 2025) · [RE-TRAC](https://arxiv.org/abs/2602.02486) (4-D state summary: findings/gaps/obstacles/directions, +15–20%) · [Pensieve/StateLM](https://arxiv.org/abs/2602.12108) (active context curation, 52% vs ~5%)
- **Transferred from**: operating systems—**virtual-memory paging (L1/L2/L3)**
- **Scheme**: model search as an MDP, each step depends only on the current "search state" (RE-TRAC 4-D summary) not full history; 3-level cache L1 immediate / L2 sub-task summary / L3 verified-fact store; RL learns L1↔L2↔L3 migration.
- **Expected**: supports unbounded search steps; fixed-size per-step state lifts BrowseComp-Plus from ~5% to 50%+.

### Direction III · Agent-Retriever co-evolution

- **Releases**: ⑨retriever–reasoner split
- **Fuses**: [CoSearch](https://arxiv.org/abs/2604.17555) (joint train reasoning+ranking) · [AgentIR](https://arxiv.org/abs/2603.04384) (reasoning-aware retrieval) · [Cycle-Consistent Search](https://arxiv.org/abs/2604.12967) (cycle-consistency proxy reward)
- **Transferred from**: GAN-style **adversarial / co-evolutionary** training
- **Scheme**: two modules (reasoning agent emits+consumes / retriever reads reasoning context); joint GRPO—agent reward=correctness, retriever reward=result-usage rate (implicit), synergy reward=cycle consistency; adversarial difficulty escalates distractors.
- **Expected**: closes CoSearch's measured **26.8% F1** retriever gap.

### Direction IV · Counterfactual search planning & adaptive strategy switching

- **Releases**: ④plan anchoring, ⑤tunnel vision
- **Fuses**: [WebAnchor](https://arxiv.org/abs/2601.03164) (plan-exec separation) · [SIGHT](https://arxiv.org/abs/2602.11551) (info-gain branching) · MCTS explore-exploit
- **Transferred from**: causal inference—**counterfactual reasoning**
- **Scheme**: counterfactual evaluator generates 2–3 alternative plans, light simulation estimates expected info gain; auto-triggers after N low-gain steps; 2–3-layer light MCTS at SIGHT high-gain nodes; RL trains an across-strategy (DFS/BFS/verify/backtrack) meta-policy.
- **Expected**: sharply reduces anchoring & tunnel vision on hard multi-hop, pushing past WebAnchor's BrowseComp 46% baseline.

### Direction V · Search-budget-aware Pareto-optimal training

- **Releases**: ⑥over-searching
- **Fuses**: [AutoSearch](https://arxiv.org/abs/2604.17337) (adaptive depth) · [beta-GRPO](https://arxiv.org/abs/2505.17281) (confidence gating) · [Ares](https://arxiv.org/abs/2603.07915) (effort selection, −52.7% tokens)
- **Transferred from**: multi-objective optimization—**Pareto front**
- **Scheme**: dual-objective RL (quality vs cost); train a *family* covering the front, not a single point; inference-time budget routing (fast/standard/deep); progressive refinement.
- **Expected**: **50%+ search-cost cut** at iso-quality, hedging beta-GRPO's 27.7% avoidable searches.

### Direction VI · Unified multimodal search RL

- **Releases**: ⑩weak multimodal, ⑫fragmented eval (multimodal axis)
- **Fuses**: [VISOR](https://arxiv.org/abs/2604.09508) (visual retrieval RL) · [LMM-Searcher](https://arxiv.org/abs/2604.12890) (external offload, 100-round search) · [MM-DeepResearch](https://arxiv.org/abs/2603.01050) (hypergraph QA) · [ProMMSearchAgent](https://arxiv.org/abs/2604.20486) (process-oriented multimodal reward)
- **Transferred from**: multi-task learning—**modality-adaptive routing**
- **Scheme**: unified action space (text/image/video_search, code_exec, file_read); modality-aware rewards; light router picks modality; multimodal context compression.
- **Expected**: lifts multimodal benchmarks from ~40% to 60%+ and drives multimodal-eval convergence.

### Direction VII · Self-evolving search world model

- **Releases**: ②interaction collapse (env cost), ⑦training-env instability
- **Fuses**: [LiteResearcher](https://arxiv.org/abs/2604.17931) (virtual-world scalable RL, 4B GAIA 71.3%) · [ZeroSearch](https://arxiv.org/abs/2505.04588) (LLM-simulated retrieval) · [WebFactory](https://arxiv.org/abs/2603.05044) (env synthesis)
- **Transferred from**: model-based RL—**world models (Dreamer family)**
- **Scheme**: train a model predicting "given a query what the engine returns"; dual loop—inner low-cost RL, outer real-Web validation+finetune; adversarial world model simulates failure modes; multi-domain-specialized models cooperate.
- **Expected**: **90%+ RL-cost cut** while preserving real-Web generalization, directly removing interaction collapse's environmental root cause.

---

## 5. Priority (impact × feasibility)

| Rank | Direction | Expected impact | Difficulty | Short-term feasibility | Entry |
|------|-----------|-----------------|------------|------------------------|-------|
| 1 | II Markov state machine | ⭐⭐⭐⭐⭐ | mid | high | **prioritize** |
| 2 | I hierarchical reward | ⭐⭐⭐⭐⭐ | mid | high | **prioritize** |
| 3 | V search-budget Pareto | ⭐⭐⭐⭐ | mid | high | quick engineering win |
| 4 | III agent-retriever co-evolution | ⭐⭐⭐⭐ | high | mid | mid-term |
| 5 | IV counterfactual planning | ⭐⭐⭐⭐ | high | mid | mid-term |
| 6 | VI unified multimodal RL | ⭐⭐⭐⭐⭐ | very high | low | long-term |
| 7 | VII self-evolving world model | ⭐⭐⭐⭐⭐ | very high | low | long-term |

> I and II top the impact×feasibility product—they land squarely on the two largest root causes exposed across Notes 01–03 (credit assignment and context rot), and this research line (multi-agent search + RL) needs both resolved simultaneously.

---

## 6. Convergence with the author's research line

Four shortest paths bite directly on **end-to-end RL optimization of LLM-driven multi-agent search systems**:
- **I** supplies hierarchical advantage estimation for an heterogeneous-group multi-agent setting—resolving credit assignment when heterogeneous role-LLMs share a team reward;
- **II** frees multi-agent long-horizon search from context rot, with state passing across agents;
- **III** generalizes the retriever–reasoner split into a unified "decoupled multi-agent sub-modules" training paradigm;
- **VII** brings world-model cheap interaction to make end-to-end multi-agent RL affordable.

Relevant publications on the author's homepage.

---

## 7. References

All listed works are verifiable on arXiv by title:

`LLDS 2512.04220` · `ASTER 2602.01204` · `CW-GRPO 2604.14267` · `SLATE 2602.23440` · `E-GRPO 2510.24694` · `Search-R1 2503.09516` · `R1-Searcher 2503.05592` · `ReSearch 2503.19470` · `ReasonRAG 2505.14069` · `LeTS 2505.17447` · `IG-Search 2604.15148` · `C-GRPO 2601.06021` · `WebAnchor 2601.03164` · `SIGHT 2602.11551` · `beta-GRPO 2505.17281` · `RE-TRAC 2602.02486` · `Pensieve/StateLM 2602.12108` · `ARC 2601.12030` · `DeepSearchQA 2601.20975` · `Rerank 2601.14224` · `CoSearch 2604.17555` · `AutoSearch 2604.17337` · `Cycle-Consistent 2604.12967` · `AgentIR 2603.04384` · `ZeroSearch 2505.04588` · `DeepDiver 2505.24332` · `Ares 2603.07915` · `AggAgent 2604.11753` · `AoT 2502.12018` · `VISOR 2604.09508` · `LMM-Searcher 2604.12890` · `MM-DeepResearch 2603.01050` · `ProMMSearchAgent 2604.20486` · `LiteResearcher 2604.17931` · `WebFactory 2603.05044` · `Search More Think Less 2602.22675` · `Kimi k1.5 2501.12599`.

QPLEX (NeurIPS 2021) and MAVEN (NeurIPS 2019) are cited by venue.

---

<p align="center"><sub>Diagnosis and schemes draw on 100+ 2025–2026 papers.</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:F5C451,50:22D3EE,100:5B8CFF&height=90&section=footer&text=Diagnose%20the%20seams%2C%20not%20the%20symptoms.&fontSize=13&fontColor=ffffff&fontAlignY=45&animation=blink"/>
