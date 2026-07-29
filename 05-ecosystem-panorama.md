<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:22D3EE,30:5B8CFF,70:9B6BFF,100:F5C451&height=200&section=header&text=Ecosystem%20Panorama&fontSize=42&fontColor=ffffff&fontAlignY=42&fontAlign=50&desc=DeepSearch%20%2F%20DeepResearch%20stack%20%C2%B7%20full%20paper%20index&descSize=14&descAlignY=66&animation=fadeIn"/>

> **Note 05 · The panorama.** Reads DeepSearch / DeepResearch as a seven-layer stack: foundation → data/trajectory → RL training → agent architecture → long-horizon context → synthesis/retrieval → evaluation. Each layer carries representative papers with quantified results, the seams between layers (where most SOTA bottlenecks live, not within layers), and a full paper index grouped by direction at the end. This is the bird's-eye view that situates Notes 01–04 on one map.

<p align="center">
  <b>🇬🇧 <a href="./05-ecosystem-panorama.md">English</a></b> &nbsp; <b>🇨🇳 <a href="./05-ecosystem-panorama.zh.md">中文</a></b>
  &nbsp; · &nbsp; <a href="./README.md">← back to index</a>
</p>

---

## 1. Why a panorama

Notes 01–04 deep-read long-horizon RL, credit assignment, process reward for search, and the problem/innovation synthesis — but all sit inside a larger stack. A panorama answers three questions: (1) which layer a paper lives in and who it competes with; (2) where the seams between layers are (most SOTA bottlenecks are seams, not within-layer); (3) why the evaluation layer chronically lags the method layer. This note organizes 2023.11–2026.04 work into seven layers and ends with a full index.

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/ecosystem-panorama.png" width="96%" alt="DeepSearch/DeepResearch 7-layer stack panorama: foundation→data→RL→architecture→long-context→synthesis→evaluation"/></p>

<sub align="center">Seven layers bottom-up: foundation → data/trajectory → RL training → agent architecture → long-horizon context → synthesis & retrieval → evaluation. Right column marks representative seam problems.</sub>

---

## 2. The seven layers

### L1 · Foundation

| Model | Role | Key property |
|-------|------|--------------|
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Nature) | Basis of pure-RLVR long CoT | GRPO + verifiable rewards, no SFT reasoning data |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | Long-context RL reasoning | AIME 77.5 / MATH 500 96.2 |
| o1/o3, Qwen3, GPT-5, Claude, Gemini | General base | — |
| 4B-class ([LiteResearcher](https://arxiv.org/abs/2604.17931), ORBIT) | Open-source democratization | 4B matches commercial systems on GAIA |

### L2 · Data / Trajectory (35+ papers)

Five paradigms coexist: ① pure RL (no trajectories) — [R1-Searcher](https://arxiv.org/abs/2503.05592)/[Search-R1](https://arxiv.org/abs/2503.09516); ② teacher-distillation SFT — [OpenSeeker](https://arxiv.org/abs/2603.15594) 11.7K denoised trajectories reaching frontier; ③ mixed SFT+RL — Mind DeepResearch 4-stage; ④ offline preference — [OffSeeker](https://arxiv.org/abs/2601.18467), [WebThinker](https://arxiv.org/abs/2504.21776); ⑤ self-evolving/env synthesis — [CoEvolve](https://arxiv.org/abs/2604.15840), [WebFactory](https://arxiv.org/abs/2603.05044), [LiteResearcher](https://arxiv.org/abs/2604.17931), [ZeroSearch](https://arxiv.org/abs/2505.04588).

> Seam: L2 caps L3's training. DeepDive-32B 4.1K→BrowseComp 15.3%; MiroThinker 147K→10.6%; OpenSeeker 11.7K→29.5%. Quality ≫ quantity; *denoised trajectory* is the watershed.

### L3 · RL Training (40+) → see [01](./01-long-horizon-rl.md), [02](./02-credit-assignment-rl.md), [03](./03-process-reward-search.md)

GRPO and variants are the de-facto standard (CW/E/C/beta/Anchor/step-level/[IG-Search](https://arxiv.org/abs/2604.15148)); PPO is rare. Two collapses: [LLDS](https://arxiv.org/abs/2512.04220) (Lazy Likelihood Displacement, +45.2% recovery), [ASTER](https://arxiv.org/abs/2602.01204) (interaction collapse, 4K cold-start). Credit assignment moved from heuristics to LLM-token level: [CW-GRPO](https://arxiv.org/abs/2604.14267) per-round, [SLATE](https://arxiv.org/abs/2602.23440) step-decomposed (7B +7.0%/3B +30.7%), [E-GRPO](https://arxiv.org/abs/2510.24694) entity partial reward.

> Seam: L3 trains a search agent as one long trajectory; GRPO's group-relative advantage breaks for multi-agent / role-heterogeneous rollouts — the entry of [02](./02-credit-assignment-rl.md).

### L4 · Agent Architecture (20+)

Five paradigms: hierarchical (MindSearch DAG, Mind DR 3-stage), pipeline ([ManuSearch](https://arxiv.org/abs/2505.18105), [STORM](https://arxiv.org/abs/2402.14207)/[Co-STORM](https://arxiv.org/abs/2408.15232)), parallel ([AggAgent](https://arxiv.org/abs/2604.11753) +5.3%/deep-research +10.3%, W&D BrowseComp 62.2%), decentralized ([AgentWebBench](https://arxiv.org/abs/2604.10938), [Mango](https://arxiv.org/abs/2604.18779) ACL26 +26.8%), self-evolving (CoEvolve, EvoMaster).

> Seam: empirical studies show no single paradigm dominates; operational stability vs deliberative depth is a fundamental trade-off — more agents deepen but destabilize.

### L5 · Long-Horizon Context (20+) → see [01](./01-long-horizon-rl.md)

Four challenges, four lines: Context Rot → [RE-TRAC](https://arxiv.org/abs/2602.02486) recursive compress (+15–20%), [Pensieve/StateLM](https://arxiv.org/abs/2602.12108) active curation (52% vs ~5%); plan anchoring → [WebAnchor](https://arxiv.org/abs/2601.03164) plan/exec split (BrowseComp 46.0%); tunnel vision → [SIGHT](https://arxiv.org/abs/2602.11551) info-gain branching; search scaling → [DeepDiver](https://arxiv.org/abs/2505.24332) intensity-scaling emergence, [Ares](https://arxiv.org/abs/2603.07915) −52.7% tokens.

> Seam: L5 decides whether L3 long-trajectory training can even deploy — unmanaged BrowseComp-Plus ~5% vs managed 52%, an order of magnitude.

### L6 · Synthesis / Retrieval (15+)

Report gen (STORM/Co-STORM, WebThinker Think-Search-Draft); retrieval ([AgentIR](https://arxiv.org/abs/2603.04384) reasoning-aware, [CoSearch](https://arxiv.org/abs/2604.17555) joint train closing 26.8% F1 retriever gap, [Rerank](https://arxiv.org/abs/2601.14224) trade-off); citation ([C-GRPO](https://arxiv.org/abs/2601.06021) rubric, Marco 3-layer); multimodal (MTA-Agent, [VISOR](https://arxiv.org/abs/2604.09508), [LMM-Searcher](https://arxiv.org/abs/2604.12890), MERRIN ~40%).

> Seam: reasoner and retriever are trained/deployed independently; retrieval quality is a hidden ceiling (⑨ retriever–reasoner split).

### L7 · Evaluation (15+ benchmarks)

GAIA (2023.11, 466, de-facto), [BrowseComp](https://arxiv.org/abs/2504.12516) (2025.04, 1266, the 2025–2026 headline), WebPuzzle, ORION, Dr. Bench, LiveResearchBench, [DeepSearchQA](https://arxiv.org/abs/2601.20975) (comprehensiveness + stopping), BrowseComp-V3 (multimodal), DeepResearch-9K, [AgentWebBench](https://arxiv.org/abs/2604.10938) (multi-agent coord), MERRIN (video/audio), MindDR Benchmark (Chinese).

> Seam: divergent axes, no unified long/credit/process eval; hallucination remains SOTA 25–35% (Trace-CUHK's G_E exposes "high-score hallucination").

---

## 3. Maturity snapshot

| Direction | Maturity | Rep. | Open problem |
|-----------|----------|------|--------------|
| RL-trained search agent | 🟢 mature | Search-R1, WebThinker | stability, credit assignment |
| Data / trajectory synthesis | 🟡 fast | OpenSeeker, OpenResearcher | long-horizon, diversity |
| Multi-agent architecture | 🟡 fast | MindSearch, Mind DR | coordination, architecture choice |
| Long-horizon reasoning | 🟠 early | RE-TRAC, Pensieve | context rot, plan anchoring |
| Multimodal search | 🔴 early | VISOR, LMM-Searcher | cross-modal reasoning, eval (~40%) |
| Evaluation | 🟡 fast | BrowseComp, GAIA, DeepSearchQA | unified standard, multimodal |

---

## 4. Key surveys (panorama entry points)

- [From Web Search towards Agentic Deep Research](https://arxiv.org/abs/2506.18959) (2025.06, UIUC/PKU etc., 24 authors) — full 5-D taxonomy, test-time scaling law. ★must-read
- [A Survey of LLM-based Deep Search Agents](https://arxiv.org/abs/2508.05668) (2025.08, SJTU) — first dedicated survey, 4-D taxonomy. ★must-read
- [RL Foundations for Deep Research Systems](https://arxiv.org/abs/2509.06733) (2025.09) — RL basics survey; long-horizon credit assignment as core.
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) (2025.01) — SFT vs RL.
- [Empirical Study on RL for Reasoning-Search Interleaved Agents](https://arxiv.org/abs/2505.15117) (2025.05) — reward/base/engine factors.

---

## 5. Stack seams (carries forward [04](./04-problems-and-innovations.md))

Most of the 12 structural problems diagnosed in [04](./04-problems-and-innovations.md) live at *seams*, not within layers: L2→L3 (data noise into training), L3→L4 (GRPO mismatch on multi-agent roles), L4→L5 (architecture amplifies context rot), L5→L6 (retriever split), L6→L7 (hallucination over-credited by eval). This is why [04](./04-problems-and-innovations.md)'s seven innovation directions are almost all *cross-layer*: hierarchical reward spans L2/L3, Markov state machine spans L3/L5, agent-retriever co-evolution spans L3/L6.

---

## 6. Numeric arXiv ids used here

R1 2501.12948 · Kimi k1.5 2501.12599 · Search-R1 2503.09516 · R1-Searcher 2503.05592 · ReSearch 2503.19470 · WebThinker 2504.21776 · DeepResearcher 2504.03160 · DeepDiver 2505.24332 · ReasonRAG 2505.14069 · LeTS 2505.17447 · beta-GRPO 2505.17281 · ManuSearch 2505.18105 · ZeroSearch 2505.04588 · BrowseComp 2504.12516 · E-GRPO 2510.24694 · LLDS 2512.04220 · C-GRPO 2601.06021 · WebAnchor 2601.03164 · ASTER 2602.01204 · SIGHT 2602.11551 · SLATE 2602.23440 · RE-TRAC 2602.02486 · Pensieve/StateLM 2602.12108 · AutoSearch 2604.17337 · Cycle-Consistent 2604.12967 · AgentIR 2603.04384 · CW-GRPO 2604.14267 · CoSearch 2604.17555 · AggAgent 2604.11753 · AgentWebBench 2604.10938 · Mango 2604.18779 · VISOR 2604.09508 · LMM-Searcher 2604.12890 · LiteResearcher 2604.17931 · WebFactory 2603.05044 · CoEvolve 2604.15840 · OffSeeker 2601.18467 · Ares 2603.07915 · DeepSearchQA 2601.20975 · Rerank 2601.14224 · STORM 2402.14207 · Co-STORM 2408.15232 · GAIA 2311.12983 · surveys 2506.18959 / 2508.05668 / 2509.06733 / 2501.17161. Unmatched (QPLEX NeurIPS21, MAVEN NeurIPS19) cited by venue.

---

<p align="center"><sub>Panorama draws on 100+ papers, 2023.11–2026.04.</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:F5C451,50:9B6BFF,100:22D3EE&height=90&section=footer&text=Read%20the%20stack%2C%20not%20the%20paper.&fontSize=13&fontColor=ffffff&fontAlignY=45&animation=blink"/>
