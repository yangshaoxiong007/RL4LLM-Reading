<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:22D3EE,30:5B8CFF,70:9B6BFF,100:F5C451&height=200&section=header&text=%E7%94%9F%E6%80%81%E5%85%A8%E6%99%AF&fontSize=42&fontColor=ffffff&fontAlignY=42&fontAlign=50&desc=DeepSearch%20%2F%20DeepResearch%20%E6%8A%80%E6%9C%AF%E6%A0%88%20%C2%B7%20%E6%96%87%E7%8C%AE%E5%85%A8%E7%B4%A2%E5%BC%95&descSize=14&descAlignY=66&animation=fadeIn"/>

> **第 05 篇 · 生态全景。** 把 DeepSearch / DeepResearch 当作一个七层技术栈来读：基座模型 → 数据/轨迹 → RL 训练 → 智能体架构 → 长程上下文 → 综合与检索 → 评测。每层给出代表性论文与量化结果、层间接缝问题，并在末尾给出按方向归类的全量文献索引。这是把前四篇放进整张地图的"鸟瞰图"。

<p align="center">
  <b>🇨🇳 <a href="./05-ecosystem-panorama.zh.md">中文</a></b> &nbsp; <b>🇬🇧 <a href="./05-ecosystem-panorama.md">English</a></b>
  &nbsp; · &nbsp; <a href="./README.md">← 返回总纲</a>
</p>

---

## 一、为什么需要一篇"全景"

前四篇分别精读长程 RL、信用分配、搜索过程奖励、问题诊断与创新方向——但它们都处在同一张更大的技术栈里。一篇全景图能回答三类问题：(1) 某篇工作处在哪一层、和谁竞争;(2) 层与层之间的接缝在哪(多数 SOTA 瓶颈不是层内而是接缝);(3) 评测层为何始终落后于方法层。本篇按七层组织 2023.11–2026.04 的工作,并在第七节给出全量索引。

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/ecosystem-panorama.png" width="96%" alt="DeepSearch/DeepResearch 七层技术栈全景图：基座→数据→RL→架构→长程→综合→评测"/></p>

<sub align="center">七层从下到上：基座模型 → 数据/轨迹合成 → RL 训练范式 → 智能体架构 → 长程上下文管理 → 综合与检索增强 → 评测基准。右栏标注层间接缝的代表性问题。</sub>

---

## 二、七层技术栈

### L1 · 基座模型层 (Foundation)

| 模型 | 定位 | 关键性质 |
|------|------|----------|
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (Nature) | 纯 RLVR 涌现长 CoT 的基石 | GRPO + 可验证奖励,无 SFT 推理数据 |
| [Kimi k1.5](https://arxiv.org/abs/2501.12599) | 长上下文 RL 推理 | AIME 77.5 / MATH 500 96.2 |
| o1/o3, Qwen3, GPT-5, Claude, Gemini | 通用基座 | — |
| 4B 级轻量 ([LiteResearcher](https://arxiv.org/abs/2604.17931)、DR-Venus、ORBIT) | 开源民主化 | 4B 已可在 GAIA 匹敌商业系统 |

### L2 · 数据与轨迹合成层 (Data / Trajectory) — 35+ 篇

五大范式共存:① 纯 RL(无轨迹) 如 [R1-Searcher](https://arxiv.org/abs/2503.05592)/[Search-R1](https://arxiv.org/abs/2503.09516);② 教师蒸馏 SFT,如 [OpenSeeker](https://arxiv.org/abs/2603.15594) 11.7K 去噪轨迹样本达到前沿;③ 混合 SFT+RL,如 Mind DeepResearch 四阶段;④ 离线偏好 [OffSeeker](https://arxiv.org/abs/2601.18467)、[WebThinker](https://arxiv.org/abs/2504.21776);⑤ 自进化/环境合成 [CoEvolve](https://arxiv.org/abs/2604.15840)、[WebFactory](https://arxiv.org/abs/2603.05044)、[LiteResearcher](https://arxiv.org/abs/2604.17931)、[ZeroSearch](https://arxiv.org/abs/2505.04588)。

> 接缝问题:L2 直接决定 L3 的训练上限。DeepDive-32B 仅 4.1K 样本→BrowseComp 15.3%;MiroThinker 147K 样本→仅 10.6%;OpenSeeker 11.7K→29.5%。质量 >> 数量,且"去噪轨迹"是分水岭。

### L3 · RL 训练范式层 (Training) — 40+ 篇 → 详见 [01](./01-long-horizon-rl.zh.md)、[02](./02-credit-assignment-rl.zh.md)、[03](./03-process-reward-search.zh.md)

GRPO 及其变体是事实标准(CW/E/C/beta/Anchor/Step-level/[IG-Search](https://arxiv.org/abs/2604.15148) 等),PPO 极少使用。两大训练崩溃:[LLDS](https://arxiv.org/abs/2512.04220)(Lazy Likelihood Displacement,+45.2% 挽回)、[ASTER](https://arxiv.org/abs/2602.01204)(interaction collapse,4K 冷启动)。信用分配已从启发式走向 LLM token 级:[CW-GRPO](https://arxiv.org/abs/2604.14267) per-round 贡献、[SLATE](https://arxiv.org/abs/2602.23440) 步级分解(7B +7.0%/3B +30.7%)、[E-GRPO](https://arxiv.org/abs/2510.24694) 实体部分奖励。

> 接缝问题:L3 把搜索 agent 当作一条长轨迹训练,GRPO 组相对优势在多智能体/角色异构 rollout 上不成立——这正是 [02](./02-credit-assignment-rl.zh.md) 的入口。

### L4 · 智能体架构层 (Architecture) — 20+ 篇

五种范式:层级式(MindSearch DAG、Mind DR 三阶段)、管道式([ManuSearch](https://arxiv.org/abs/2505.18105)、[STORM](https://arxiv.org/abs/2402.14207)/[Co-STORM](https://arxiv.org/abs/2408.15232))、并行式([AggAgent](https://arxiv.org/abs/2604.11753) +5.3%/深度研究+10.3%、W&D BrowseComp 62.2%)、去中心化([AgentWebBench](https://arxiv.org/abs/2604.10938)、[Mango](https://arxiv.org/abs/2604.18779) ACL26 +26.8%)、自进化(CoEvolve、EvoMaster)。

> 接缝问题:实证研究表明无单一范式主导;操作稳定性 vs 深思能力是根本权衡——更多 agent 提升深度却降低稳定性。

### L5 · 长程上下文管理层 (Long-Horizon Context) — 20+ 篇 → 详见 [01](./01-long-horizon-rl.zh.md)

四大挑战对应四条解决线:Context Rot → [RE-TRAC](https://arxiv.org/abs/2602.02486) 递归压缩(+15–20%)、[Pensieve/StateLM](https://arxiv.org/abs/2602.12108) 主动策展(52% vs ~5%);规划锚定 → [WebAnchor](https://arxiv.org/abs/2601.03164) 规划-执行分离(BrowseComp 46.0%);隧道视觉 → [SIGHT](https://arxiv.org/abs/2602.11551) 信息增益分支;搜索缩放 → [DeepDiver](https://arxiv.org/abs/2505.24332) 强度缩放涌现、[Ares](https://arxiv.org/abs/2603.07915) −52.7% token。

> 接缝问题:L5 决定 L3 长轨迹训练是否发得出去——无状态管理时 BrowseComp-Plus 仅 ~5%,有管理 52%,差一个数量级。

### L6 · 综合与检索增强层 (Synthesis / Retrieval) — 15+ 篇

报告生成(STORM/Co-STORM、WebThinker Think-Search-Draft);检索增强([AgentIR](https://arxiv.org/abs/2603.04384) 推理感知、[CoSearch](https://arxiv.org/abs/2604.17555) 联合训练抹平 26.8% F1 检索器差距、[Rerank](https://arxiv.org/abs/2601.14224) 性价比分析);引用验证([C-GRPO](https://arxiv.org/abs/2601.06021) 引用感知 rubric、Marco 三层验证);多模态综合(MTA-Agent、[VISOR](https://arxiv.org/abs/2604.09508)、[LMM-Searcher](https://arxiv.org/abs/2604.12890)、MERRIN ~40%)。

> 接缝问题:推理 agent 与检索器独立训练/部署,检索质量成为隐性上限(⑨ 检索器-推理器割裂)。

### L7 · 评测层 (Evaluation) — 15+ 个基准

GAIA(2023.11,466 题,事实标准)、[BrowseComp](https://arxiv.org/abs/2504.12516)(2025.04,1266 题,2025-2026 配置指标)、WebPuzzle、ORION、Dr. Bench、LiveResearchBench、[DeepSearchQA](https://arxiv.org/abs/2601.20975)(搜索全面性+停止准则)、BrowseComp-V3(多模态)、DeepResearch-9K、[AgentWebBench](https://arxiv.org/abs/2604.10938)(多智能体协调)、MERRIN(视频/音频)、MindDR Bench(中文)。

> 接缝问题:评测维度各异,缺统一长程-信用-过程联合标准;瀚菌率仍是 SOTA 25-35%(Trace-CUHK G_E 指标揭示"高分瀚菌")。

---

## 三、技术成熟度速览

| 方向 | 成熟度 | 代表 | 开放问题 |
|------|--------|------|----------|
| RL 训练搜索 agent | 🟢 较成熟 | Search-R1、WebThinker | 训练稳定性、信用分配 |
| 数据/轨迹合成 | 🟡 快速发展 | OpenSeeker、OpenResearcher | 长程轨迹、多样性 |
| 多 Agent 架构 | 🟡 快速发展 | MindSearch、Mind DR | 协调效率、架构选择 |
| 长程推理 | 🟠 初步探索 | RE-TRAC、Pensieve | Context rot、规划锚定 |
| 多模态搜索 | 🔴 早期 | VISOR、LMM-Searcher | 跨模态推理、评测(仅 ~40%) |
| 评测基准 | 🟡 快速发展 | BrowseComp、GAIA、DeepSearchQA | 统一标准、多模态 |

---

## 四、关键综述(读全景的入口)

- [From Web Search towards Agentic Deep Research](https://arxiv.org/abs/2506.18959) (2025.06, UIUC/北大等 24 位作者) — 五维完整分类法,test-time scaling law 形式化。⭐必读
- [A Survey of LLM-based Deep Search Agents](https://arxiv.org/abs/2508.05668) (2025.08, 上交) — 首个专门综述,架构/优化/应用/评测四维。⭐必读
- [RL Foundations for Deep Research Systems](https://arxiv.org/abs/2509.06733) (2025.09) — RL 基础综述,长程信用分配为核心挑战。
- [SFT Memorizes, RL Generalizes](https://arxiv.org/abs/2501.17161) (2025.01) — SFT vs RL 比较研究。
- [Empirical Study on RL for Reasoning-Search Interleaved Agents](https://arxiv.org/abs/2505.15117) (2025.05) — 奖励结构/基座/搜索引擎因素实测。

---

## 五、全栈接缝问题一览(承接 [04](./04-problems-and-innovations.zh.md))

前四篇已诊断的 12 条结构性问题,大多发生在**接缝**而非层内:L2→L3(数据噪声传到训练)、L3→L4(GRPO 在多智能体角色上失配)、L4→L5(架构选择放大 context rot)、L5→L6(检索割裂)、L6→L7(幻觉在评测上被高估)。这解释了为何 [04](./04-problems-and-innovations.zh.md) 的 7 个创新方向几乎都落在"跨层"上——层次化奖励跨 L2/L3、马尔可夫状态机跨 L3/L5、Agent-Retriever 协同跨 L3/L6。

---

## 六、本篇用到的所有数字 arXiv id

R1 2501.12948 · Kimi k1.5 2501.12599 · Search-R1 2503.09516 · R1-Searcher 2503.05592 · ReSearch 2503.19470 · WebThinker 2504.21776 · DeepResearcher 2504.03160 · DeepDiver 2505.24332 · ReasonRAG 2505.14069 · LeTS 2505.17447 · beta-GRPO 2505.17281 · ManuSearch 2505.18105 · ZeroSearch 2505.04588 · agent IR 2505.15117(alias),BrowseComp 2504.12516 · E-GRPO 2510.24694 · LLDS 2512.04220 · C-GRPO 2601.06021 · WebAnchor 2601.03164 · ASTER 2602.01204 · SIGHT 2602.11551 · SLATE 2602.23440 · RE-TRAC 2602.02486 · Pensieve/StateLM 2602.12108 · AutoSearch 2604.17337 · Cycle-Consistent 2604.12967 · AgentIR 2603.04384 · CW-GRPO 2604.14267 · CoSearch 2604.17555 · AggAgent 2604.11753 · AgentWebBench 2604.10938 · Mango 2604.18779 · VISOR 2604.09508 · LMM-Searcher 2604.12890 · LiteResearcher 2604.17931 · WebFactory 2603.05044 · CoEvolve 2604.15840 · OffSeeker 2601.18467 · Ares 2603.07915 · DeepSearchQA 2601.20975 · Rerank 2601.14224 · STORM 2402.14207 · Co-STORM 2408.15232 · GAIA 2311.12983 · 综述 2506.18959 / 2508.05668 / 2509.06733 / 2501.17161。不可匹配者(QPLEX NeurIPS21、MAVEN NeurIPS19)按会议引用。

---

<p align="center"><sub>全景基于 100+ 篇 2023.11–2026.04 文献。</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:F5C451,50:9B6BFF,100:22D3EE&height=90&section=footer&text=Read%20the%20stack%2C%20not%20the%20paper.&fontSize=13&fontColor=ffffff&fontAlignY=45&animation=blink"/>
