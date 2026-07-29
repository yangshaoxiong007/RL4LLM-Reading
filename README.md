<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,40:203A6C,75:5B8CFF,100:9B6BFF&height=260&section=header&text=RL4LLM%20Reading&fontSize=58&fontColor=ffffff&fontAlignY=40&fontAlign=50&desc=Long-Horizon%20%C2%B7%20Credit%20Assignment%20%C2%B7%20Process%20Reward%20for%20Search&descSize=15&descAlignY=64&descAlign=50&animation=fadeIn"/>

<p align="center">
  <b>🇨🇳 <a href="#中文版">中文</a></b> &nbsp;&nbsp; <b>🇬🇧 <a href="#english">English</a></b>
  <br/><br/>
  <img src="https://img.shields.io/badge/type-reading_notes-0E1117?style=flat-square"/>
  <img src="https://img.shields.io/badge/topic-RL_for_LLMs-FF4081?style=flat-square"/>
  <img src="https://img.shields.io/badge/focus-multi--agent%20%7C%20search%20%7C%20long--horizon-F5C451?style=flat-square"/>
  <img src="https://img.shields.io/badge/arXiv_id-✓_cross--checked-22D3EE?style=flat-square"/>
</p>

---

> RL for LLMs · DeepSearch / DeepResearch 的精读笔记。每篇自洽深读:**问题 → 形式化 → 方法谱系 → 论文剖析 → 批判 → 开放问题**。文中数字 arXiv id 均在 `arxiv.org/abs/<id>` 标题匹配核验;不可匹配者按会议/标题引用。

## 文献景观

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/concept-map.png" width="94%" alt="DeepSearch/DeepResearch 文献景观图：基座模型 → 推理 RL →（过程奖励 / 信用分配两条汇入接缝）→ 搜索智能体 → 评测；长程时间轴横贯其下"/></p>

<sub align="center">把 2023.11–2026.04 的工作摆进同一张景观:从基座推理模型、推理 RL 训练范式,经「过程奖励」与「信用分配」两条接缝,汇入搜索/DeepResearch 智能体,再到评测基准。多数瓶颈不在某一层内,而在层间接缝——稀疏末端奖励无法忠实分解到中间步与各智能体。长程时间轴横贯其下:奖励滞后数十步才到,正是过程奖励与信用分配要解决的问题。</sub>

本仓库沿这条景观把笔记组织成几条可互相印证的线索:

- **长程 RL** — 奖励在决定性动作之后数十步才到,是贯穿全图的时间轴。
- **信用分配** — 多个 agent / 步 / 工具共造一个结果时,归功于谁、哪一步。
- **搜索过程奖励** — 把末端稀疏结果转成密集、可检的步级信号,搜索是其最自然的领域。
- **问题诊断与生态全景** — 把上述线索落进结构性问题与创新方向,以及一张完整的全量索引。

## 中文版

| # | 主题 | 核心问题 | 锚定方法 | 文件 |
|---|------|---------|---------|------|
| 01 | **长程 RL** | 奖励远滞后如何训练 LLM? | GRPO, R1-RL, LLDS/ASTER 崩溃诊断 | [`01-long-horizon-rl.zh.md`](./01-long-horizon-rl.zh.md) |
| 02 | **信用分配** | 共享奖励团队里谁/哪步记功? | COMA, MAPPO/QMIX, CW-GRPO/SLATE | [`02-credit-assignment-rl.zh.md`](./02-credit-assignment-rl.zh.md) |
| 03 | **搜索过程奖励** | 怎么给搜索步密集忠实奖励? | PRM800K, Math-Shepherd, Search-R1 系列, ReasonRAG/LeTS | [`03-process-reward-search.zh.md`](./03-process-reward-search.zh.md) |
| 04 | **问题诊断与创新方向** | 12 条结构性问题怎么治根因? | LLDS/ASTER/CW-GRPO/RE-TRAC + 7 创新方向 | [`04-problems-and-innovations.zh.md`](./04-problems-and-innovations.zh.md) |
| 05 | **生态全景** | 七层技术栈里每篇处在哪一层? | 基座→数据→RL→架构→长程→综合→评测 | [`05-ecosystem-panorama.zh.md`](./05-ecosystem-panorama.zh.md) |

## English

| # | Topic | Core question | Anchor methods | File |
|---|-------|---------------|----------------|------|
| 01 | **Long-Horizon RL** | How to train LLMs when reward lands far downstream? | GRPO, R1-RL, LLDS/ASTER collapse | [`01-long-horizon-rl.md`](./01-long-horizon-rl.md) |
| 02 | **Credit Assignment** | Who/which step deserves credit in shared-reward teams? | COMA, MAPPO/QMIX, CW-GRPO/SLATE | [`02-credit-assignment-rl.md`](./02-credit-assignment-rl.md) |
| 03 | **Process Reward for Search** | How to give dense, faithful rewards to search steps? | PRM, Math-Shepherd, Search-R1 family, ReasonRAG/LeTS | [`03-process-reward-search.md`](./03-process-reward-search.md) |
| 04 | **Problem Diagnosis & Innovations** | How to treat the 12 failure modes at the root? | LLDS/ASTER/CW-GRPO/RE-TRAC + 7 directions | [`04-problems-and-innovations.md`](./04-problems-and-innovations.md) |
| 05 | **Ecosystem Panorama** | Which of the seven layers does each paper live in? | Foundation→Data→RL→Arch→LongHorizon→Synthesis→Eval | [`05-ecosystem-panorama.md`](./05-ecosystem-panorama.md) |

## 阅读约定

每篇同构,使各线索可读成一条论证:

1. **为何重要** — 动机失败模式。
2. **形式化** — 所需的 MDP/MADP/代理目标框架。
3. **方法谱系** — 景观,非孤立论文。
4. **论文剖析** — 每篇实际贡献(及其界限)。
5. **批判** — 领域何处过度声称或欠理论。
6. **开放问题** — 活跃问题,集中在长程多智能体搜索 + RL。

## 参考文献

文中涉及的数字 arXiv id 均可在 arXiv 上按标题核对:GRPO 2402.03300、R1 2501.12948、Kimi k1.5 2501.12599、LLDS 2512.04220、ASTER 2602.01204、Dr.GRPO 2503.20783、DAPO 2503.14476、ReFT 2401.08967、Let's Verify 2305.20050、Math-Shepherd 2312.08935、Search-R1 2503.09516、R1-Searcher 2503.05592、ReSearch 2503.19470、WebThinker 2504.21776、DeepResearcher 2504.03160、BrowseComp 2504.12516、ZeroSearch 2505.04588、DeepDiver 2505.24332、ReasonRAG 2505.14069、LeTS 2505.17447、E-GRPO 2510.24694、C-GRPO 2601.06021、SLATE 2602.23440、RE-TRAC 2602.02486、PRMBench 2501.03124、COMA 1705.08926、QMIX 1803.11485、VDN 1706.05296、MAPPO 2103.01955、MADDPG 1706.02275、AgentWebBench 2604.10938。QPLEX(NeurIPS 2021)、MAVEN(NeurIPS 2019)按会议引用。

---

<p align="center"><sub>笔记随阅读持续更新。</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:9B6BFF,50:5B8CFF,100:0F2027&height=100&section=footer&text=Toward%20faithful%20credit%20attribution.&fontSize=14&fontColor=ffffff&fontAlignY=45&animation=blink"/>
