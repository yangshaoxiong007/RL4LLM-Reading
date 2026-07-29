<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,40:203A6C,75:5B8CFF,100:9B6BFF&height=260&section=header&text=RL4LLM%20Reading&fontSize=58&fontColor=ffffff&fontAlignY=40&fontAlign=50&desc=Long-Horizon%20%C2%B7%20Credit%20Assignment%20%C2%B7%20Process%20Reward%20for%20Search&descSize=15&descAlignY=64&descAlign=50&animation=fadeIn"/>

<p align="center">
  <b>🇨🇳 <a href="#中文版">中文</a></b> &nbsp;&nbsp; <b>🇬🇧 <a href="#english">English</a></b>
  <br/><br/>
  <img src="https://img.shields.io/badge/level-expert_reading_notes-0E1117?style=flat-square"/>
  <img src="https://img.shields.io/badge/topic-RL_for_LLMs-FF4081?style=flat-square"/>
  <img src="https://img.shields.io/badge/focus-multi--agent%20%7C%20search%20%7C%20long--horizon-F5C451?style=flat-square"/>
  <img src="https://img.shields.io/badge/id_verification-✓_cross--checked-22D3EE?style=flat-square"/>
</p>

---

> 三条交错前沿的**专家级**精读笔记。每篇自洽深读:**问题 → 形式化 → 方法谱系 → 论文剖析 → 批判 → 开放问题**。所有数字 arXiv id 均在 `arxiv.org/abs/<id>` 标题匹配核验过;不可匹配者按会议/标题引用。

## 为什么这三条一起

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/concept-map.png" width="92%" alt="三线索概念图：信用归因的时间轴/结构轴/机制尺度，论文流入，交汇于作者研究线"/></p>

<sub align="center">三条交错前沿 = 同一信用归因问题在时间轴(长程)、结构轴(多智能体)、机制(搜索过程奖励)三尺度的投影，关键论文沿各轴流入，交汇于多智能体搜索 + RL。</sub>


这三条线是**同一问题在三个信用归因尺度上的投影**,当 agent 在长程上搜索时,它们塌缩为同一研究议程:

```
                    long horizon ────────────────────────▶
   ┌──────────────┐        ┌──────────────────┐        ┌─────────────────────────────────┐
   │ 长程 RL      │  ═══▶  │ 信用分配          │  ═══▶  │ 搜索过程奖励                     │
   │(时间轴)      │        │(多智能体/多步)    │        │(密集步级监督)                    │
   │              │        │ LLM 角色 ∩ 团队奖 │        │ state=上下文, action=查询,       │
   │              │        │                  │        │ reward=步进展                    │
   └──────────────┘        └──────────────────┘        └─────────────────────────────────┘
```

- **长程 RL** 是*时间*轴:奖励在决定性动作之后数十步才到。
- **信用分配** 是*结构*轴:多个 agent / 步 / 工具共造一个结果时,归功于谁——而 LLM agent 的"角色"多是共享整轨迹奖励的角色特化 LLM(异构分组 RL 设定)。
- **搜索过程奖励** 是同时让前两者可解的*机制*:把末端稀疏结果转成密集可检步级信号,而搜索是它最自然的领域。

## 中文版

| # | 主题 | 核心问题 | 锚定方法 | 文件 |
|---|------|---------|---------|------|
| 01 | **长程 RL** | 奖励远滞后如何训练 LLM? | GRPO, R1-RL, LLDS/ASTER 崩溃诊断 | [`01-long-horizon-rl.zh.md`](./01-long-horizon-rl.zh.md) |
| 02 | **信用分配** | 共享奖励团队里谁/哪步记功? | 异构分组, COMA, MAPPO/QMIX, CW-GRPO/SLATE | [`02-credit-assignment-rl.zh.md`](./02-credit-assignment-rl.zh.md) |
| 03 | **搜索过程奖励** | 怎么给搜索步密集忠实奖励? | PRM800K, Math-Shepherd, Search-R1 系列, ReasonRAG/LeTS | [`03-process-reward-search.zh.md`](./03-process-reward-search.zh.md) |

## English

| # | Topic | Core question | Anchor methods | File |
|---|-------|---------------|----------------|------|
| 01 | **Long-Horizon RL** | How to train LLMs when reward lands far downstream? | GRPO, R1-RL, LLDS/ASTER collapse | [`01-long-horizon-rl.md`](./01-long-horizon-rl.md) |
| 02 | **Credit Assignment** | Who/which step deserves credit in shared-reward teams? | Heterogeneous-group, COMA, CW-GRPO/SLATE | [`02-credit-assignment-rl.md`](./02-credit-assignment-rl.md) |
| 03 | **Process Reward for Search** | How to give dense, faithful rewards to search steps? | PRM, Math-Shepherd, Search-R1 family, ReasonRAG/LeTS | [`03-process-reward-search.md`](./03-process-reward-search.md) |

## 阅读约定

每篇同构,使三条可读成一个论证:

1. **为何重要** — 动机失败模式。
2. **形式化** — 专家需要的 MDP/MADP/代理目标框架。
3. **方法谱系** — 景观,非孤立论文。
4. **论文剖析** — 每篇实际贡献(及其界限)。
5. **批判** — 领域何处过度声称或欠理论。
6. **开放问题** — 活跃问题,多个直指*多智能体搜索 + RL*(作者研究线)。

## id 核验方法学

本笔记里每个数字 arXiv id 都从 `arxiv.org/abs/<id>` 抓取并标题匹配。本会话核验确认(✓):GRPO 2402.03300、R1 2501.12948、LLDS 2512.04220、ASTER 2602.01204、Dr.GRPO 2503.20783、DAPO 2503.14476、ReFT 2401.08967、Let's Verify 2305.20050、Math-Shepherd 2312.08935、Search-R1 2503.09516、R1-Searcher 2503.05592、ReSearch 2503.19470、ReasonRAG 2505.14069、LeTS 2505.17447、E-GRPO 2510.24694、StepSearch 2505.15107、IGPO 2510.14967、IG-Search 2604.15148、CW-GRPO 2604.14267、SLATE 2602.23440、C-GRPO 2601.06021、Search-R2 2602.03647、PRMBench 2501.03124、COMA 1705.08926、QMIX 1803.11485、VDN 1706.05296、MAPPO 2103.01955、MADDPG 1706.02275。QPLEX/MAVEN 的 arXiv id 未匹配成功,按会议引用,不臆造数字。

## 作者背景

本笔记与**LLM 驱动多智能体搜索系统端到端 RL 优化**研究并行维护——三条线索在此交汇。相关发表见作者主页。

---

<p align="center"><sub>§ 切勿引用未标记 ✓ 的数字 id;不可匹配者见上「按会议引用」。</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:9B6BFF,50:5B8CFF,100:0F2027&height=100&section=footer&text=Toward%20faithful%20credit%20attribution.&fontSize=14&fontColor=ffffff&fontAlignY=45&animation=blink"/>
