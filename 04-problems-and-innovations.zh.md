<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:5B8CFF,40:9B6BFF,75:22D3EE,100:F5C451&height=200&section=header&text=%E9%97%AE%E9%A2%98%E8%AF%8A%E6%96%AD%20%C2%B7%20%E5%88%9B%E6%96%B0%E6%96%B9%E5%90%91&fontSize=40&fontColor=ffffff&fontAlignY=42&fontAlign=50&desc=12%20%E5%A4%B1%E8%B4%A5%E6%A8%A1%E5%BC%8F%20%C2%B7%207%20%E5%88%9B%E6%96%B0%E6%96%B9%E5%90%91&descSize=14&descAlignY=66&animation=fadeIn"/>

> **第 04 篇 · 问题诊断与创新方向总纲。** 把前三篇（长程 RL / 信用分配 / 搜索过程奖励）暴露的失败模式收敛成 **12 条结构性问题**，并给出每条已落到方法组合迁移方案的 **7 个创新方向**。

<p align="center">
  <b>🇨🇳 <a href="./04-problems-and-innovations.zh.md">中文</a></b> &nbsp; <b>🇬🇧 <a href="./04-problems-and-innovations.md">English</a></b>
  &nbsp; · &nbsp; <a href="./README.md">← 返回总纲</a>
</p>

---

## 一、为什么需要这一篇

前三篇各走一轴：长程 RL 在**时间**轴上把奖励推到数十步之后；信用分配在**结构**轴上追问"团队/多步里谁记功"；搜索过程奖励是让前两者可解的**机制**——把末端稀疏结果榨成密集可检的步级信号。但当真把三者叠到一个多智能体搜索系统里，**真正的瓶颈不是单点方法，而是方法间的接缝**。这篇把接缝处的失败模式全数列出，并为每条给出一个可落地的创新方向，使三条轴塌缩成同一个研究议程——**多智能体搜索 + RL**。

每条问题都标注：**严重度**、**所在层面**、**代表论文（已核验）**、**量化表现**、**目前最佳方案及其不足**。

---

## 二、十二条结构性问题（诊断总表）

| # | 问题 | 层面 | 严重度 | 代表论文 | 量化表现 | 现有最佳方案 |
|---|------|------|--------|----------|----------|--------------|
| 1 | GRPO 训练崩溃 | 训练 | 🔴 高 | [LLDS](https://arxiv.org/abs/2512.04220) | 根因 Lazy Likelihood Displacement；三阶段 停滞→衰退→崩溃 | LLDS 正则化 **+45.2%**，但仅治标 |
| 2 | 交互崩溃（退化为纯内部推理、不用工具） | 训练 | 🔴 高 | [ASTER](https://arxiv.org/abs/2602.01204) | 工具反馈稀疏延迟，内部推理奖励更密集 | 4K 交互密集冷启动，缺通用性 |
| 3 | 信用分配困难 | 训练 | 🟠 中高 | [CW-GRPO](https://arxiv.org/abs/2604.14267)· [SLATE](https://arxiv.org/abs/2602.23440)· [E-GRPO](https://arxiv.org/abs/2510.24694) | 多轮搜索因果依赖复杂，outcome 仅给组级奖励 | 贡献加权 / 步级采样 / 实体部分奖励，均依赖启发式或外部 judge |
| 4 | 规划锚定（固守初始计划） | 策略 | 🟠 中高 | [WebAnchor](https://arxiv.org/abs/2601.03164) | 规划与执行同模型耦合，早期成功规划被过度强化 | 规划-执行两阶段 RL；BrowseComp 46.0% / GAIA 76.4% |
| 5 | 隧道视觉（噪声初始检索致方向偏移） | 策略 | 🟠 中高 | [SIGHT](https://arxiv.org/abs/2602.11551) | 缺探索-利用平衡，无"退出重启"机制 | 信息增益分支 + 自证据支持 |
| 6 | 过度搜索 | 策略 | 🟡 中 | [beta-GRPO](https://arxiv.org/abs/2505.17281)（Search Wisely） | **27.7%** 的搜索可避免 | 置信度门控；3B 超基线 +4% EM |
| 7 | Context Rot（上下文退化） | 长程 | 🔴 高 | [RE-TRAC](https://arxiv.org/abs/2602.02486)· [Pensieve/StateLM](https://arxiv.org/abs/2602.12108)· [ARC](https://arxiv.org/abs/2601.12030) | 无状态管理时 BrowseComp-Plus 仅 **~5%**；有状态管理 **52%** | 递归压缩 / 主动策展 / 反射驱动，仍有损失 |
| 8 | 搜索停止时机 | 长程 | 🟠 中高 | [DeepSearchQA](https://arxiv.org/abs/2601.20975) | 过早停止 vs recall 膨胀并存 | 尚无成熟停止准则推理方案 |
| 9 | 检索器-推理器割裂 | 检索 | 🟠 中高 | [CoSearch](https://arxiv.org/abs/2604.17555) | 固定检索与 oracle 差距 **26.8% F1** | 联合训练 |
| 10 | 多模态搜索能力不足 | 综合 | 🔴 高 | [VISOR](https://arxiv.org/abs/2604.09508)· [LMM-Searcher](https://arxiv.org/abs/2604.12890) | 最强系统在多模态基准仅 **~40%** | 统一动作空间起步，跨模态奖励未成熟 |
| 11 | 幻觉与引用不准 | 综合 | 🟠 中高 | [C-GRPO](https://arxiv.org/abs/2601.06021) | citation stuffing / brute-force / 证据-声称脱节 | 引用奖励 / 三层验证 |
| 12 | 评测碎片化 | 评测 | 🟡 中 | 15+ 基准（GAIA/BrowseComp/DeepSearchQA…） | 维度各异，缺统一长程-信用-过程联合评测 | 尚无统一标准 |

> **根因串陈**：①③ 是把搜索 agent 当长轨迹训练时 GRPO 优势估计的不稳定；⑦是固定上下文窗口的有界有损信道；⑨是训练时把推理器和检索器解耦。这三处是后续 7 个方向的真正发力点。

---

## 三、问题 → 创新方向 映射图

<p align="center"><img src="https://raw.githubusercontent.com/yangshaoxiong007/RL4LLM-Reading/main/images/problem-innovation-map.png" width="96%" alt="问题→创新方向映射图：12 条结构性问题按层面分组，流入 7 个创新方向"/></p>

<sub align="center">12 条问题按 训练／策略／长程／检索·综合 四组，沿"治根因"而非"治症状"的边流入 7 个创新方向；每条边的方向 = 该创新所解除的根因。</sub>

```
训练层(①③) ──▶ Ⅰ.自适应层次化奖励
训练层(②)   ───▶ Ⅶ.自进化搜索世界模型
策略层(④⑤) ──▶ Ⅳ.反事实搜索规划
策略层(⑥)   ───▶ Ⅴ.搜索预算帕累托
长程层(⑦⑧) ──▶ Ⅱ.马尔可夫搜索状态机
检索层(⑨)   ───▶ Ⅲ.Agent-Retriever 协同进化
综合层(⑩⑫) ──▶ Ⅵ.多模态统一搜索 RL
综合层(⑪)   ───▶ Ⅰ.自适应层次化奖励（细粒度信号）
```

---

## 四、七大创新方向（方法组合迁移详解）

> 每个方向 = **解除哪些根因** + **融合哪些已核验方法** + **迁移自哪个成熟学科** + **具体方案** + **预期突破（量化）**。

### 方向 Ⅰ · 自适应层次化奖励架构

- **解除根因**: ①GRPO 长轨迹崩溃、③信用分配、⑪引用不准
- **融合方法**: [CW-GRPO](https://arxiv.org/abs/2604.14267)（per-round 贡献评分，ACL 2026）· [SLATE](https://arxiv.org/abs/2602.23440)（步级截断采样 + 分解过程奖励，7B +7.0% / 3B +30.7%）· [LeTS](https://arxiv.org/abs/2505.17447)（无标注过程奖励估计）· [IG-Search](https://arxiv.org/abs/2604.15148)（信息增益）
- **迁移自**: 控制论——**层次化反馈控制**
- **方案**: 三层奖励——**微观 token 级**（边际信息增益）→ **中观步级**（每个 search-reason cycle 的独立贡献）→ **宏观轨迹级**（结果 + 效率惩罚）；按训练阶段动态调三层权重，早期侧重微观、后期侧重宏观。
- **预期突破**: 在不增加过程标注成本的前提下稳定 GRPO 长轨迹训练，复刻 ReasonRAG **18× 数据效率**级别的收益（5K 样本≈90K 样本，[ReasonRAG](https://arxiv.org/abs/2505.14069)）。

### 方向 Ⅱ · 马尔可夫搜索状态机 + 递归上下文压缩

- **解除根因**: ⑦Context Rot、⑧停止时机
- **融合方法**: [AoT](https://arxiv.org/abs/2502.12018)（马尔可夫推理，NeurIPS 2025）· [RE-TRAC](https://arxiv.org/abs/2602.02486)（四维状态摘要：发现/空缺/障碍/方向，+15–20%）· [Pensieve/StateLM](https://arxiv.org/abs/2602.12108)（主动上下文策展，52% vs ~5%）
- **迁移自**: 操作系统——**虚拟内存分页（L1/L2/L3）**
- **方案**: 搜索过程建模为 MDP，每步只依赖当前"搜索状态"（RE-TRAC 四维摘要）而非完整历史；三级缓存 L1 即时推理 / L2 子任务摘要 / L3 已验证事实库；RL 学习 L1↔L2↔L3 迁移策略。
- **预期突破**: 理论上支持无限步搜索；每步状态表示固定大小，把 BrowseComp-Plus 从 ~5% 拉到 50%+。

### 方向 Ⅲ · Agent-Retriever 协同进化训练

- **解除根因**: ⑨检索器-推理器割裂
- **融合方法**: [CoSearch](https://arxiv.org/abs/2604.17555)（联合训练推理+排序）· [AgentIR](https://arxiv.org/abs/2603.04384)（推理感知检索）· [Cycle-Consistent Search](https://arxiv.org/abs/2604.12967)（循环一致性代理奖励）
- **迁移自**: GAN 的**对抗训练 / 协同进化**
- **方案**: 双模块（推理 agent 生成查询并消费结果 + 检索器理解推理上下文）；联合 GRPO——agent 奖励=答案正确性，retrieever 奖励=结果被使用率（隐式反馈），协同奖励=循环一致性；对抗性难度升级让检索器逐步塞入更难的干扰文档。
- **预期突破**: 抹平 CoSearch 测出的 **26.8% F1** 检索器性能差距。

### 方向 Ⅳ · 反事实搜索规划与自适应策略切换

- **解除根因**: ④规划锚定、⑤隧道视觉
- **融合方法**: [WebAnchor](https://arxiv.org/abs/2601.03164)（规划-执行分离）· [SIGHT](https://arxiv.org/abs/2602.11551)（信息增益分支）· MCTS 探索-利用
- **迁移自**: 因果推理——**反事实推演**
- **方案**: 反事实规划评估器对当前规划生成 2–3 个替代规划，轻量模拟估期望信息增益；当连续 N 步信息增益低于阈值自动触发；在 SIGHT 识别的高增益节点展开 2–3 层轻量 MCTS；RL 训练"深度优先/广度优先/验证/回退"四策略的切换元策略。
- **预期突破**: 复杂多跳问题上显著减少规划锚定与隧道视觉，逼近 WebAnchor 的 BrowseComp 46% 基线并向上突破。

### 方向 Ⅴ · 搜索预算感知的帕累托最优训练

- **解除根因**: ⑥过度搜索
- **融合方法**: [AutoSearch](https://arxiv.org/abs/2604.17337)（自适应搜索深度）· [beta-GRPO](https://arxiv.org/abs/2505.17281)（置信度门控）· [Ares](https://arxiv.org/abs/2603.07915)（推理努力选择，−52.7% token）
- **迁移自**: 多目标优化——**帕累托前沿**
- **方案**: 双目标 RL（答案质量 vs 搜索成本）；训练一族模型覆盖帕累托前沿而非单点；推理时按用户预算（快速/标准/深度）路由策略；渐进式细化先给粗答再加深。
- **预期突破**: 同质量水平下搜索成本**降 50%+**，对冲 beta-GRPO 测出的 27.7% 可避免搜索。

### 方向 Ⅵ · 多模态统一搜索 RL 框架

- **解除根因**: ⑩多模态不足、⑫评测碎片化（多模态维度）
- **融合方法**: [VISOR](https://arxiv.org/abs/2604.09508)（视觉检索 RL）· [LMM-Searcher](https://arxiv.org/abs/2604.12890)（外部存储 offload，百轮搜索）· [MM-DeepResearch](https://arxiv.org/abs/2603.01050)（超图 QA）· [ProMMSearchAgent](https://arxiv.org/abs/2604.20486)（过程导向多模态奖励）
- **迁移自**: 多任务学习——**模态自适应路由**
- **方案**: 统一动作空间（text_search/image_search/video_search/code_execute/file_read）；模态感知奖励（文本=覆盖度 / 图像=相关 / 视频=时间定位 / 混合=跨模态一致性）；轻量分类器路由最优模态；多模态上下文压缩。
- **预期突破**: 多模态基准从 ~40% 提升到 60%+，并带动多模态评测维度的收敛。

### 方向 Ⅶ · 自进化搜索世界模型

- **解除根因**: ②交互崩溃（环境成本）、⑦训练环境不稳定
- **融合方法**: [LiteResearcher](https://arxiv.org/abs/2604.17931)（虚拟世界可扩展训练，4B GAIA 71.3%）· [ZeroSearch](https://arxiv.org/abs/2505.04588)（LLM 模拟检索）· [WebFactory](https://arxiv.org/abs/2603.05044)（环境合成）
- **迁移自**: Model-based RL——**世界模型（Dreamer 系列）**
- **方案**: 训练一个"给定查询预测搜索引擎返回"的世界模型；双循环——内循环低成本 RL、外循环真实 Web 验证并微调；对抗性世界模型模拟极端场景（引擎故障/信息污染）；多领域特化世界模型协作。
- **预期突破**: RL 训练成本**降 90%+** 同时保对真实 Web 的泛化，直接缓解交互崩溃的环境根因。

---

## 五、创新优先级（影响 × 可行性）

| 排名 | 方向 | 预期影响 | 实施难度 | 短期可行性 | 建议切入点 |
|------|------|----------|----------|------------|-----------|
| 1 | Ⅱ 马尔可夫状态机 | ⭐⭐⭐⭐⭐ | 中 | 高 | **优先投入** |
| 2 | Ⅰ 层次化奖励 | ⭐⭐⭐⭐⭐ | 中 | 高 | **优先投入** |
| 3 | Ⅴ 搜索预算帕累托 | ⭐⭐⭐⭐ | 中 | 高 | 工程化快见效 |
| 4 | Ⅲ Agent-Retriever 协同 | ⭐⭐⭐⭐ | 高 | 中 | 中期 |
| 5 | Ⅳ 反事实规划 | ⭐⭐⭐⭐ | 高 | 中 | 中期 |
| 6 | Ⅵ 多模态统一 RL | ⭐⭐⭐⭐⭐ | 很高 | 低 | 长期 |
| 7 | Ⅶ 自进化世界模型 | ⭐⭐⭐⭐⭐ | 很高 | 低 | 长期 |

> Ⅰ与Ⅱ的"影响×可行性"乘积最高——它们正好压在前三篇揭示的两个最大根因（信用分配与 Context Rot）上，且本研究线（多智能体搜索 + RL）天然需要二者同时解决。

---

## 六、与作者研究线的交汇

本篇四条最短路径都与**LLM 驱动多智能体搜索系统的端到端 RL 优化**直接咬合:
- **Ⅰ** 给多智能体异构分组提供层次化优势估计——解决"异构角色共享团队奖励"的信用分配;
- **Ⅱ** 让多智能体长程搜索摆脱 Context Rot,状态可跨 agent 传递;
- **Ⅲ** 把"检索器-推理器割裂"推广为"多 agent 子模块解耦"的统一训练范式;
- **Ⅶ** 用世界模型把多 agent 环境交互成本降下来,使端到端 RL 可负担.

相关发表见作者主页。

---

## 七、参考文献

下列工作均可在 arXiv 上按标题核对：

`LLDS 2512.04220` · `ASTER 2602.01204` · `CW-GRPO 2604.14267` · `SLATE 2602.23440` · `E-GRPO 2510.24694` · `Search-R1 2503.09516` · `R1-Searcher 2503.05592` · `ReSearch 2503.19470` · `ReasonRAG 2505.14069` · `LeTS 2505.17447` · `IG-Search 2604.15148` · `C-GRPO 2601.06021` · `WebAnchor 2601.03164` · `SIGHT 2602.11551` · `beta-GRPO 2505.17281` · `RE-TRAC 2602.02486` · `Pensieve/StateLM 2602.12108` · `ARC 2601.12030` · `DeepSearchQA 2601.20975` · `Rerank 2601.14224` · `CoSearch 2604.17555` · `AutoSearch 2604.17337` · `Cycle-Consistent 2604.12967` · `AgentIR 2603.04384` · `ZeroSearch 2505.04588` · `DeepDiver 2505.24332` · `Ares 2603.07915` · `AggAgent 2604.11753` · `AoT 2502.12018` · `VISOR 2604.09508` · `LMM-Searcher 2604.12890` · `MM-DeepResearch 2603.01050` · `ProMMSearchAgent 2604.20486` · `LiteResearcher 2604.17931` · `WebFactory 2603.05044` · `Search More Think Less 2602.22675` · `Kimi k1.5 2501.12599`.

此外 QPLEX（NeurIPS 2021）、MAVEN（NeurIPS 2019）等按会议引用。

---

<p align="center"><sub>诊断与方案基于 100+ 篇 2025–2026 文献。</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:F5C451,50:22D3EE,100:5B8CFF&height=90&section=footer&text=Diagnose%20the%20seams%2C%20not%20the%20symptoms.&fontSize=13&fontColor=ffffff&fontAlignY=45&animation=blink"/>
