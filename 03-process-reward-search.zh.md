<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:22D3EE,100:5B8CFF&height=140&section=header&text=03%20%C2%B7%20%E6%90%9C%E7%B4%A2%E7%9A%84%E8%BF%87%E7%A8%8B%E5%A5%96%E5%8A%B1&fontSize=28&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

<p align="center"><sub>🇨🇳 中文版 &nbsp;·&nbsp; <a href="./03-process-reward-search.md">🇬🇧 English</a> &nbsp;·&nbsp; 交叉引用 <a href="./01-long-horizon-rl.zh.md">01 长程</a> · <a href="./02-credit-assignment-rl.zh.md">02 信用分配</a></sub></p>

## 1 · 为什么重要

**过程奖励**是推理/搜索轨迹上的*步级、可检验*信号 $r_t$,相对末端*结果奖励* $r_H$。它同时化解前两条线索:

- 对照 doc01:密集 $r_t$ 把信用的"时间 horizon"从 $H$ 缩到 ~1,每步就地被奖;
- 对照 doc02:每步/每轮奖励*本身就是*信用信号——它把团队结果分解成可归因的片段,无需集中式 $Q$。

**搜索是过程奖励最自然的领域**,因为每个检索/推理步都有*可检验的中间状态*:抓到的证据是否支撑局部答案?查询是否检索到*新*且有用的信息?推理步是否局部正确?搜索把"过程奖励"从手标奢侈品(PRM800K)变成*部分可机检*的东西——这正是 Search-R1 类工作能用远少人工标注训练的原因。

## 2 · 形式化

把搜索建模为 MDP:$s_t=$累积上下文,$a_t\in\{$查询, 工具调用, 推理步, 终止$\}$。结果 RL 用 $r_t=0(t<H),r_H=\text{verifier}(\text{answer})$。**过程奖励 RL** 把它换成 $r_t$(定义或*从 rollout 学*)。三态 $r_t$:

1. **人工标注 PRM** — 分类器 $\rho_\phi(s_t,a_t)\to[0,1]$,用步标签训练(PRMM800K)。
2. **自动标注 PRM(MC)** — 用"到达此步后最终成功"的经验概率估计 $\rho_\phi$,靠采样续写(Math-Shepherd);让 PRM 廉价的窍门。
3. **隐式/可验证过程奖励** — 不单独训 $\rho_\phi$;用辅助目标把结果奖励塑形到步(如 token 级优势,或 Search-R1 的逐搜索步规则)。

把 $r_t$ 变成梯度的估计器与 doc01 同:在步粒度上做留一/组优势——这正是过程奖励与 GRPO 可组合之处,也是 search-RL 三件套所在。

## 3 · 方法谱系(含 2025–2026 搜索 RL 精确分类)

下表融合经典 PRM 与最新搜索 RL,标注【奖励类型 / 作用域 / RL 算法】——这是过程奖励现状最细的一张图,数据交叉核对自公开 awesome-list 与 arXiv 标题核验。

| 工作 | $r_t$ 来源 | 奖励类型 | 作用域 | RL 算法 | arXiv |
|---|---|---|---|---|---|
| PRM800K / Let's Verify | 人工步正确性 | PRM | step | (BoN 评估) | **2305.20050** ✓ |
| Math-Shepherd | MC 自动标注 | PRM | step | RL on PRM | **2312.08935** ✓ |
| Search-R1 | 逐检索步规则 | 规则 PRM | step | GRPO | **2503.09516** ✓ |
| R1-Searcher | 逐检索步规则 | PRM | step | GRPO | **2503.05592** ✓ |
| ReSearch | 推理-证据对齐 | PRM | step | GRPO | **2503.19470** ✓ |
| ReasonRAG | 最短路径(MCTS) | PRM | step | DPO | **2505.14069** ✓ |
| LeTS | 过程+结果混合 | ORM+PRM | step | RL | **2505.17447** ✓ |
| E-GRPO | 实体匹配率作部分奖励 | PRM | step | GRPO | **2510.24694** ✓ |
| StepSearch | 信息增益/冗余惩罚 | ORM+PRM | step | PPO | **2505.15107** ✓ |
| IGPO / IG-Search | 信息增益拟奖励 | ORM+PRM | step | IGPO/GRPO | **2510.14967** ✓ / **2604.15148** ✓ |
| CW-GRPO | LLM judge 贡献分 | ORM+PRM | step | CW-GRPO | **2604.14267** ✓ (ACL26) |
| SLATE | 截断步级+分解 | PRM | step | 步级 GRPO | **2602.23440** ✓ |
| C-GRPO | 引用 rubric | PRM | step | GRPO | **2601.06021** ✓ |
| Search-R2 | actor-refiner 密集 | PRM | single | — | **2602.03647** ✓ |
| PRMBench | PRM 评测 | (评测) | — | — | **2501.03124** ✓ |

> 谱系结论:2025–2026 的搜索 RL **几乎全员 step-level + PRM**(或 ORM+PRM 混合),纯结果 RL 已成少数派——这是 doc03 §5 批判"PRM 普遍更优"被实践接受、同时"judge/启发式脆弱"被实践继承的双面证据。

## 4 · 论文剖析(专家级)

### PRM800K(arXiv:2305.20050✓)
**贡献。** 把人工*步级*标签(每 CoT 步对/错)扩到 ~800k;训 $\rho_\phi$;证明 **best-of-N with $\rho_\phi$ 超过结果奖励 best-of-N**。奠基实证:*过程监督有用*。
**专家警示。** 增益在(i)任务可分解为可检验步、(ii)人工能可靠标这两条件下最大。研究/搜索只部分满足(ii);本论文的遗产不是"全标",而是**"找到廉价 verifier"**——search-RL 三件套正是用检索期启发式替换人工标注员。
**局限。** 标注员在模糊步上分歧;标注成本使不可复现于大规模。

### Math-Shepherd(arXiv:2312.08935✓)
**贡献。** **自动标注**步奖励:对每个部分解采样续写,+ 标签=续写常达正确*终态*、− 否则。去掉人工标注员,PRM 变数据生成——这是让过程奖励 RL 可部署的实际枢纽。
**微妙处。** 自动标签与*标注期策略*一致:策略变好,标签变。既是特性(自改进监督)也是 bug(PRM 滞后、标注/学习分布偏移)。同一张力是某些 regime 下 R1 式结果 RL 反超 PRM-RL 的原因:结果 RL 无需策略一致标注员。
**跨档读。** Math-Shepherd 的 MC 标注*就是*一种信用分配(doc02):它按步估"到达此处后的成功概率"——即该步对结果的边际信用。PRM 与 COMA 在估同一对象的表亲。

### Search-RL 三件套:Search-R1 / R1-Searcher / ReSearch
**共同贡献。** 在 RL 下训练**推理-检索交错**的 LLM 智能体,在最终结果奖励外用**逐搜索步过程信号**(证据充分性/答案 grounding)。过程奖励在此同时遇上长程(doc01)**和** search-as-MDP(§2)。
- **Search-R1**(arXiv:2503.09516✓):每次检索为动作,按检索是否改善局部答案塑形奖励 → 过程奖励=查询的边际信息增益。
- **R1-Searcher**(arXiv:2505.03392✓……实为 2503.05592✓):学*何时搜、搜什么*,$r_t=$检索质量——最接近开放设定下"可验证过程奖励"。
- **ReSearch**(arXiv:2503.19470✓):分离推理与搜索状态,用过程奖励把推理跳转与检索证据对齐——信号干净因跳转局部可检。
**专家综合(三档交汇)。** Search-RL 是**当动作空间是搜索时,过程奖励可比结果 verifier 更廉价**的存在证明,因检索步有可检中间态。这正解释为何*作者的多智能体搜索 + RL 线*落在此:角色特化搜索 agent 共享一个结果,唯一可负担的信用信号是搜索步过程奖励——让 doc02 的组间信用问题*从数据可解*(doc02 §6 开放问题)。

### 贡献/步级信用四连胜(RAGRL / CW-GRPO / SLATE / E-GRPO)——credit 走向过程奖励
这四篇把过程奖励与信用分配(doc02)焊在一起,是 2025–2026 最清晰的"以 PRM 做信用"谱系:
- **ReasonRAG**(arXiv:2505.14069✓):系统证明细粒度过程奖励**优于**纯结果,5K 样本 ≈ Search-R1 90K(**18× 数据效率**)——过程奖励不只是更好,还更省。
- **E-GRPO**(arXiv:2510.24694✓):标准 GRPO 浪费实体信息;为错样本分与实体匹配率成正比的*部分*奖励,从"近似正确"轨迹中学习。
- **SLATE**(arXiv:2602.23440✓):截断步级采样 + 分解过程奖励(推理质量+查询质量+答案正确),7B +7.0%、3B +30.7%(对 Search-R1)。
- **CW-GRPO**(arXiv:2604.14267✓, ACL26):LLM judge 估每轮检索贡献,把 outcome 优势按贡献重缩放——细粒度信用,8B +5.0%、1.7B +6.3%。
**联合读。** 此谱系坐实 doc02 的批判:**GRPO 的可交换性在多步搜索上失效,已形成 E-GRPO→CW-GRPO→SLATE→C-GRPO 的"以贡献/步级补信用"主流;但全靠启发式或外部 judge,缺精确因果归因**——这正是组间信用 + auto-PRM 的未解交汇。

## 5 · 批判

- **"PRM 总优于 ORM。"** 仅在步可廉价验证时成立。研究/搜索里 PRM 要么启发式(漂移)、要么学得(Math-Shepherd 滞后);相对结果 RL 的增益是 regime 相关、非普适。近期结果 RL 复现成功(R1 类)反证了 PRM 的普适优越。
- **auto-PRM 策略一致欠理论。** MC 标签是策略定义的;少有论文分析 PRM-learner 反馈环。它行为像自蒸馏,有坍缩/停滞风险——实践中未验证。
- **搜索奖励塑形是脆弱手设计。** "证据充分性""新颖性""grounding" 论文各异,三件套间很少一起 ablate。缺共享、可验证的搜索步奖励基准(PRMBench 只评 PRM 本身,不评搜索步奖励)。

## 6 · 开放问题(→ 作者的研究线)

1. **统一、可验证的搜索过程奖励。** 把三件套的临时步奖励收敛到一个可检 surrogate(搜索版的数学 verifier)。
2. **多智能体 rollout 下的 auto-PRM。** Math-Shepherd 假设*单一*续写策略;异构搜索里标注的"最终成功"混了角色贡献。信用感知 auto-PRM = docs 02 ⇆ 03 的开放枢轴。
3. **PRM 作 critic 而非 reward。** 用 PRM 作信用(COMA 式,doc02)的集中 critic,而非仅奖励附加——LLM 搜索 agent 几乎未探。
4. **研究的 verifier。** 终极约束:无研究 verifier,再完美的过程奖励机制也只能在 verifier 可信之处止步。

## 参考文献(经 § 逐条核验)

- Let's Verify(PRMM800K) — **arXiv:2305.20050** ✓
- Math-Shepherd — **arXiv:2312.08935** ✓
- Search-R1 — **arXiv:2503.09516** ✓;R1-Searcher — **arXiv:2503.05592** ✓;ReSearch — **arXiv:2503.19470** ✓
- ReasonRAG — **arXiv:2505.14069** ✓;LeTS — **arXiv:2505.17447** ✓;E-GRPO — **arXiv:2510.24694** ✓
- StepSearch — **arXiv:2505.15107** ✓;IGPO — **arXiv:2510.14967** ✓;IG-Search — **arXiv:2604.15148** ✓
- CW-GRPO — **arXiv:2604.14267** ✓(ACL26);SLATE — **arXiv:2602.23440** ✓;C-GRPO — **arXiv:2601.06021** ✓
- Search-R2 — **arXiv:2602.03647** ✓;PRMBench — **arXiv:2501.03124** ✓
- GRPO/Dr.GRPO/DAPO/LLDS/ASTER 估计器语境 — doc 01

<p align="center"><sub>03 / 03 · 搜索过程奖励 · 交叉引用 01 长程、02 信用分配</sub></p>
