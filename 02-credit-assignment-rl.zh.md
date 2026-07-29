<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,55:9B6BFF,100:22D3EE&height=140&section=header&text=02%20%C2%B7%20%E4%BF%A1%E7%94%A8%E5%88%86%E9%85%8D%20%2B%20RL&fontSize=30&fontColor=ffffff&fontAlignY=50&fontAlign=50&animation=fadeIn"/>

<p align="center"><sub>🇨🇳 中文版 &nbsp;·&nbsp; <a href="./02-credit-assignment-rl.md">🇬🇧 English</a> &nbsp;·&nbsp; 交叉引用 <a href="./01-long-horizon-rl.zh.md">01 长程</a> · <a href="./03-process-reward-search.zh.md">03 过程奖励</a></sub></p>

## 1 · 为什么重要

信用分配问的是:给定**一个共享的团队奖励**,哪个智能体/哪一步/哪个动作该记功或背锅?它是长程归因(doc 01)的结构孪生——doc 01 是信用在*时间*轴,本文档是信用在*智能体与角色*轴;二者是同一套数学、不同坐标。对 LLM 多智能体系统,问题新近尖锐化,因为:

- "智能体"是**角色特化的 LLM**(规划/搜索/验证/撰写),共产生一条整轨迹奖励;
- 单一角色失败很少会决定性地废掉团队奖励(不像数学里一步走错),所以共享奖励**按角色看更噪**;
- 而自然的 RL 估计器(GRPO 的组基线,doc 01)在**角色异构下是有偏的**,因为它假设样本可交换。

因此,共享一个稀疏结果的角色异构团队,同时命中三条线索的最坏情形:长程(多搜索步)、共享奖励(团队结果)、稀疏验证(研究、非数学)。

## 2 · 形式化

$n$ 智能体的合作 MADP:联合策略 $\pi(\mathbf a\mid\mathbf s)=\prod_i\pi_i(a_i\mid\tau_i)$,共享奖励 $r(\mathbf s,\mathbf a)$,联合优势 $A^{\text{team}}$。智能体 $i$ 的梯度:

$$\nabla_\theta J=\mathbb E\big[(A^{\text{team}})\,\nabla_\theta\log\pi_i(a_i\mid\tau_i)\big]$$

病症一目了然:**每个**智能体的梯度由*同一个* $A^{\text{team}}$ 缩放——团队优势。一个无用智能体与一个关键智能体拿到相同学习信号。信用分配即把 $A^{\text{team}}$ 换成**反事实优势** $\hat A_i$,孤立出智能体 $i$ 的边际贡献:

$$\hat A_i(s,\mathbf a)\approx Q(s,\mathbf a)-Q(s,(a_{-i},\bar a_i))$$

即"因为智能体 $i$ 出了 $a_i$ 而非默认/平均 $\bar a_i$,结果好了多少"。整个领域就是**在不同表征约束下对这个反事实的估计器**。

## 3 · 方法谱系

| 流派 | 反事实形式 | 优势 | 局限 | 代表 |
|---|---|---|---|---|
| **差分奖励/COMA** | $D_i=Q(\mathbf a)-Q(\mathbf a_{-i},c_i)$ | 精确的逐智能体边际 | 需学习版集中式 $Q$ | COMA(arXiv:1705.08926✓) |
| **价值分解(VDN→QMIX→QPLEX)** | 设 $Q_{\text{tot}}=f(Q_1,\dots,Q_n)$,单调保 IGM | 可扩展,有信用一致性定理 | QMIX 单调性限制了对抗角色 | QMIX(arXiv:1803.11485✓);QPLEX(NeurIPS21*) |
| **CTDE+PPO(MAPPO)** | 训一个集中 $V_\phi$,演员本地 | 简单、易扩展 | $V_\phi$ 共享 → **无逐角色信用**,退回 $A^{\text{team}}$ | MAPPO(arXiv:2103.01955✓) |
| **角色感知/异构分组** | 把智能体划入同质**组**,组内基线 | 组内恢复可交换、组间隔离角色 | 组的定义是设计偏置 | 异构分组 RL(作者, ACL2026);MADDPG(arXiv:1706.02275✓) |
| **隐式角色发现(MAVEN)** | 学角色隐变量 $z_i$,值函数条件化 $(z_i,\tau_i)$ | 角色涌现而非手工 | 信用依赖 $z$ 质量;不稳 | MAVEN(NeurIPS19*) |
| **结果重加权的细粒度信用** | 把 outcome 优势按每轮贡献**重缩放**,无需集中 $Q$ | 在 LLM token 上可落地 | 仍靠 LLM judge/启发式,非精确因果 | CW-GRPO(arXiv:2604.14267✓,ACL26);SLATE(arXiv:2602.23440✓);E-GRPO(arXiv:2510.24694✓) |

> *注:QPLEX(NeurIPS21)、MAVEN(NeurIPS19) 的 arXiv id 本会话未能逐条核验匹配,按会议引用,不臆造数字 id。

## 4 · 论文剖析(专家级)

### COMA(arXiv:1705.08926✓)
**贡献。** 集中式 critic 对智能体 $i$ 的*其他*动作求期望 $Q$,用队友实际动作;反事实基线 $=\sum_{a'}\pi_i(a'\mid\tau_i)Q(s,(a_{-i},a'))$。这是**典范级**逐智能体信用信号:策略梯度的"只奖励偏离你平时会做的部分"。
**对 LLM 智能体的意义。** COMA 的思想——别把"反正也会产生的那部分结果"算给某个智能体——正是 GRPO 在角色异构下缺失的。一个搜到显然首条的 searcher,与一个做了艰难检索的 searcher,拿到相同 $A^{\text{team}}$;COMA 会把前者信用缩到接近零。
**局限。** 需在组合联合动作上学习集中 $Q$——离散小 $n$ 可行,LLM token 级动作上痛苦。

### QMIX(arXiv:1803.11485✓)(+ VDN 1706.05296✓ → QPLEX NeurIPS21*)
**贡献。** 把 $Q_{\text{tot}}$ 分解为逐智能体 $Q_i$ 的**单调**超网(VDN 是可加特例;QMIX 放宽到单调;QPLEX 进一步到 duplex dueling)。单调性保证 **IGM**:$\arg\max_{\mathbf a}Q_{\text{tot}}=(\arg\max_{a_i}Q_i)_i$,故去中心化贪心=全局贪心——干净的*信用一致性*定理。
**专家读。** IGM 是领域唯一严格的信用一致性结果,代价是:**单调性禁止表示角色边际贡献对抗的情形**(一智能体得利=另一智能体失信用)。LLM 团队里角色重叠(两个 searcher)、相互纠错(planner vs. verifier)天然*非单调*——QMIX 对它们是错误归纳偏置,这是可引用的、精确的理由,去偏好**异构分组**(不做单调假设、组内恢复可交换)而非 QMIX 式分解。

### MAPPO(arXiv:2103.01955✓)
**贡献。** PPO + 共享集中 $V_\phi(s)$ + 逐智能体演员。多智能体 LLM RL 的障眼法:"上 MAPPO 就行"。
**尖锐批判。** MAPPO 训的**一个共享团队优势**(或 agent-shared value);它只解决*跨智能体稳定*,**不解决信用分配**。把 LLM 角色映射成 MAPPO 演员 = 把信用问题*反退回* $A^{\text{team}}$,每个角色同酬。MAPPO 作为"多智能体基线"的流行,掩盖了它让信用问题*更糟*而非解决这一事实。

### 异构分组 RL(作者的研究线, ACL 2026)
**贡献/定位。** 把智能体划入角色同质**组**;组内用 GRPO 式可交换(组基线无偏),组间用组级结构处理。这是 COMA 的逐角色反事实(贵)与 MAPPO 的共享值(信用盲)之间的桥:廉价买到*无偏组内*信用,只牺牲细粒度*组间*区分——而组间差异由组结构显式化而非隐式。
**为何是对 GRPO 的正确外科手术。** GRPO 的偏差源于可交换性失效;异构分组*按构造恢复*可交换。它留下的开放问题——无集中 $Q$ 的组间信用——正好是 COMA/QMIX 机制若可行可重新引入之处。

### CW-GRPO:贡献加权 GRPO(arXiv:2604.14267✓, ACL 2026)
**贡献。** 用 LLM judge 估每轮检索的*贡献分数*,把 outcome 优势按每轮贡献**重新缩放**——把 COMA 的"反事实边际"思想,以 judge 近似落到搜索步级。实证:Qwen3-8B +5.0%、Qwen3-1.7B +6.3%。
**专家读。** 这是 doc02 §3 谱系里第一类"在 LLM token 上可落地的细粒度信用",并与 doc03 (过程奖励) 共享同一估计对象:每步对结果的边际贡献。它坐实了我对 GRPO 可交换性的批判**已被 ACL 2026 正式承认**。
**待解。** 贡献分数靠 LLM judge、非精确因果归因;judge 偏差会污染信用——这正是 doc03 §5 批判的"search reward shape 脆弱手设计"在信用侧的镜像。

## 5 · 批判

- **"CTDE 解决了信用分配。"** 否。CTDE *使能*集中式值共享;不分配信用。多数"多智能体 LLM PPO"在做 MAPPO,却声称拿到了估计器给不了的信用增益。
- **IGM/单调性对 LLM 角色是负债。** 众人引用的定理(QMIX)恰恰*禁止*真实团队里主导的非单调角色交互,却很少被承认。
- **角色标签当免费午餐。** 把角色视为固定/已知,让异构分组显得干净——但深度研究里角色*特化本身*应被学。MAVEN 式隐角色对 LLM 智能体欠探索,因为 token 动作使 $z$-条件估计不稳。
- **judge-based 信用的循环。** CW-GRPO/SLATE 类方法用 LLM judge 估贡献,但 judge 与被训练策略同源——存在自我强化偏差风险,缺乏理论保证(呼应 doc03 的 auto-PRM 策略一致性问题)。

## 6 · 开放问题(→ 作者的研究线)

1. **无集中 $Q$ 的组间信用。** 异构分组 RL 的诚实缺口:过程奖励(doc 03)能否廉价地、从数据提供组间信号?
2. **学得而非手工的角色。** 共享奖励 + 长程下的端到端角色涌现。
3. **非单调信用一致性。** 一个弱于 IGM、强于 $A^{\text{team}}$ 的定理——LLM 团队缺失的形式对象。
4. **verifier 即信用信号。** 深度研究里唯一可靠的逐步信用可能是 verifier;让 verifier 廉价是 doc 03 的活。

## 参考文献(经 § 逐条核验)

- COMA — **arXiv:1705.08926** ✓
- QMIX — **arXiv:1803.11485** ✓
- VDN — **arXiv:1706.05296** ✓
- MAPPO — **arXiv:2103.01955** ✓
- MADDPG — **arXiv:1706.02275** ✓
- CW-GRPO — **arXiv:2604.14267** ✓(ACL 2026)
- SLATE — **arXiv:2602.23440** ✓
- E-GRPO — **arXiv:2510.24694** ✓
- QPLEX — NeurIPS 2021(按会议引用);MAVEN — NeurIPS 2019(按会议引用)
- 异构分组 RL(作者, ACL 2026)无 arXiv

<p align="center"><sub>02 / 03 · 信用分配 · 交叉引用 01 长程、03 过程奖励</sub></p>
