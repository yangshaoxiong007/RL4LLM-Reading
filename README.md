<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2027,40:203A6C,75:5B8CFF,100:9B6BFF&height=240&section=header&text=RL4LLM%20Reading&fontSize=60&fontColor=ffffff&fontAlignY=42&fontAlign=50&desc=Long-Horizon%20%C2%B7%20Credit%20Assignment%20%C2%B7%20Process%20Reward%20for%20Search&descSize=15&descAlignY=66&descAlign=50&animation=fadeIn"/>

<p align="center">
  <a href="https://github.com/yangshaoxiong007/RL4LLM-Reading/blob/main/01-long-horizon-rl.md"><img src="https://img.shields.io/badge/01-Long_Horizon_RL-5B8CFF?style=flat-square&logo=readme&logoColor=white"/></a>
  <a href="https://github.com/yangshaoxiong007/RL4LLM-Reading/blob/main/02-credit-assignment-rl.md"><img src="https://img.shields.io/badge/02-Credit_Assignment-9B6BFF?style=flat-square&logo=readme&logoColor=white"/></a>
  <a href="https://github.com/yangshaoxiong007/RL4LLM-Reading/blob/main/03-process-reward-search.md"><img src="https://img.shields.io/badge/03-Process_Reward_for_Search-22D3EE?style=flat-square&logo=readme&logoColor=white"/></a>
  <br/><br/>
  <img src="https://img.shields.io/badge/level-expert_reading_notes-0E1117?style=flat-square"/>
  <img src="https://img.shields.io/badge/topic-RL_for_LLMs-FF4081?style=flat-square"/>
  <img src="https://img.shields.io/badge/focus-multi--agent%20%7C%20search%20%7C%20long--horizon-F5C451?style=flat-square"/>
</p>

---

> A curated, **expert-level** reading program on three interlocking frontiers of reinforcement learning for LLM agents. Each file is a self-contained deep dive: **problem formalization → method taxonomy → paper dissection → critique → open problems**. Papers are cited by title + venue + arXiv where verified from primary sources; IDs flagged `⟨id→?⟩` are pending cross-check against `arxiv.org/abs/<id>`.

## Why these three, together

The three threads are **one problem seen at three scales of credit attribution** — and they collapse into a single research agenda when agents search over long horizons:

```
                    long horizon ────────────────────────▶
   ┌──────────────┐        ┌──────────────────┐        ┌─────────────────────────────────┐
   │ Long-Horizon │  ═══▶  │ Credit           │  ═══▶  │ Process Reward for Search       │
   │ RL for LLMs  │        │ Assignment       │        │ (dense, step-level supervision) │
   │ (temporal)   │        │ (multi-agent +   │        │ over retrieval / reasoning      │
   │              │        │  multi-step)     │        │ trajectories                    │
   └──────────────┘        └──────────────────┘        └─────────────────────────────────┘
        O(n) steps               k heterogeneous                       search as MDP:
        of reasoning             roles sharing one                     state = context,
        / tool calls              sparse team reward                    action = query/tool,
                                                                       reward = step progress
```

- **Long-Horizon RL** is the *temporal* axis: how to learn when the reward arrives dozens of steps (or many search iterations) after the decision that mattered.
- **Credit Assignment** is the *structural* axis: when many agents / steps / tools co-produce one outcome, who/what gets credit — and critically, **for LLM agents the "agents" are often role-specialized LLMs sharing one whole-trajectory reward** (the setting of heterogeneous-group RL).
- **Process Reward for Search** is the *mechanism* that makes both tractable: it converts a terminal, sparse outcome into dense, *verifiable* per-step signals — and search is the domain where process supervision is most natural, because each retrieval/reasoning step has checkable intermediate correctness.

## Reading map

| # | Topic | Core question | Anchor method | File |
|---|-------|---------------|---------------|------|
| 01 | **Long-Horizon RL** | How to train LLMs on tasks whose reward lands far downstream? | GRPO, R1-style RL, length/generalization control | [`01-long-horizon-rl.md`](./01-long-horizon-rl.md) |
| 02 | **Credit Assignment** | Who/which step deserves credit in shared-reward LLM-agent teams? | Heterogeneous-group RL, COMA, MAPPO/QMIX lineage | [`02-credit-assignment-rl.md`](./02-credit-assignment-rl.md) |
| 03 | **Process Reward for Search** | How to give dense, faithful rewards to search/reasoning steps? | PRM (Let's Verify), Math-Shepherd, Search-R1 / R1-Searcher / ReSearch | [`03-process-reward-search.md`](./03-process-reward-search.md) |

## How to read

Each note follows the same skeleton so the three can be read as one argument:

1. **Why it matters** — the failure mode that motivates the thread.
2. **Formalization** — the MDP / MADP / surrogate-objective framing an expert needs.
3. **Method taxonomy** — the landscape, not isolated papers.
4. **Paper dissection** — what each actually contributes (and its limits).
5. **Critique** — where the field is over-claiming or under-theorized.
6. **Open problems** — the live questions, several of which map directly to *Multi-Agent Search + RL* (the author's own line).

## Author context

These notes are maintained alongside research toward **end-to-end RL optimization of LLM-driven multi-agent search systems** — the precise intersection where all three threads meet. See the author's profile for related publications.

---

<p align="center"><sub>§ Verify every arXiv ID against the primary source <code>https://arxiv.org/abs/&lt;id&gt;</code> before citing. Unconfirmed IDs are tagged <code>⟨id→?⟩</code>.</sub></p>
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:9B6BFF,50:5B8CFF,100:0F2027&height=100&section=footer&text=Toward%20faithful%20credit%20attribution.&fontSize=14&fontColor=ffffff&fontAlignY=45&animation=blink"/>
