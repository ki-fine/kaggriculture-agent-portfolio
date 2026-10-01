# 简历条目与面试说明

这些表述对应已完成的项目工作。最终排名公布后，再按官方记录补充比赛结果。

## 中文简历条目

**Kaggriculture 多智能体策略 Agent｜三人团队｜策略整合与最终构建｜2026.09–2026.10**

Python · 仿真规划 · 行为克隆 · 模型集成 · 实验评估

- 汇总三人团队的生产与交易方案，整合固定任务路线、六种对手影子模型和有限预算搜索，完成最终 Agent 构建与 Kaggle 提交
- 处理交易排程与既有往返交易的时序冲突；复用相同动作前缀并保持 SDK 状态复制语义，在 32 局／16 配对本地对照中保持动作及现金一致，平均配对回调耗时比下降 15.6%
- 建立固定种子、双席对照、官方现金审计与源码哈希对应的验证流程；最终 R7 在 2 个条件世界、3 类对手的 12 个配对中净金币差均增加，平均 +961，保留异常标记与样本边界

如果版面较小，可以使用以下两条：

- 负责三人团队策略整合与最终 Agent 构建，对接生产路线、六种对手影子模型和市场搜索，并处理跨模块的资源与交易时序冲突
- 完成固定种子／双席配对、回放现金审计及哈希验证；32 局独立对照保持动作与现金一致，平均配对回调耗时比下降 15.6%，最终版本已提交，成绩待公布

## 机器学习岗位的补充表述

机器学习方向可从数据、模型评估和部署差异切入，具体训练口径见 [ML_EXPERIMENTS.md](ML_EXPERIMENTS.md)。

> 参与完整动作行为克隆实践，分析离线动作一致率与自主完整对局收益的差异，复盘状态分布变化、动作编码和错误累积；据此调整模型评估与最终策略选择。

需要突出训练实践时，可使用更具体的一条：

- 参与 GRU 与 masked Transformer 的完整动作行为克隆实验，使用 416 条轨迹、29.9 万训练样本；世界种子不重叠的开发验证集动作一致率达 88.25%，并用自主对局检验部署效果，保留 0 胜 6 负的负结果

六种影子模型是规则／行为仿真假设，不能写成“训练六个深度学习模型”。强化学习没有完成训练，不作为项目成果。

## English resume entry

**Kaggriculture strategy agent | Three-person team | Strategy integration and final build**

- Integrated team farming schedules, six opponent behavior models, and budgeted market search; resolved cross-module resource and transaction-order conflicts and built the final competition submission.
- Reused shared action prefixes and preserved SDK state-copy semantics. In 16 matched local comparisons across 32 games, actions and cash stayed identical while the mean paired callback-time ratio decreased by 15.6%.
- Organized fixed-seed, swapped-seat evaluation, cash reconciliation, and SHA-256 release checks. R7 improved cash margin in all 12 comparisons on a two-world, three-opponent conditional panel, with a mean gain of 961 coins; final competition results are pending.

## 60 秒项目介绍

> 我做的是 Kaggle 的双人农场经营 Agent，负责整合三人团队思路并完成最终代码构建。这个任务同时有生产义务、共享市场和运行预算。我们把浇水喂食等任务固定执行，把推理用于策略和交易选择。我重点处理了模块时序冲突、重复仿真和状态复制问题，并用固定种子、双席对照和现金审计验证。最后提交了 R7，最终排名还没公布；本地结果和限制都在 GitHub 上可以核对。

## 需要能够独立解释的技术问题

面试前应结合公开代码和数据，独立讲清这些问题：

| 问题 | 回答需要覆盖的内容 |
|---|---|
| 为什么把义务写成固定任务？ | 动作预算、跨日依赖、遗漏成本，以及固定路线的适用边界 |
| 为什么使用净金币差？ | 共享市场也影响对手，自身现金增加不等于竞争优势 |
| 同 seed 是否足够公平？ | 控制初始条件、双席；策略可能改变后续随机数消耗与店铺路径 |
| 状态复制为何困难？ | 字典／属性视图偏离、别名、循环、协议覆盖和保守回退 |
| 为什么行为克隆准确率高仍可能失败？ | 状态分布变化、错误累积、动作序列以及部署目标 |
| 团队里你具体做了什么？ | 汇总方案、接口整合、失败复现、版本筛选与最终构建 |
| AI 工具参与了多少？ | 说明实际辅助过程，展示自己能解释并复核的设计与代码 |
| 赛后最需要改进什么？ | 失败结论的范围、重要想法是否实际验证、AI 写入规则的审查与复查条件 |

过程反思与下一次改进计划见 [RETROSPECTIVE.md](RETROSPECTIVE.md)。面试时区分已经完成的工作和下一次准备实施的流程。

## 成绩公布后的措辞

只有在官方结果可核对后，才增加最终排名、参赛队伍数或奖牌。百分位需使用同一榜单口径计算。当前材料保留“成绩待公布”，不写预测排名或奖牌。
