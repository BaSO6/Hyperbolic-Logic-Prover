# FVxH：逐条应对与证据门槛

## 总判断

Reviewer 没有漏读核心结果；其最关键的 Q2、Q4 和 split 质疑都击中了真实问题。
最有效的 rebuttal 不是继续解释 65.75%，而是提供一个透明的同硬件 frontier，
同时主动收缩当前 artifact 无法支持的 Lie-dynamics 声称。

| Reviewer 问题 | 判断 | Rebuttal 必需动作 |
|---|---|---|
| Q1 split/Table 10 | 真实问题 | 承认 D.10 的 valid/test 标注冲突；撤回跨 split lift，所有新增数字只用 official test |
| Q2 native frontier | 真实且最高优先 | 同 A100 跑 native whole-proof 与 HLP 的 k=1,2,4,8,16,32，报告累计 wall-clock/显存/calls/tokens |
| Q3 depth>10 | 真实问题 | 先跑 ProofNet-test N=1；没有 ODE/RGD checkpoint 时撤回“SO(n,1) 将占优”的经验性表述 |
| Q4 forward/order | 真实实现冲突 | 用日志报告每 attempt 的精确 LLM/Lean calls；承认恢复代码实际是 retrieval→LLM→Lie |
| Q5 五个 31.00% | 真实问题 | 不解释成巧合；Table 10 先撤回，只有同 split 重跑后才能恢复 |
| Q6 ProofCompass/BFS | 缺失实验 | 主 rebuttal 先给 native DeepSeek matched frontier；若时间足够，只用官方公开 harness 在同 GPU 重跑，不能引用异机论文数字冒充 matched compute |

## E1：必须完成

运行 `cloud/run_rebuttal_n32.sh`。预注册：

- test set 恰好 244 题，否则 full run 拒绝启动；
- native 采用 DeepSeek 官方 CoT whole-proof prompt；
- temperature=1.0、top-p=0.95；
- 每个 `(problem, attempt)` 有确定且可续跑的独立 seed；
- 所有失败、timeout、crash 留在分母；
- k 为 1/2/4/8/16/32；
- 不在成功后停止采样，避免 adaptive stopping 改变时间定义；
- 主要图横轴为平均累计 wall-clock/problem，纵轴 pass@k；
- 同时报 raw solved count、Wilson 95% CI、peak VRAM、token、forward calls、
  Lean calls；
- 同题比较报告 a-only/b-only 和 exact McNemar。

只有当 HLP frontier 在相同时间附近更高，才能说“效率优势”。若曲线交叉，应写
“HLP 在低/中预算区域占优”，不能写 Pareto dominate。若不占优，仍可把贡献
收缩为 reviewer 已认可的 cone retrieval idea，但当前恢复代码本身不能验证 cone。

## E2：必须修正文稿，不应临时编数字

Table 10 的 `31.00%` 不是 244 题上的整数 solved count，对五个 backbone 完全
相同也缺乏可信的 raw records。Rebuttal 应直接说我们发现 aggregation/split
provenance 不足，撤回该表及 D.10 的 lift，不把它作为支持性证据。时间允许时
重跑也必须固定同一 test list、同一 harness、每个 backbone 独立 manifest。

## E3：有 GPU 时间再做

`MAX_ATTEMPTS=1 bash cloud/run_proofnet_n1.sh` 在 official ProofNet-test 186
题上运行两个可执行系统。它是 frontier significance 的 out-of-domain 检查，
不是 depth>10 Lie-vs-ODE-vs-RGD 实验。

真正的 depth 实验必须在看结果前固定 reference-proof depth，并具备三个训练好
的 variant checkpoint。当前 artifact 三者均不齐，所以最高诚信的处理是：

1. 承认 reviewer 指出的 Lean4 ProofNet port 已存在；
2. 删除/弱化 “SO(n,1) will dominate beyond depth 10”；
3. 把它改成 future hypothesis，而不是实验结论。

## E4：定性 trace

从 `results/rebuttal/hlp/traces/` 选同 split、同预算下：

- HLP 成功；
- native 在相同或更大累计时间仍失败；
- 展示每个 goal、retrieved hint、tactic、Lean response、坐标和调用数。

不要从 valid 挑图解释 test 数字，也不要只给 PCA 轨迹而省略 Lean tactic。

## 建议 rebuttal 文字骨架

> We thank the reviewer for identifying that our original efficiency comparison
> mixed inference regimes. We now evaluate the official DeepSeek-Prover-V1.5-RL
> whole-proof mode and our recovered stepwise system on the same A100, exact
> MiniF2F-test list, Mathlib commit, and checkpoint. We report the complete
> pass@k–wall-clock frontier through k=32, including all retrieval, HGCN, LLM,
> and Lean verification time, plus memory and call counts. [Insert measured
> numbers only after the run.]

随后必须主动说明：

> During artifact audit we also found that the recovered implementation order
> is retrieval→LLM→Lie and that the available checkpoints do not contain a
> trained Lie policy. We therefore label these measurements as “recovered HLP”
> rather than claiming they instantiate Algorithm 1, and we retract the
> unsupported depth>10 and cross-split Table 10 claims.

这会牺牲一部分原始 claim，但比用不可复现结果硬顶更可能挽回 reviewer 对原创性的
正面评价。

## Meta-review 的 cone 方向：现在可以怎样回答

Reviewer 对“为什么要往后看”的质疑里混合了两个层次。逻辑依赖边确实是
`premise → theorem`；但在给定当前 theorem 检索 premise 时，必须对每个候选
premise 检查 `theorem ∈ Cone(premise)`。因此计算看起来是从 query 反查 cone
apex，却没有反转逻辑蕴含方向。

新增实验固定同一 reconstructed encoder 和同一 proof-search harness，只改变：

1. embedding distance；
2. origin-angle + query-forward；
3. apex-angle + query-forward；
4. apex-angle + candidate-premise-to-query（corrected inverse）。

优先报告 held-out dependency recall@32/MRR 和 cone containment，再报告 N=1
verified proof success；只有 corrected inverse 同时优于合理 controls 时，才把它
作为 rebuttal 证据。完整的 N=32 只给最强 corrected arm 和官方 native baseline，
避免把四个昂贵 arms 全跑到 N=32。
