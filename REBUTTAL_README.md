# HLP rebuttal：本地已准备，华为云一键启动

这是从原始保留目录
`/Users/liubeibeizzicloud.com/Documents/Hyperbolic-Logic-Prover` 的 Git HEAD
导出的独立工作副本。原目录没有被修改。这里新增的实验专门回答 reviewer
FVxH 的 Questions 2/4/6，并为 Question 3 准备 ProofNet 测试。

## 先看结论

当前恢复出来的代码**不能诚实地称为论文 Algorithm 1 的实现**。自动审计已经确认：

- 实现是 heap/A*，不是单 trajectory；
- premise retrieval 是 Poincaré distance top-k，没有 entailment-cone filter；
- 顺序是 retrieval → LLM → Lie move，不是论文描述的 move → retrieval → LLM；
- 下一次 retrieval 重新编码 goal 文本，没有使用 Lie move 后的点；
- 恢复的 checkpoint 只有 HGCN 权重，没有训练后的 Lie/tactic policy；
- 标为 “Entailment Cone Loss” 的训练代码实际是 norm regularizer + distance margin。

因此本套件把结果固定标记为 `recovered_hlp_astar_stepwise`，不会把它冒充
paper-strict HLP。`python -m rebuttal.audit_reproducibility` 会自动阻止误标。

## 本地能做什么

MacBook Pro 可以做、而且已经通过：

- Python 静态编译；
- 数据集/资产/源代码审计；
- N=1/2/4/8/16/32 累积统计、Wilson CI、配对 solved-set 和 exact McNemar；
- shell 脚本语法检查；
- 结果汇总与作图。

完整推理不能在这台 Mac 上做：恢复的 HLP 路径写的是 CUDA，native baseline
依赖 vLLM 0.4.1/CUDA，而且还要加载 DeepSeek-Prover-V1.5-RL。请使用
CUDA A100；40 GB 应能顺序运行，80 GB 更稳妥。native 与 HLP 是两个独立
进程串行运行，不会同时占显存。

## 上传并启动

从本机工作副本执行：

```bash
bash cloud/push_to_server.sh USER@HOST:/absolute/remote/hlp-rebuttal
```

然后 SSH 到服务器：

```bash
cd /absolute/remote/hlp-rebuttal
bash cloud/launch_huawei.sh
tail -f results/rebuttal/cloud_run.log
```

`launch_huawei.sh` 用 `nohup` 后台启动，SSH 断开也会继续。流水线会：

1. 建立隔离 venv；
2. 固定官方 DeepSeek commit、Mathlib commit 和 Lean 4.9.0-rc1；
3. 下载 DeepSeek-Prover-V1.5-RL 与 all-MiniLM-L6-v2；
4. 先跑 2 道题的 GPU/Lean smoke test；
5. 单卡时串行运行；检测到至少 4 卡时，自动按题目分成 4 份并行运行；
6. 计算 MiniF2F-test 的 N=1/2/4/8/16/32 frontier；
7. 保存逐 attempt 证明、seed、token、LLM/Lean 调用、wall-clock、VRAM、
   paired comparison 和图。

如果华为云不能直接访问 Hugging Face，可在启动前设置镜像，例如平台允许的
`HF_ENDPOINT`；pip 镜像也可通过标准 `PIP_INDEX_URL` 设置。流水线不会把
密码或 token 写入结果。

## 先做预算再跑满

两题 smoke 只用于发现环境问题。更可靠的时间预算是先跑完整 test 的 N=1：

```bash
MAX_ATTEMPTS=1 bash cloud/run_rebuttal_n32.sh
```

确认时间后再直接续跑到 N=32：

```bash
MAX_ATTEMPTS=32 bash cloud/run_rebuttal_n32.sh
```

JSONL 是逐行 fsync、按 `(method, problem, attempt)` 续跑的；中断后不会重算
已完成项。不要删除 `results/rebuttal/{native,hlp}/results.jsonl`。

如果希望一键直接跑满，使用前面的 `cloud/launch_huawei.sh` 即可。

## 单机 4×A100

华为云节点能看到 4 张卡时，`cloud/launch_huawei.sh` 会自动选择四卡脚本。
若环境已经 bootstrap 完成，也可直接运行：

```bash
MAX_ATTEMPTS=1 bash cloud/run_rebuttal_4gpu.sh
MAX_ATTEMPTS=32 bash cloud/run_rebuttal_4gpu.sh
```

执行方式是 data parallel，不是 tensor parallel：每张 A100 加载一个模型副本，
处理约 61 道 MiniF2F-test 题。四张卡先同时跑 native，严格合并完成后再同时跑
HLP，避免两套模型争抢同一张卡。默认每个 shard 使用 4 个 Lean worker，
整机共 16 个；可用 `LEAN_WORKERS_PER_SHARD=2` 降低 CPU/内存压力。

每卡日志在 `results/rebuttal/logs/{native,hlp}_shard_I.log`。结果先写到：

```text
results/rebuttal/native/shard-00-of-04/results.jsonl
results/rebuttal/hlp/shard-00-of-04/results.jsonl
```

四份 manifest 的数据哈希、模型/检查点哈希、seed 和推理参数必须相同；每题
必须有完整的 1..N attempts，且不能重复。只有这些检查全部通过，才原子写入
顶层 `native/results.jsonl` 和 `hlp/results.jsonl` 并生成统计。中断后重复同一
命令即可按 shard 续跑。

## 4 个单卡节点

四台机器使用同一 commit 和环境，分别运行（`I` 为 0、1、2、3）：

```bash
SHARD_INDEX=I NUM_SHARDS=4 MAX_ATTEMPTS=32 \
  bash cloud/run_rebuttal_shard.sh
```

把每台机器产生的 `native/shard-II-of-04/` 和 `hlp/shard-II-of-04/` 原样收集
到一台协调机的 `results/rebuttal/` 下，再运行：

```bash
NUM_SHARDS=4 MAX_ATTEMPTS=32 EXPECTED_COUNT=244 \
  bash cloud/merge_rebuttal_shards.sh
```

不要让多个节点写同一个共享 `results.jsonl`；只同步各自的 shard 目录。

## Reviewer 对应实验

主实验（必须先跑）：

- official MiniF2F-test 244 题；
- 同一 A100、同一 DeepSeek checkpoint、同一官方 Mathlib；
- native DeepSeek whole-proof 与 recovered HLP；
- 每题保留全部 32 次 attempt，即使较早已经成功；
- 报告 pass@k–time frontier、memory、token、LLM/Lean calls、95% CI；
- 同题配对比较和 exact McNemar。

ProofNet（Question 3，次优先）：

```bash
MAX_ATTEMPTS=1 bash cloud/run_proofnet_n1.sh
```

它会跑 official ProofNet-test 的 186 题。注意：这只能回答“当前恢复系统在
ProofNet 是否有效”，**不能**验证论文 SO(n,1) vs ODE vs RGD 的 depth>10
预测，因为 ODE/RGD 实现和对应训练 checkpoint 没有保留下来。

完整实验定义见 `rebuttal/EXPERIMENT_PROTOCOL.md`，逐条 reviewer 策略见
`rebuttal/REVIEWER_RESPONSE_PLAN.md`。

## 输出

- `results/rebuttal/audit.json`：资产哈希和 paper-strict gate；
- `results/rebuttal/native/results.jsonl`：native 逐 attempt 结果；
- `results/rebuttal/hlp/results.jsonl`：recovered HLP 逐 attempt 结果；
- `results/rebuttal/{native,hlp}/shard-II-of-04/`：四卡独立断点与 manifest；
- `results/rebuttal/summary/frontier.{json,csv,png}`：accuracy–time frontier；
- `results/rebuttal/summary/paired_comparisons.{json,csv}`：配对统计；
- `results/rebuttal/gpu_samples.csv`：独立 `nvidia-smi` 采样；
- `results/rebuttal/smoke/runtime_estimate.json`：初步全量耗时投影；
- `results/rebuttal/*/manifest.json`：数据/model/checkpoint/commit/硬件记录。

## 本地回归测试

```bash
PYTHONPYCACHEPREFIX=/tmp/hlp-pycache \
  python3 -m unittest -v tests.test_rebuttal_common
bash -n cloud/*.sh src/system2/run_repl_wrapper.sh
python3 -m rebuttal.audit_reproducibility
```
