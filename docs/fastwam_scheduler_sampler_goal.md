# FastWAM Scheduler / Sampler Goal

这份文档是给新 worktree / 新 agent 的中文 handoff。目标是从当前
FastWAM 真实 10-step baseline 出发，系统搜索 scheduler / sampler 配置，
找到速度和成功率之间最好的 Pareto frontier，并把能力做成可复现、可追踪、
可比较的 EazyWAM optimization profile。

## 给 Goal Objective 的短版

```text
在新 worktree 中实现并评估 FastWAM scheduler/sampler 加速路线。先读 AGENTS.md 和 docs/fastwam_scheduler_sampler_goal.md。当前 FastWAM LIBERO/RoboTwin eval baseline 是 num_inference_steps=10，不要误写成默认 20。任务是训练免费的 inference-time 加速：把 scheduler 做成显式 optimization profile，保持默认路径不变，让用户能配置 num_inference_steps、sigma_shift 和后续 schedule/sampler 参数；trace 必须记录实际 scheduler 配置、timesteps/sigmas 摘要、latency、success rate、action drift、fallback reason。优先对齐 Hugging Face Diffusers 的 FlowMatchEulerDiscreteScheduler 思路，因为 FastWAM 当前是 continuous FlowMatch-style scheduler；DPM-Solver++/UniPC 只作为后续可行性评估，不能强行接入。由 agent 自己设计 coarse-to-fine 实验搜索，不要写死只跑某几个 step。真实 LIBERO/RoboTwin 实验去 SuperPod 跑，产出表格、图、命令、trace path、job id、推荐配置和不推荐配置。本地至少跑相关 pytest 与 ruff。不要宣称 parity_verified，除非有足够重复实验和统计证据。
```

## 当前真实状态

不要从“默认 20 steps”开始讲。当前 EazyWAM FastWAM eval 路径的真实
baseline 是 10 denoise steps：

- `src/eazywam/manifests/fastwam-libero.yaml` 的 LIBERO eval 默认
  `num_inference_steps: "10"`。
- `src/eazywam/manifests/fastwam-robotwin.yaml` 的 RoboTwin eval 默认
  `num_inference_steps: "10"`。
- `src/fastwam/configs/train.yaml` 中 `eval_num_inference_steps: 10`。
- `src/eazywam/backends/fastwam.py` 会优先使用
  `runtime_options["num_inference_steps"]`，然后使用 eval config，最后才
  fallback。
- `src/fastwam/models/wan22/fastwam.py` 里 `infer_action(...,
  num_inference_steps=20)` 只是函数级 fallback，不是当前产品/eval baseline。
- 当前 action scheduler 是 continuous FlowMatch-style scheduler，入口在
  `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`。

建议新 agent 在开工时先写一小段 baseline note，避免后续实验报告把问题讲错。

## 方法定位

Scheduler / sampler 加速不训练模型。它通过改变推理时的 denoise 路线来减少
模型调用次数，或者在相同 step 数下选择更好的 timestep / sigma schedule。

这和 TeaCache 不同：

- TeaCache：denoise step 数可能不变，但某些 step 内部少算。
- Scheduler / sampler：直接改变 denoise step 数、step 位置、step size 或
  update rule。

这和 CUDA Graph 也不同：

- CUDA Graph：同样的计算跑得更快。
- Scheduler / sampler：改变推理路径，让模型少被调用或更有效地调用。

## 开源参考

第一优先级参考 Hugging Face Diffusers 的 scheduler 体系，尤其是
`FlowMatchEulerDiscreteScheduler`。

原因：FastWAM 当前是 FlowMatch-style update，而不是标准 DDPM/DDIM 路线。
DPM-Solver++ 和 UniPC 是成熟 fast sampler，但是否适合 FastWAM 当前 action
scheduler，需要先判断模型输出参数化和 step update 是否匹配。

参考链接：

- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- FlowMatchEulerDiscreteScheduler docs:
  https://huggingface.co/docs/diffusers/en/api/schedulers/flow_match_euler_discrete
- FlowMatchEulerDiscreteScheduler source:
  https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
- Diffusers scheduler overview:
  https://huggingface.co/docs/diffusers/en/api/schedulers/overview
- DPM-Solver: https://github.com/luchengthu/dpm-solver
- UniPC: https://github.com/wl-zhao/unipc

## 设计哲学

- 不训练模型。
- 不写死一个实验结论。
- 不提前决定“必须降到几个 steps”。
- 不把 Diffusers / DPM-Solver / UniPC 逻辑硬塞进 core runner。
- core harness 只负责 profile、runtime option、trace、CLI、文档和实验复现。
- FastWAM native model / scheduler 层负责 scheduler 细节。
- 默认 FastWAM eval 行为不能被实验配置悄悄改变。
- 所有非默认 scheduler/sampler 配置必须显式开启。
- 结论来自实验：latency、denoise wall time、success rate、action drift 和
  failure cases 一起看。
- 本机是开发工作站；真实 checkpoint / simulator 实验去 SuperPod 跑。

## 阶段 0：确认 Baseline

目标：先确认现状，不要直接写代码。

必须确认并记录：

- 当前 LIBERO / RoboTwin 默认是 10 denoise steps。
- `runtime_options["num_inference_steps"]` 如何覆盖默认值。
- `sigma_shift` 如何传入。
- 当前 scheduler 的 timestep / sigma / delta / step 逻辑。
- 当前 trace 已有和缺失的 scheduler 字段。

产物：

- 一段简洁 baseline note。
- 如果发现 README 或 docs 中对当前默认值表述含糊，修正文档。

## 阶段 1：Scheduler Profile

目标：把 scheduler / sampler 配置做成清晰的 optimization profile。

要求：

- `scheduler` 是独立 profile，family 为 `scheduler`。
- 用户可以显式配置 `num_inference_steps`、`sigma_shift`，并为后续
  schedule/sampler 参数留接口。
- 默认 10-step baseline 不改变。
- profile 和 trace 的命名跟 `docs/optimization_profiles.md` 对齐。

trace 至少包含：

- `scheduler_name`
- `schedule_type`
- `num_inference_steps`
- `sigma_shift`
- `timestep_count`
- `timesteps` 或摘要
- `sigmas` 或摘要
- `denoise_wall_ms`
- `total_ms`

本地测试应证明：

- runtime option 能传到 FastWAM `infer_action()`。
- profile metadata 能进入 trace/backend metadata。
- 默认路径不变。

## 阶段 2：实验搜索，而不是写死配置

目标：由 agent 自己设计 coarse-to-fine 搜索，找到 Pareto frontier。

要求：

- 必须包含当前 10-step baseline。
- 必须探索比 baseline 更快的配置。
- 应包含少量更高成本配置作为 quality reference。
- 不要只跑一个 candidate 就下结论。
- 不要把某几个 step count 写死成唯一实验空间。
- 先粗搜，再围绕有希望的区域细搜。

每个配置至少记录：

- exact command
- trace path
- SuperPod job id
- `num_inference_steps`
- scheduler / schedule 参数
- `total_ms`
- `denoise_wall_ms`
- speedup
- success rate
- action drift
- failure cases
- fallback reason

## 阶段 3：对齐 Diffusers FlowMatch

目标：研究 FastWAM 当前 scheduler 与 Diffusers FlowMatchEuler 的关系。

要求：

- 阅读 Diffusers `FlowMatchEulerDiscreteScheduler`。
- 比较 FastWAM 的 shift、sigma、timestep、delta 和 step update。
- 判断哪些概念值得复现：
  - custom sigmas
  - custom timesteps
  - timestep spacing
  - shift / dynamic shift
  - schedule metadata
- 如果适合，实现到 FastWAM native scheduler 层。
- 如果不适合，写清楚原因。
- 不要在 core runner 里放模型数学细节。

验收：

- 文档说明两者对应关系。
- trace 能解释 schedule 是怎么生成的。
- 不破坏当前 10-step baseline。

## 阶段 4：扩展候选

目标：让 agent 自己选择值得实现和评估的 scheduler / sampler 候选。

候选包括但不限于：

- 当前 FastWAM shifted FlowMatch schedule 的参数搜索。
- custom timestep / sigma schedule。
- Diffusers 风格 timestep spacing。
- Karras-like schedule。
- AYS-like schedule。
- DPM-Solver++ 可行性评估。
- UniPC 可行性评估。

要求：

- 不要求全部实现。
- 必须解释为什么选择某些候选、放弃某些候选。
- 数学参数化不匹配的 solver 不要硬接。
- 新 sampler 必须是 experimental，不能默认启用。

## 阶段 5：SuperPod 实验和图表

真实实验必须在 SuperPod 上跑。

至少需要：

- FastWAM LIBERO baseline vs candidate configs。
- FastWAM RoboTwin baseline vs candidate configs。

图表至少包括：

- latency / denoise wall time 对比图。
- success rate 对比图。
- speedup vs success rate Pareto 图。
- 如果记录 action drift，画 drift vs speedup 图。

实验产物应落盘，包含：

- 配置表。
- 指标表。
- 图表。
- 命令。
- trace 路径。
- job id。
- 失败案例说明。
- 推荐配置和不推荐配置。

## 阶段 6：最终判断

最终报告必须回答：

- 当前 10-step baseline 表现如何？
- 哪些配置更快？
- 哪些配置成功率没有明显下降？
- 哪些配置速度快但失败率高？
- 是否存在推荐默认候选？
- 哪些配置只能 experimental？
- 是否值得继续做 DPM-Solver++ / UniPC？
- 是否需要更多重复实验才能判断？
- 是否应该更新 roadmap / README 的当前状态？

## 验收标准

- 默认 FastWAM eval 行为不被破坏。
- `scheduler` 是清晰 optimization profile。
- 用户可以显式配置 scheduler / sampler。
- trace 可以复现实验配置。
- 本地测试通过：

```bash
uv run pytest -q tests/test_registry.py tests/test_fastwam_native.py tests/test_eval_runner.py
uv run ruff check .
```

- 如果影响范围大，跑：

```bash
uv run pytest -q
```

- SuperPod 上有 LIBERO 和 RoboTwin 的实验结果。
- 有表格、有图、有命令、有 trace、有 job id。
- 给出推荐配置和不推荐配置。
- 不要宣称 `parity_verified`，除非有足够重复实验和统计证据。

## 建议 Worktree

```bash
git fetch origin
git worktree add ../wam-harness-scheduler-sampler -b feat/scheduler-sampler origin/main
cd ../wam-harness-scheduler-sampler
```
