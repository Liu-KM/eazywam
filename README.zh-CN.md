<p align="center">
  <img src="docs/assets/logo/eazywam-logo-readme.png" alt="EazyWAM logo" width="760">
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
</p>

# EazyWAM

EazyWAM 是一个面向 world-action model 的部署与推理加速框架。它把分散在不同
仓库里的 checkpoint、运行环境、资产准备、评测脚本、服务入口、优化开关和
trace，整理成一个以 model id 为中心的 `wam` 工作流。

## 快速上手

从 PyPI 安装 core CLI：

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install eazywam
```

如果要开发或改源码，再 clone 仓库：

```bash
git clone https://github.com/Liu-KM/eazywam.git
cd eazywam
```

### 创建源码开发环境

使用 `uv` 创建一个干净的 Python 3.10+ 环境，并安装当前源码。它会安装
`pyproject.toml` 里声明的 core package 依赖；core CLI 不需要额外执行
`requirements.txt`。

方案 A：手动安装

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

方案 B：使用 setup 脚本

```bash
scripts/setup_core_env.sh
source .venv/bin/activate
```

真实 WAM runtime、checkpoint、simulator 和 GPU 依赖会通过对应模型的
`doctor` 和 `prepare` 路径处理。

如果你更习惯 Conda，也可以先创建并激活 Python 3.10+ Conda 环境，然后用
`python -m pip install -e .` 安装当前源码。

### 查看模型

```bash
wam list
wam info fastwam-libero
```

### 验证本地 harness

```bash
wam run fake-open-loop
wam run fake-open-loop --opt fake_cache
```

### 验证本地 policy server

```bash
wam serve fake-open-loop --smoke
```

### 准备 FastWAM 资产

```bash
wam doctor fastwam-libero --cache-dir /path/to/wam-cache
wam prepare fastwam-libero --cache-dir /path/to/wam-cache --download --asset eval
```

### 用一条 observation 跑 FastWAM

```bash
wam run fastwam-libero \
  --input examples/fastwam_libero/obs.json \
  --output /tmp/fastwam-action.json \
  --cache-dir /path/to/wam-cache
```

### 在 LIBERO 里评测 FastWAM

需要在已经准备好的 FastWAM runtime 里运行。

```bash
wam eval fastwam-libero \
  --workload libero-single-task \
  --task-id 0 \
  --num-trials 1 \
  --cache-dir /path/to/wam-cache
```

### 在 RoboTwin 里评测 FastWAM

需要在已经准备好的 FastWAM + RoboTwin runtime 里运行。

```bash
wam prepare fastwam-robotwin --cache-dir /path/to/wam-cache --download --asset eval

wam eval fastwam-robotwin \
  --workload robotwin-single-task \
  --task-name click_alarmclock \
  --num-episodes 1 \
  --cache-dir /path/to/wam-cache
```

### 查看 trace

```bash
ls runs/*/trace.jsonl
```

## 模型库

模型库只列 curated real WAM entries。内置 smoke-test backend 只用于本地检查
harness contract，不作为模型库 entry 展示。

| Model id | 上游资源 | 起步命令 | 当前状态 |
| --- | --- | --- | --- |
| `fastwam-libero` | [![GitHub](https://img.shields.io/badge/GitHub-FastWAM-181717?logo=github)](https://github.com/yuantianyuan01/FastWAM) [![Hugging Face](https://img.shields.io/badge/HF-yuanty%2Ffastwam-FFD21E?logo=huggingface)](https://huggingface.co/yuanty/fastwam) | `wam prepare fastwam-libero --download --asset eval` | 第一个真实模型集成目标。SuperPod H800 上 single-task native eval、serve smoke、reference full-suite eval 和 native full-suite sweep 都已跑通；native sweep 是 9/10，对齐后的 task6 证据是 native 和 reference 都为 4/5。 |
| `fastwam-robotwin` | [![GitHub](https://img.shields.io/badge/GitHub-FastWAM-181717?logo=github)](https://github.com/yuantianyuan01/FastWAM) [![GitHub](https://img.shields.io/badge/GitHub-RoboTwin-181717?logo=github)](https://github.com/RoboTwin-Platform/RoboTwin) [![Hugging Face](https://img.shields.io/badge/HF-yuanty%2Ffastwam-FFD21E?logo=huggingface)](https://huggingface.co/yuanty/fastwam) | `wam prepare fastwam-robotwin --download --asset eval` | 第二个真实 FastWAM simulator 目标。SuperPod H800 上 single-task smoke、serve smoke、reference manager full-suite 和 native full-suite 都已跑通，覆盖 50 个 task x clean/randomized。reference summary 是 100/100 phases，clean mean 1.0、randomized mean 0.84；native summary 是 100/100 phases，结构性失败 0，success rate 0.88。 |
| `cosmos-policy-libero` | [![GitHub](https://img.shields.io/badge/GitHub-Cosmos--Policy-181717?logo=github)](https://github.com/NVlabs/cosmos-policy) [![Hugging Face](https://img.shields.io/badge/HF-Cosmos--Policy--LIBERO-FFD21E?logo=huggingface)](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `wam info cosmos-policy-libero` | native smoke 和官方脚本 parity 集成已开始。 |
| `dreamzero-droid-sim` | [![GitHub](https://img.shields.io/badge/GitHub-DreamZero-181717?logo=github)](https://github.com/dreamzero0/dreamzero) [![Hugging Face](https://img.shields.io/badge/HF-DreamZero--DROID-FFD21E?logo=huggingface)](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) [![Hugging Face](https://img.shields.io/badge/HF-DROID_sim_assets-FFD21E?logo=huggingface)](https://huggingface.co/owhan/DROID-sim-environments) | `wam info dreamzero-droid-sim` | resident policy-server 路径已开始；DROID sim 需要更重的多 GPU runtime。 |

## 加速路线图

EazyWAM 把推理加速做成显式 optimization profile。一个 profile 在进入默认路径前，
必须声明作用范围、参数、trace 字段、输出检查和 rollout 状态。完整 profile 合同见
[`docs/optimization_profiles.md`](docs/optimization_profiles.md)。

| 类别 | 技术 | 当前状态 |
| --- | --- | --- |
| Policy runtime | action chunking、receding horizon、execute horizon、temporal ensemble | runner 已实现 `action_horizon` 和 `replan_steps`；`execute_horizon` 和 `temporal_ensemble` 仍在规划中。 |
| Output control | action-only inference、no future video decode/save、text/goal embedding cache | FastWAM 已使用 action-only native inference，并默认 `return_future=false`；text/goal embedding cache 仍在规划中。 |
| Scheduler / sampler | `num_inference_steps`、自定义 timesteps/sigmas、DPM-Solver++、UniPC、AYS、Karras、FlowMatch schedules | FastWAM 已实现 opt-in FlowMatch Euler scheduler profile，并有 SuperPod 单任务候选配置证据；跨 backend adapter 和其它 solver 仍在规划中。 |
| Precision and attention | bf16/fp16/fp32、TF32、SDPA、FlashAttention、xFormers、SageAttention | bf16 默认值和 PyTorch SDPA 已部分支持；显式 backend selector 仍在规划中。SageAttention 保持 optional。 |
| Native cache and exact runtime | FastWAM `dit_cache` (`video_kv`)、CUDA Graph、torch.compile、warmup/preallocation | FastWAM 已实现 `dit_cache` 和 `cuda_graph(auto)`；CUDA Graph 已有 SuperPod H800 加速证据。`torch_compile` 仍是 experimental，默认关闭。 |
| WAM-specific approximate cache | TeaCache、PAB、FasterCache、cross-chunk cache、step skipping | FastWAM TeaCache L1 已实现为 opt-in 近似 action-denoise step-output cache；PAB 和 FasterCache 仍是 optional benchmark backend，不默认启用。 |
| Throughput and serving | eval sharding、batched action denoise、dynamic batch serving、xDiT/multi-GPU | `wam serve --batch`、remote eval batching 和 FastWAM `infer_batch` 已有 smoke evidence；完整吞吐验收和 xDiT-style multi-GPU 仍在规划中。 |
| Experimental / not default | token merging、AsymRnR、Sparse VideoGen、PTQ methods、FP8 TensorRT | 只作为研究方向跟踪；进入 runtime profile 前需要模型级 success-rate 证据。 |

## 常用命令

```bash
wam --help
wam <command> --help
wam list
wam info <model-id>
wam doctor <model-id>
wam prepare <model-id>
wam run <model-id> --input obs.json --output action.json
wam eval <model-id> --workload <workload>
wam serve <model-id>
wam compare <trace-a> <trace-b>
```

## 开发

core package 开发使用 `uv` 来保持本地检查可复现：

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```

## 文档

- `docs/cli_entrypoints.md` - 命令行为。
- `docs/fastwam_libero_eval_setup.md` - FastWAM 环境和 eval 流程。
- `docs/dependency_isolation.md` - 容器和自管理环境。
- `docs/wamfile.md` - model entry schema。
- `docs/optimization_integration.md` - optimization profile 设计。
- `docs/optimization_profiles.md` - 加速 profile 分类和状态。
- `docs/trace_schema.md` - trace 事件 schema。
- `docs/roadmap.md` - 当前里程碑。

## License

EazyWAM 使用 [MIT License](LICENSE.md)。Vendored 第三方代码和外部模型资产仍遵循
各自上游许可证。
