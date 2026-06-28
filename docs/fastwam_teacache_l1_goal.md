# FastWAM TeaCache L1 Goal

这份文档是给新 worktree / 新 agent 的中文 handoff。目标不是一次性实现所有
video DiT cache 方法，而是先把 FastWAM 的 TeaCache L1 做成一个可开关、可追踪、
可实验验证的近似加速 profile。

## 可直接交给 Codex 的 Prompt

```text
你在 /home/liu/wam-harness 的新 worktree 中工作。请先阅读 AGENTS.md、
docs/optimization_profiles.md、docs/optimization_integration.md、
docs/trace_schema.md、docs/backends.md，以及本文档。

目标：实现 FastWAM TeaCache L1。

背景：
FastWAM 现在已有 dit_cache(video_kv) 和 cuda_graph(auto)。dit_cache(video_kv)
是精确的 request-local video K/V cache；CUDA Graph 是系统层加速。现在要做的
TeaCache 属于 feature_cache：它是训练免费的近似加速方法，目标是在 FastWAM 的
action denoise 循环里复用部分 DiT/MoT 中间结果或 step output，让模型少算。

总体原则：
1. 不要把 FastWAM block、attention、tensor layout 写进 core runner。
2. core harness 只负责 profile、CLI/runtime option、trace 字段和文档。
3. 真正的 TeaCache 逻辑放在 FastWAM native model 层。
4. 第一版只支持 FastWAM action-only inference。
5. 第一版只做 request-local cache，不跨 replan、episode 或 server 请求复用。
6. 第一版只在 dit_cache.mode=video_kv 的路径上启用。
7. 第一版默认关闭，必须通过 --opt teacache 或 runtime option 显式启用。
8. 第一版不要默认和 CUDA Graph 组合；先让 teacache + cuda_graph_mode=off 跑通。
9. 不做 PAB、FasterCache、token pruning、step skipping、cross-replan cache。
10. 不要宣称 native/reference parity；这是近似加速，只能报告 speedup、action drift
    和 success rate。

实现范围：

第一层：profile / manifest / registry
- 确认 teacache 是独立 optimization profile，family=feature_cache。
- FastWAM manifest 可以声明支持 teacache，但默认 disabled。
- teacache 参数保持少量即可：
  - mode: off / auto
  - threshold: 可选 float
  - warmup_steps: 可选 int
  - layers: 先保留为可选字段，不要求 L1 做 layer-level cache
- 不要改变 dit_cache(video_kv) 的语义。

第二层：FastWAM backend adapter
- 在 src/eazywam/backends/fastwam.py 中把 teacache profile/runtime options 转成
  model.infer_action(...) 的 kwargs。
- 只有当模型 infer_action 签名支持对应参数时才传入，沿用现有 cuda_graph /
  torch_compile 的兼容风格。
- backend metadata 要透传 native model 返回的 teacache_* 字段。
- batch 路径已从产品主路径暂缓；TeaCache L1 只覆盖单请求 FastWAM inference，
  不再把批处理入口作为当前实现目标或新的 fallback 合同。

第三层：FastWAM native model
- 在 src/fastwam/models/wan22/fastwam.py 的 infer_action action denoise 循环中接入
  TeaCache。
- 最小 L1 目标是围绕 _predict_action_noise_step(...) 做 request-local
  timestep-aware cache：
  - 正常计算若干 warmup steps。
  - 每个 denoise step 计算一个便宜 drift score，用来判断当前 step 与上一次可复用
    step 是否足够接近。
  - 如果 drift score 低于 threshold，复用上一次缓存的 expensive output，跳过
    这一步昂贵的 MoT/DiT action body 计算。
  - 如果 drift score 高于 threshold，正常调用 _predict_action_noise_step(...)，
    并更新缓存。
  - 无论是否复用，都继续走 scheduler.step(...)，保证 action latent 更新路径清楚。
- L1 可以先做 step-output cache；不要声称已经实现 layer-level TeaCache。
- 代码要保持简单，不要加入过度防御、复杂抽象或宽泛异常吞掉错误。

第四层：trace / metadata
- native metadata 至少包含：
  - teacache_enabled
  - teacache_mode
  - teacache_threshold
  - teacache_warmup_steps
  - teacache_hit_rate
  - teacache_skipped_steps
  - teacache_drift_score
  - teacache_fallback_reason
- 如果 disabled，也要能看出 disabled，而不是字段缺失导致无法比较。
- 保留现有 denoise_wall_ms、total_ms、dit_cache_*、cuda_graph_* 字段。

第五层：测试
- 添加或更新 registry / manifest 测试，证明 teacache profile 能构建但默认关闭。
- 添加 FastWAM backend adapter 测试，证明 --opt teacache 会传递 teacache kwargs。
- 添加 native model 层的轻量单元测试。不要要求真实 checkpoint；可以用小型 fake /
  monkeypatch 证明：
  - teacache off 时每个 denoise step 都调用 expensive path。
  - teacache on 且 drift 低时会减少 expensive path 调用。
  - metadata 中 hit_rate / skipped_steps / fallback_reason 合理。
- 跑：
  uv run pytest -q tests/test_registry.py tests/test_fastwam_native.py
  uv run ruff check .
- 如果改动影响更广，再跑 uv run pytest -q。

第六层：SuperPod 实验
本机只是开发工作站，不是实际跑大模型实验的机器。真实 FastWAM checkpoint /
LIBERO / RoboTwin 实验要去 SuperPod 跑。

至少做两组对比：
1. FastWAM LIBERO:
   - baseline: dit_cache(video_kv), cuda_graph_mode=off, teacache off
   - candidate: dit_cache(video_kv), cuda_graph_mode=off, teacache on
   - 固定 task、seed、num_trials、num_inference_steps。
2. FastWAM RoboTwin:
   - baseline 同上
   - candidate 同上
   - 固定 task、episode、num_inference_steps。

记录：
- total_ms / denoise_wall_ms speedup
- teacache_hit_rate
- teacache_skipped_steps
- teacache_drift_score
- success rate / task success 是否下降
- 任何 fallback reason

完成标准：
- teacache 是显式 profile，默认关闭。
- FastWAM action denoise 路径真的能少调用 expensive MoT/DiT path。
- trace 能解释这次到底省了多少 step、用了什么 threshold、有没有 fallback。
- local tests 和 ruff 通过。
- SuperPod 至少有 LIBERO 和 RoboTwin 的 baseline vs teacache 对比结果。
- 文档清楚说明：TeaCache 是近似 feature_cache，不是 dit_cache(video_kv)，也不是
  native/reference parity。
```

## 新 Worktree 建议

```bash
git fetch origin
git worktree add ../wam-harness-teacache -b feat/fastwam-teacache origin/main
cd ../wam-harness-teacache
```

建议每个阶段小提交：

```text
1. profile/manifest/trace contract
2. FastWAM backend adapter plumbing
3. native model teacache L1 implementation
4. tests and docs
5. SuperPod experiment report
```

## 高层设计说明

FastWAM 当前 action 推理的主要流程是：

```text
observation
-> encode image / context
-> prefill video_kv cache
-> action denoise step 1
-> action denoise step 2
-> ...
-> action denoise step N
-> action chunk
```

现有 `dit_cache(video_kv)` 复用的是 observation/video 条件。TeaCache L1 要尝试复用
denoise steps 之间变化不大的 action-side 计算结果。

第一版最重要的边界是：只在一次 `infer_action()` 请求内部复用，不跨 replan。这样风险
可控，也容易和 baseline 做一一对比。

## 不要做的事情

- 不要把 TeaCache 合并进 `dit_cache`。
- 不要让 TeaCache 默认开启。
- 不要把 core runner 写成知道 FastWAM MoT block 的样子。
- 不要第一版就做 PAB / FasterCache / token pruning。
- 不要第一版就追求 CUDA Graph + TeaCache 组合。
- 不要只报告 speedup，不报告 action drift 和 success rate。
- 不要在本机假装完成大模型实验；真实实验去 SuperPod。

## 相关文件

- `docs/optimization_profiles.md`
- `docs/optimization_integration.md`
- `docs/trace_schema.md`
- `src/eazywam/defaults.py`
- `src/eazywam/backends/fastwam.py`
- `src/eazywam/manifests/fastwam-libero.yaml`
- `src/eazywam/manifests/fastwam-robotwin.yaml`
- `src/fastwam/models/wan22/fastwam.py`
- `tests/test_registry.py`
- `tests/test_fastwam_native.py`

## 参考

- TeaCache: https://github.com/ali-vilab/TeaCache
- PAB / VideoSys: https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/master/docs/pab.md
- FasterCache: https://github.com/Vchitect/FasterCache
