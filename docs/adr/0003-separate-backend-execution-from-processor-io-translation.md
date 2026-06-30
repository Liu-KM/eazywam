# Separate Backend Execution From Processor I/O Translation

EazyWAM separates backend execution from processor I/O translation. A backend
owns model execution and runtime state: load, warmup, reset, infer, close,
runtime information, and execution-path acceleration methods. A processor owns
semantic translation between EazyWAM observations/results and backend-native
inputs/outputs.

Acceleration methods may modify backend-native execution paths such as
schedulers, caches, CUDA Graph capture, attention kernels, or denoising loops.
The core runner should not expose backend-native tensor layouts, normalization
details, cache hooks, or upstream repository control flow.
