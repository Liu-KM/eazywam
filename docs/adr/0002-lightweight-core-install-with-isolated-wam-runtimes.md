# Keep Core Install Lightweight And Isolate Heavy WAM Runtimes

The EazyWAM core package must remain lightweight: `pip install eazywam` should
install the CLI, contracts, registry, model entry parsing, trace, compare,
fake backend, and lightweight orchestration without pulling in every heavy WAM
runtime.

Heavy dependencies for FastWAM, Cosmos-Policy, DreamZero, simulators, CUDA
stacks, and upstream-specific environments belong in backend runtimes,
containers, external checkouts, or explicit prepare/doctor paths. This keeps
the public install usable while still allowing deep model-specific integrations.
