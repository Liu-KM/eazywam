import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_backend_dockerfiles_install_eazywam_cli() -> None:
    for relative in [
        "containers/fastwam/Dockerfile",
        "containers/cosmos-policy/Dockerfile",
        "containers/dreamzero/Dockerfile",
    ]:
        content = (ROOT / relative).read_text(encoding="utf-8")

        assert "COPY pyproject.toml README.md uv.lock ./" in content
        assert "COPY src ./src" in content
        assert "/workspace/eazywam" in content
        assert "wam" in content


def test_core_dockerfile_uses_wam_cli_as_default_command() -> None:
    content = (ROOT / "containers/core/Dockerfile").read_text(encoding="utf-8")

    assert 'PATH="/workspace/eazywam/.venv/bin:${PATH}"' in content
    assert 'CMD ["wam", "--help"]' in content


def test_fastwam_native_setup_script_defines_self_managed_environment() -> None:
    script_path = ROOT / "scripts/setup_fastwam_native_env.sh"
    content = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert "Optional FastWAM checkout path for reference eval/debug" in content
    assert "--upstream-dir is required" not in content
    assert "--venv" in content
    assert "--clone" in content
    assert "--torch-backend" in content
    assert "uv venv" in content
    assert "--allow-existing" in content
    assert "uv pip install --python \"$python_bin\" \\" in content
    assert "torch==2.7.1" in content
    assert "transformers==4.49.0" in content
    assert "-e \"$upstream_dir\"" not in content
    assert "WAM_FASTWAM_TORCH_BACKEND" in content
    assert "LIBERO_CONFIG_PATH" in content
    assert "WAM_LIBERO_DIR" in content
    assert "fastwam-libero-eval.sh" in content
    assert "wam_fastwam_libero.pth" in content
    assert 'pth_path.write_text(f"{repo_root}\\n", encoding="utf-8")' in content
    assert '[[ -d "$libero_dir/libero/libero" ]]' in content
    assert "compat_package = libero_package / \"libero\"" not in content
    assert "import libero" in content
    assert "wam doctor fastwam-libero" in content
    assert "wam prepare fastwam-libero" in content
    assert "bddl==1.0.1" in content
    assert "robosuite==1.4.0" in content
    assert "macros_private.py" in content
    assert "shutil.copyfile(macros, macros_private)" in content
    assert "mujoco==3.3.2" in content
    assert "FastWAM env smoke: mujoco warning" in content
    assert "except Exception as exc:" in content
    assert "numpy==1.26.4" in content
    assert "--extra-index-url https://download.pytorch.org/whl/cu128" not in content
    assert "sbatch" not in content
    assert "srun" not in content


def test_fastwam_dockerfile_reuses_native_setup_script() -> None:
    content = (ROOT / "containers/fastwam/Dockerfile").read_text(encoding="utf-8")

    assert "ENV WAM_FASTWAM_VENV=/opt/wam-fastwam-venv" in content
    assert "ENV WAM_CACHE_DIR=/mnt/wam-cache" in content
    assert "ENV WAM_TRACE_DIR=/mnt/runs" in content
    assert "ENV WAM_LIBERO_DIR=/opt/LIBERO" in content
    assert "ENV LIBERO_CONFIG_PATH=/mnt/wam-cache/libero/config" in content
    assert "ENV MUJOCO_GL=egl" in content
    assert "ENV PYOPENGL_PLATFORM=egl" in content
    assert "ENV WAM_FASTWAM_REPO" not in content
    assert "COPY scripts/setup_fastwam_native_env.sh" in content
    assert "COPY scripts/fastwam-teacache-l1-superpod.sh" in content
    assert "COPY scripts/fastwam-teacache-l1-report.py" in content
    assert "./scripts/setup_fastwam_native_env.sh" in content
    assert "./scripts/fastwam-teacache-l1-superpod.sh" in content
    assert "./scripts/fastwam-teacache-l1-report.py" in content
    assert "wam-fastwam-teacache-l1-superpod" in content
    assert "wam-fastwam-teacache-l1-report" in content
    assert "--harness-dir /workspace/eazywam" in content
    assert "--clone" in content


def test_fastwam_libero_eval_acceptance_script_checks_simulator_env() -> None:
    script_path = ROOT / "scripts/fastwam-libero-eval.sh"
    content = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert "--skip-simulator-check" in content
    assert "--min-success-rate" in content
    assert "--libero-dir" in content
    assert "WAM_LIBERO_DIR" in content
    assert 'min_success_rate="${WAM_ACCEPT_MIN_SUCCESS_RATE:-1.0}"' in content
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in content
    assert 'elif [[ -d /opt/LIBERO/libero/libero ]]; then' in content
    assert 'assets: $libero_dir/libero/libero/assets' in content
    assert "_import_libero_modules" in content
    assert "_load_task" in content
    assert "_create_libero_env" in content
    assert "_observation_from_libero" in content
    assert "env.set_init_state(states[0])" in content
    assert "absolute_path()" in content
    assert 'cache_dir="$(absolute_path "$cache_dir")"' in content
    assert 'trace_dir="$(absolute_path "$trace_dir")"' in content
    assert "wam native-smoke" in content
    assert 'print_cmd wam "${eval_args[@]}"' in content
    assert '--set "mujoco_gl=$mujoco_gl"' in content
    assert '--set "pyopengl_platform=$pyopengl_platform"' in content
    assert 'eval_raw_output_path="$trace_dir/${model_id}-${workload}-eval-output.txt"' in content
    assert 'eval_args+=(--summary-path "$eval_summary_path")' in content
    assert 'wam "${eval_args[@]}" | tee "$eval_raw_output_path"' in content
    assert "raw_decode" not in content
    assert "eval_summary_path=" in content
    assert "acceptance_report_path=" in content
    assert '${model_id}-${workload}-acceptance.json' in content
    assert "cat \"$eval_summary_path\"" in content
    assert "tee \"$acceptance_report_path\"" in content
    assert "python -m eazywam.evals.acceptance" in content
    assert "python -m eazywam.evals.acceptance --json" in content
    assert '"$min_success_rate"' in content


def test_fastwam_teacache_l1_superpod_script_is_scheduler_agnostic() -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-superpod.sh"
    content = script_path.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    assert "--execute" in content
    assert "--cache-dir" in content
    assert "--robotwin-root" in content
    assert "--run-id" in content
    assert 'trace_run_root="$trace_root/$run_id"' in content
    assert 'report_run_root="$report_root/$run_id"' in content
    assert 'cache_args=(--cache-dir "$cache_dir")' in content
    assert 'upstream_args=(--upstream-dir "$upstream_dir")' in content
    assert 'robotwin_root_args=(--set "robotwin_root=$robotwin_root")' in content
    assert "Slurm/PBS/LSF" in content
    assert "sbatch" not in content
    assert "wam eval fastwam-libero" in content
    assert "wam eval fastwam-robotwin" in content
    assert "--opt teacache" in content
    assert "--set dit_cache_mode=video_kv" in content
    assert "--set cuda_graph_mode=off" in content
    assert "teacache_threshold" in content
    assert "teacache_warmup_steps" in content
    assert "wam compare" in content
    assert "latest_trace" in content
    assert "run_report" in content
    assert "fastwam-teacache-l1-report.md" in content
    assert "fastwam-teacache-l1-report.json" in content
    assert "fastwam-libero-teacache-compare.json" in content
    assert "fastwam-robotwin-teacache-compare.json" in content


def test_fastwam_teacache_l1_superpod_script_passes_cache_dir() -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-superpod.sh"

    result = subprocess.run(
        [
            str(script_path),
            "--cache-dir",
            "/mnt/wam-cache",
            "--run-id",
            "test-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.count("--cache-dir /mnt/wam-cache") == 4
    assert "runs/fastwam-teacache-l1/test-run/fastwam-libero-eager-cache" in result.stdout
    assert "runs/fastwam-teacache-l1-reports/test-run/fastwam-libero-eager-cache-summary.json" in result.stdout
    assert "runs/fastwam-teacache-l1-reports/test-run/fastwam-teacache-l1-report.md" in result.stdout
    assert "runs/fastwam-teacache-l1-reports/test-run/fastwam-teacache-l1-report.json" in result.stdout
    assert "+ wam compare" in result.stdout
    assert "+ " in result.stdout
    assert "fastwam-teacache-l1-report.py --report-root" in result.stdout
    assert "fastwam-teacache-l1-report.py --report-root runs/fastwam-teacache-l1-reports/test-run --json" in result.stdout


def test_fastwam_teacache_l1_superpod_script_passes_upstream_dir() -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-superpod.sh"

    result = subprocess.run(
        [
            str(script_path),
            "--upstream-dir",
            "/mnt/FastWAM",
            "--run-id",
            "test-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.count("--upstream-dir /mnt/FastWAM") == 4


def test_fastwam_teacache_l1_superpod_script_passes_robotwin_root() -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-superpod.sh"

    result = subprocess.run(
        [
            str(script_path),
            "--robotwin-root",
            "/mnt/RoboTwin",
            "--run-id",
            "test-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.count("--set robotwin_root=/mnt/RoboTwin") == 2
    assert "fastwam-libero-eager-cache" in result.stdout
    assert "fastwam-robotwin-eager-cache" in result.stdout


def test_fastwam_teacache_l1_superpod_script_resolves_symlinked_report_helper(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-superpod.sh"
    link_path = tmp_path / "wam-fastwam-teacache-l1-superpod"
    link_path.symlink_to(script_path)

    result = subprocess.run(
        [
            str(link_path),
            "--run-id",
            "symlink-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        f"+ {ROOT / 'scripts/fastwam-teacache-l1-report.py'} --report-root "
        "runs/fastwam-teacache-l1-reports/symlink-run"
    ) in result.stdout
    assert (
        f"+ {ROOT / 'scripts/fastwam-teacache-l1-report.py'} --report-root "
        "runs/fastwam-teacache-l1-reports/symlink-run --json"
    ) in result.stdout
    assert f"{tmp_path}/fastwam-teacache-l1-report.py" not in result.stdout


def test_fastwam_teacache_l1_report_script_extracts_markdown_rows(tmp_path) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"mean": 0.0004, "max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    libero_baseline = {"metrics": {"success_rate": 0.8}}
    libero_variant = {"metrics": {"success_rate": 1.0}}
    robotwin_baseline = {
        "metrics": {
            "overall": {
                "clean_mean_success_rate": 0.7,
                "random_mean_success_rate": 0.6,
            }
        },
    }
    robotwin_variant = {
        "metrics": {
            "overall": {
                "clean_mean_success_rate": 0.9,
                "random_mean_success_rate": 0.8,
            }
        },
    }
    (report_root / "fastwam-libero-teacache-compare.json").write_text(
        json.dumps(compare_payload),
        encoding="utf-8",
    )
    (report_root / "fastwam-robotwin-teacache-compare.json").write_text(
        json.dumps(compare_payload),
        encoding="utf-8",
    )
    (report_root / "fastwam-libero-eager-cache-summary.json").write_text(
        json.dumps(libero_baseline),
        encoding="utf-8",
    )
    (report_root / "fastwam-libero-teacache-summary.json").write_text(
        json.dumps(libero_variant),
        encoding="utf-8",
    )
    (report_root / "fastwam-robotwin-eager-cache-summary.json").write_text(
        json.dumps(robotwin_baseline),
        encoding="utf-8",
    )
    (report_root / "fastwam-robotwin-teacache-summary.json").write_text(
        json.dumps(robotwin_variant),
        encoding="utf-8",
    )

    subprocess.run(["python", "-m", "py_compile", str(script_path)], check=True)
    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "| LIBERO | 1.5 | 2 | 0.6 | 6 | 0.02 | 0.0008 | 0.8 | 1 | TODO |" in result.stdout
    assert "| RoboTwin | 1.5 | 2 | 0.6 | 6 | 0.02 | 0.0008 | 0.6 | 0.8 | TODO |" in result.stdout
    assert "speedup blocked:" not in result.stdout


def test_fastwam_teacache_l1_report_script_hides_invalid_speedups(tmp_path) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "invalid",
        "output_gate_passed": False,
        "output_gate_details": {"observed": {"max": 0.01}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["hit_rate"] == 0.6
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == [
        "compare_decision:invalid",
        "output_gate_not_passed",
    ]

    markdown = subprocess.run(
        [str(script_path), "--report-root", str(report_root)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (
        "- LIBERO speedup blocked: compare_decision:invalid, output_gate_not_passed"
        in markdown.stdout
    )


def test_fastwam_teacache_l1_report_script_hides_unavailable_gate_speedups(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "not_comparable",
        "output_gate_passed": None,
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["compare_decision"] == "not_comparable"
    assert rows[0]["output_gate_passed"] is None
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == [
        "compare_decision:not_comparable",
        "output_gate_not_passed",
        "action_drift_missing",
    ]


def test_fastwam_teacache_l1_report_script_hides_non_faster_decisions(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "same",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.01},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 1.02},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.01
    assert rows[0]["denoise_mean_speedup"] == 1.02
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == ["compare_decision:same"]


def test_fastwam_teacache_l1_report_script_hides_fallback_speedups(tmp_path) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
            "backend_metadata_values": {
                "teacache_fallback_reason": {
                    "values": ["requires_video_kv_cache"],
                },
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["fallback_reason"] == "requires_video_kv_cache"
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == ["teacache_fallback_reason"]


def test_fastwam_teacache_l1_report_script_hides_speedups_without_success_rate(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"total_episodes": 5}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["success_rate_baseline"] is None
    assert rows[0]["success_rate_teacache"] is None
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == [
        "baseline_success_rate_missing",
        "teacache_success_rate_missing",
    ]


def test_fastwam_teacache_l1_report_script_hides_speedups_when_success_rate_drops(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    baseline_summary = {"metrics": {"success_rate": 0.9}}
    variant_summary = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(baseline_summary), encoding="utf-8")
    for name in (
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(variant_summary), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["success_rate_baseline"] == 0.9
    assert rows[0]["success_rate_teacache"] == 0.8
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == ["success_rate_drop"]


def test_fastwam_teacache_l1_report_script_hides_profile_mismatch_speedups(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": ["teacache"]},
        "variant": {
            "profiles": [],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["speedup_blockers"] == [
        "baseline_has_teacache_profile",
        "variant_missing_teacache_profile",
    ]


def test_fastwam_teacache_l1_report_script_hides_missing_profile_speedups(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {},
        "variant": {
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["speedup_blockers"] == [
        "baseline_profiles_missing",
        "variant_profiles_missing",
    ]


def test_fastwam_teacache_l1_report_script_hides_missing_telemetry_speedups(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "output_gate_details": {"observed": {"max": 0.0008}},
        "metric_comparisons": {
            "latency_ms.mean": {"speedup": 1.5},
            "backend_metadata.denoise_wall_ms.mean": {"speedup": 2.0},
        },
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] == 1.5
    assert rows[0]["denoise_mean_speedup"] == 2.0
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["hit_rate"] == 0.6
    assert rows[0]["skipped_steps"] is None
    assert rows[0]["drift_score"] is None
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == [
        "teacache_skipped_steps_missing",
        "teacache_drift_score_missing",
    ]


def test_fastwam_teacache_l1_report_script_hides_missing_result_fields_speedups(
    tmp_path,
) -> None:
    script_path = ROOT / "scripts/fastwam-teacache-l1-report.py"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    compare_payload = {
        "decision": "faster",
        "output_gate_passed": True,
        "metric_comparisons": {},
        "baseline": {"profiles": []},
        "variant": {
            "profiles": ["teacache"],
            "backend_metadata": {
                "teacache_hit_rate": {"mean": 0.6},
                "teacache_skipped_steps": {"mean": 6.0},
                "teacache_drift_score": {"mean": 0.02},
            },
        },
    }
    summary_payload = {"metrics": {"success_rate": 0.8}}
    for name in (
        "fastwam-libero-teacache-compare.json",
        "fastwam-robotwin-teacache-compare.json",
    ):
        (report_root / name).write_text(json.dumps(compare_payload), encoding="utf-8")
    for name in (
        "fastwam-libero-eager-cache-summary.json",
        "fastwam-libero-teacache-summary.json",
        "fastwam-robotwin-eager-cache-summary.json",
        "fastwam-robotwin-teacache-summary.json",
    ):
        (report_root / name).write_text(json.dumps(summary_payload), encoding="utf-8")

    result = subprocess.run(
        [str(script_path), "--report-root", str(report_root), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(result.stdout)

    assert rows[0]["latency_mean_speedup"] is None
    assert rows[0]["denoise_mean_speedup"] is None
    assert rows[0]["latency_mean_speedup_reportable"] is None
    assert rows[0]["denoise_mean_speedup_reportable"] is None
    assert rows[0]["action_drift"] is None
    assert rows[0]["speedup_reportable"] is False
    assert rows[0]["speedup_blockers"] == [
        "latency_mean_speedup_missing",
        "denoise_mean_speedup_missing",
        "action_drift_missing",
    ]


def test_backend_native_smoke_scripts_define_container_contract() -> None:
    scripts = {
        "fastwam": ("fastwam-libero", None, "wam-fastwam-native-smoke"),
        "cosmos-policy": (
            "cosmos-policy-libero",
            "/opt/cosmos-policy",
            "wam-cosmos-policy-native-smoke",
        ),
        "dreamzero": ("dreamzero-droid-sim", "/opt/dreamzero", "wam-dreamzero-native-smoke"),
    }

    for backend, (model_id, upstream_dir, command_name) in scripts.items():
        script_path = ROOT / "containers" / backend / "native-smoke.sh"
        content = script_path.read_text(encoding="utf-8")

        subprocess.run(["bash", "-n", str(script_path)], check=True)
        assert f"WAM_MODEL_ID:-{model_id}" in content
        if upstream_dir is None:
            assert "WAM_UPSTREAM_DIR" not in content
            assert "--upstream-dir" not in content
        else:
            assert f"WAM_UPSTREAM_DIR:-{upstream_dir}" in content
        assert "wam prepare" in content
        assert "prepare_status=0" in content
        assert "running wam doctor for native readiness" in content
        assert "wam doctor" in content
        assert "--json --strict" in content
        assert "wam native-smoke" in content
        assert "--require-ready" in content

        dockerfile = (ROOT / "containers" / backend / "Dockerfile").read_text(
            encoding="utf-8"
        )
        assert f"COPY containers/{backend}/native-smoke.sh /usr/local/bin/{command_name}" in dockerfile
        assert f"chmod +x /usr/local/bin/{command_name}" in dockerfile
