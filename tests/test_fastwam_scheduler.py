import pytest

torch = pytest.importorskip("torch")

from fastwam.models.wan22.schedulers.scheduler_continuous import (  # noqa: E402
    WanContinuousFlowMatchScheduler,
)


def test_flowmatch_scheduler_metadata_records_timesteps_sigmas_and_deltas() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        shift_override=3.0,
    )
    metadata = scheduler.inference_schedule_metadata(
        num_inference_steps=4,
        timesteps=timesteps,
        deltas=deltas,
        shift_override=3.0,
    )

    assert metadata["scheduler_name"] == "fastwam_flowmatch_euler"
    assert metadata["solver"] == "euler"
    assert metadata["schedule_type"] == "shifted_flowmatch"
    assert metadata["schedule_preset"] is None
    assert metadata["schedule_source"] == "generated"
    assert metadata["num_inference_steps"] == 4
    assert metadata["sigma_shift"] == 3.0
    assert metadata["timestep_count"] == 4
    assert metadata["timesteps"]["count"] == 4
    assert metadata["sigmas"]["values"][0] == 1.0
    assert metadata["sigmas"]["values"][-1] > 0.0
    assert metadata["deltas"]["count"] == 4
    assert metadata["deltas"]["values"][-1] < 0.0


def test_flowmatch_scheduler_accepts_custom_sigmas() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        sigmas="1.0,0.5,0.125",
    )
    metadata = scheduler.inference_schedule_metadata(
        num_inference_steps=3,
        timesteps=timesteps,
        deltas=deltas,
        schedule_source="custom_sigmas",
    )

    assert timesteps.tolist() == pytest.approx([1000.0, 500.0, 125.0])
    assert deltas.tolist() == pytest.approx([-0.5, -0.375, -0.125])
    assert metadata["schedule_source"] == "custom_sigmas"
    assert metadata["sigmas"]["values"] == [1.0, 0.5, 0.125]


def test_flowmatch_scheduler_accepts_custom_timesteps() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        timesteps=[1000.0, 500.0, 125.0],
    )

    assert timesteps.tolist() == pytest.approx([1000.0, 500.0, 125.0])
    assert deltas.tolist() == pytest.approx([-0.5, -0.375, -0.125])


def test_flowmatch_scheduler_accepts_karras_preset() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
        schedule_preset="karras",
    )
    metadata = scheduler.inference_schedule_metadata(
        num_inference_steps=5,
        timesteps=timesteps,
        deltas=deltas,
        schedule_source="preset_karras",
        schedule_preset="karras",
    )

    assert metadata["schedule_preset"] == "karras"
    assert metadata["schedule_source"] == "preset_karras"
    assert metadata["sigmas"]["values"][0] == 1.0
    assert metadata["sigmas"]["values"][-1] == pytest.approx(0.001)
    assert all(delta < 0.0 for delta in metadata["deltas"]["values"])


def test_flowmatch_scheduler_accepts_ays_preset() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
        schedule_preset="ays",
    )
    metadata = scheduler.inference_schedule_metadata(
        num_inference_steps=10,
        timesteps=timesteps,
        deltas=deltas,
        schedule_source="preset_ays",
        schedule_preset="ays",
    )

    assert metadata["schedule_preset"] == "ays"
    assert metadata["schedule_source"] == "preset_ays"
    assert metadata["sigmas"]["values"] == [
        1.0,
        0.845,
        0.73,
        0.587,
        0.443,
        0.31,
        0.193,
        0.116,
        0.053,
        0.013,
    ]


def test_flowmatch_scheduler_rejects_ambiguous_or_bad_custom_schedule() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)

    with pytest.raises(ValueError, match="mutually exclusive"):
        scheduler.build_inference_schedule(
            num_inference_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            timesteps=[1000.0, 500.0],
            sigmas=[1.0, 0.5],
        )
    with pytest.raises(ValueError, match="length must match"):
        scheduler.build_inference_schedule(
            num_inference_steps=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sigmas=[1.0, 0.5],
        )
    with pytest.raises(ValueError, match="range"):
        scheduler.build_inference_schedule(
            num_inference_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sigmas=[1.1, 0.5],
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        scheduler.build_inference_schedule(
            num_inference_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            schedule_preset="karras",
            sigmas=[1.0, 0.5],
        )
    with pytest.raises(ValueError, match="schedule_preset"):
        scheduler.build_inference_schedule(
            num_inference_steps=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
            schedule_preset="dpm_solver",
        )
