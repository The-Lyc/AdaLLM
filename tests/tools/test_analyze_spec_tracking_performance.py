# SPDX-License-Identifier: Apache-2.0

from tools.analyze_spec_tracking_performance import (
    build_tracking_episodes,
    summarize_tracking,
)


def _row(
    layer: int,
    state: str,
    imbalance_ms: float,
    *,
    step: int = 7,
    decode_only: bool = True,
) -> dict:
    return {
        "scenario": "test",
        "workload": "1k",
        "variant": "full",
        "speculative_enabled": True,
        "step": step,
        "layer_idx": layer,
        "decode_only": decode_only,
        "imbalance_ms": imbalance_ms,
        "sm_state": state,
    }


def test_tracking_episode_counts_first_deadband_and_settle_exit() -> None:
    rows = [
        _row(1, "transit", 8.0),
        _row(2, "tracking", 6.0),
        _row(3, "tracking", 2.0),
        _row(4, "tracking", 0.1),
        _row(5, "tracking", 0.2),
        _row(6, "steady", 0.1),
        _row(7, "steady", 0.0),
    ]

    episodes = build_tracking_episodes(rows, deadband_ms=0.5, settle_layers=3)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode["tracking_state_observations"] == 4
    assert episode["episode_observations_including_exit"] == 5
    assert episode["observations_to_first_deadband"] == 3
    assert episode["outside_deadband_observations_before_first_hit"] == 2
    assert episode["max_consecutive_deadband_observations"] == 3
    assert episode["exit_reason"] == "settled_via_deadband"
    assert episode["settled_via_deadband"] is True
    assert episode["right_censored"] is False


def test_tracking_summary_exposes_timeout_and_censoring() -> None:
    rows = [
        _row(1, "steady", 0.0),
        _row(2, "tracking", 4.0),
        _row(3, "tracking", 3.0),
        _row(4, "steady", 2.0),
        _row(5, "tracking", 4.0),
        _row(6, "tracking", 0.1),
    ]
    episodes = build_tracking_episodes(rows, deadband_ms=0.5, settle_layers=3)
    summary = summarize_tracking(
        episodes,
        rows,
        scenario="test",
        workload="1k",
        variant="full",
        phase_scope="decode_only",
    )

    assert len(episodes) == 2
    assert episodes[0]["exit_reason"] == "timeout_or_other"
    assert episodes[1]["exit_reason"] == "right_censored"
    assert summary["controller_observations"] == 6
    assert summary["tracking_state_observations"] == 4
    assert summary["episode_count"] == 2
    assert summary["complete_episode_count"] == 1
    assert summary["deadband_reached_episode_count"] == 1
    assert summary["mean_observations_to_first_deadband"] == 2.0
    assert summary["timeout_or_other_episode_count"] == 1
    assert summary["right_censored_episode_count"] == 1
