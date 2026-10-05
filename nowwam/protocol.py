"""Case identity and action-noise seeds for the reference evaluations."""

from __future__ import annotations

import hashlib
import json

LIBERO_SUITE_COUNTS = {
    "libero_10": 2519,
    "libero_goal": 2591,
    "libero_spatial": 2402,
    "libero_object": 2518,
}
LIBERO_MANIFEST_SHA256 = "8caa2d7533ab8c4d84ffe0376da8750ce99ead2e46087996301b76a052d2c1ae"


def libero_cases() -> list[dict]:
    """Return the complete, ordered 10,030-case reference manifest."""
    cases = [
        {"suite": suite, "task_idx": i}
        for suite, count in LIBERO_SUITE_COUNTS.items()
        for i in range(count)
    ]
    encoded = json.dumps(cases, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(encoded).hexdigest() != LIBERO_MANIFEST_SHA256:
        raise ValueError("LIBERO-Plus task identity or order changed")
    return cases


def lane_indices(case_count: int, lanes: int, lane: int) -> range:
    """Partition canonical indices without renumbering the cases."""
    if type(case_count) is not int or case_count < 0:
        raise ValueError("case_count must be a nonnegative integer")
    if type(lanes) is not int or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    if type(lane) is not int or not 0 <= lane < lanes:
        raise ValueError("lane must be in [0, lanes)")
    return range(lane, case_count, lanes)


def libero_action_seed(profile: str, canonical_index: int, control_step: int) -> int:
    """Select a documented noise protocol; the caller also selects its RNG.

    zimage_jax: NumPy PCG64, then float32 conversion, as in the corrected
    Z-Image 10,030-case run. The divisor records historical JAX sharding.
    klein_torch_reference: CPU torch.Generator, as in the 87.6471% run;
    it is not the JAX per-case noise protocol.
    """
    if type(canonical_index) is not int or not 0 <= canonical_index < 10030:
        raise ValueError("canonical_index must identify a full-manifest case")
    if type(control_step) is not int or control_step < 0:
        raise ValueError("control_step excludes warmup and must be nonnegative")
    if profile == "zimage_jax":
        return (canonical_index // 32) * 1000 + control_step
    if profile == "klein_torch_reference":
        return control_step
    raise ValueError(f"Unknown action-noise profile: {profile}")


def robocasa_action_seed(task: str, seed: int, control_step: int) -> int:
    if not isinstance(task, str) or not task or "|" in task:
        raise ValueError("task must be a nonempty task name without a separator")
    if type(seed) is not int or not 0 <= seed < 50:
        raise ValueError("RoboCasa reference seeds are 0 through 49")
    if type(control_step) is not int or control_step < 0:
        raise ValueError("control_step must be nonnegative")
    key = f"dial-action-v1|{task}|{seed}|{control_step}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def validate_results(cases: list[str], receipts: list[dict], contract_sha256: str) -> dict:
    """Require one valid outcome per case before computing a micro average.

    This is the release receipt schema, not an implicit adapter for historical
    result JSON. An infrastructure error must not be encoded as success=False.
    """
    if len(contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in contract_sha256):
        raise ValueError("Invalid contract SHA256")
    if not cases or any(not isinstance(case, str) or not case for case in cases):
        raise ValueError("Expected nonempty case IDs")
    expected = set(cases)
    if len(expected) != len(cases):
        raise ValueError("Duplicate expected case IDs")
    seen, successes = set(), 0
    for receipt in receipts:
        case = receipt.get("case_id")
        if not isinstance(case, str) or case not in expected or case in seen:
            raise ValueError(f"Unknown or duplicate receipt case: {case}")
        if receipt.get("contract_sha256") != contract_sha256:
            raise ValueError(f"Mismatched contract for {case}")
        if receipt.get("status") != "completed" or receipt.get("error") is not None:
            raise ValueError(f"Unresolved rollout error for {case}")
        if type(receipt.get("success")) is not bool:
            raise ValueError(f"Expected boolean success for {case}")
        seen.add(case)
        successes += int(receipt["success"])
    if seen != expected:
        raise ValueError(f"Incomplete evaluation: {len(seen)}/{len(expected)}")
    return {
        "successes": successes,
        "episodes": len(expected),
        "success_rate": successes / len(expected),
    }
