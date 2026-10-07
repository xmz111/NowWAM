import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as f:
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def contract(cfg, world_size):
    from omegaconf import OmegaConf

    return dict(
        world_size=int(world_size),
        batch_size=int(cfg.batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        lr_scheduler_type=str(cfg.lr_scheduler_type),
        warmup_steps=None if cfg.warmup_steps is None else int(cfg.warmup_steps),
        max_steps=int(cfg.max_steps),
        mixed_precision=str(cfg.mixed_precision),
        model_parameter_dtype=str(cfg.model_parameter_dtype),
        ema_decays=sorted((float(d) for d in cfg.ema_decays)),
        ema_decay=float(cfg.ema_decay),
        ema_update_every=int(cfg.ema_update_every),
        model=OmegaConf.to_container(cfg.model, resolve=True),
        max_grad_norm=float(cfg.max_grad_norm),
        implementation="libero_v1",
    )


def inspect(root, state, expected):
    root, state = (Path(root).resolve(), Path(state).resolve())
    assert state.parent == root / "state", "Checkpoint must belong to this run"
    tag = state.name
    assert tag.startswith("step_") and tag[5:].isdigit()
    step = int(tag[5:])
    progress = json.loads((state / "trainer_state.json").read_text())
    assert progress["global_step"] == step
    assert isinstance(progress["epoch"], int) and isinstance(progress["batch_in_epoch"], int)
    files = [
        state / f
        for f in ("model.safetensors", "optimizer.bin", "scheduler.bin", "trainer_state.json")
    ]
    files += [state / f"random_states_{rank}.pkl" for rank in range(expected["world_size"])]
    weights = root / "weights"
    files += [weights / f"{tag}.pt", weights / f"ema_{tag}.pt"]
    files += [
        weights / f"ema_d{str(d).replace('.', 'p')}_{tag}.pt"
        for d in expected["ema_decays"]
        if d != expected["ema_decay"]
    ]
    scan_file = weights / f"ema_scan_{tag}.json"
    scan = json.loads(scan_file.read_text())
    assert scan == dict(
        version=1,
        step=step,
        decays=expected["ema_decays"],
        primary_decay=expected["ema_decay"],
        update_every=expected["ema_update_every"],
        updates=step // expected["ema_update_every"],
    ), "EMA save contract mismatch"
    files.append(scan_file)
    sizes = {}
    for file in files:
        size = file.stat().st_size
        assert size > 0, f"Empty checkpoint payload: {file}"
        sizes[str(file.relative_to(root))] = size
    return dict(
        version=1,
        step=step,
        state=str(state.relative_to(root)),
        progress=progress,
        contract=expected,
        files=sizes,
    )


def commit(root, state, expected):
    root, state = (Path(root).resolve(), Path(state).resolve())
    manifest = inspect(root, state, expected)
    for relative in manifest["files"]:
        with (root / relative).open("rb") as f:
            os.fsync(f.fileno())
    atomic_json(state / "checkpoint_complete.json", manifest)
    atomic_json(root / "ckptlast.json", manifest)
    return manifest


def validate(root, manifest, expected):
    assert manifest["contract"] == expected, "Resume configuration changed"
    current = inspect(root, Path(root) / manifest["state"], expected)
    assert current == manifest, "Committed checkpoint payload/metadata changed"
    return str(Path(root) / manifest["state"])


def select(root, expected):
    root = Path(root)
    pointer = root / "ckptlast.json"
    if pointer.exists():
        manifest = json.loads(pointer.read_text())
        validate(
            root,
            json.loads((root / manifest["state"] / "checkpoint_complete.json").read_text()),
            expected,
        )
        selected = validate(root, manifest, expected)
        for state in sorted((root / "state").glob("step_*"), reverse=True):
            marker = state / "checkpoint_complete.json"
            if state.name > Path(selected).name and marker.exists():
                newer = json.loads(marker.read_text())
                selected = validate(root, newer, expected)
                atomic_json(pointer, newer)
                break
        return selected
    states = sorted((root / "state").glob("step_*"), reverse=True)
    for state in states:
        marker = state / "checkpoint_complete.json"
        if marker.exists():
            return validate(root, json.loads(marker.read_text()), expected)
    if states:
        raise RuntimeError(
            "Run has checkpoints but no committed ckptlast; inspect/migrate before resuming"
        )
    if any((root / "weights").glob("*.pt")):
        raise RuntimeError("Weights-only run cannot be silently resumed as fresh")
    return None


def migrate(root, expected):
    root = Path(root)
    if (root / "ckptlast.json").exists():
        return select(root, expected)
    candidates = sorted((root / "state").glob("step_*"), reverse=True)
    errors = []
    for state in candidates:
        try:
            inspect(root, state, expected)
        except (AssertionError, OSError, ValueError) as error:
            errors.append((str(state), str(error)))
            continue
        commit(root, state, expected)
        return str(state)
    if candidates:
        raise RuntimeError(f"No complete checkpoint for explicit migration: {errors}")
    return select(root, expected)
