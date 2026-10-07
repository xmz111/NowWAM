from __future__ import annotations
import ctypes
import gc
import os
from typing import Optional

M_TRIM_THRESHOLD = -1
M_TOP_PAD = -2
M_MMAP_THRESHOLD = -3
M_MMAP_MAX = -4
M_ARENA_TEST = -7
M_ARENA_MAX = -8
_libc: Optional[ctypes.CDLL] = None
_libc_resolved = False


def _get_libc() -> Optional[ctypes.CDLL]:
    global _libc, _libc_resolved
    if _libc_resolved:
        return _libc
    _libc_resolved = True
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    except OSError:
        _libc = None
    return _libc


def malloc_trim(pad: int = 0) -> bool:
    libc = _get_libc()
    if libc is None:
        return False
    try:
        return bool(libc.malloc_trim(pad))
    except (AttributeError, OSError):
        return False


def mallopt(option: int, value: int) -> bool:
    libc = _get_libc()
    if libc is None:
        return False
    try:
        return bool(libc.mallopt(option, value))
    except (AttributeError, OSError):
        return False


def trim_now(do_gc: bool = True) -> None:
    if do_gc:
        gc.collect()
    malloc_trim(0)


def apply_glibc_tuning_from_env(verbose: bool = False) -> dict:
    spec = [
        ("IMAGEWAM_M_TRIM_THRESHOLD", M_TRIM_THRESHOLD, "M_TRIM_THRESHOLD"),
        ("IMAGEWAM_M_MMAP_THRESHOLD", M_MMAP_THRESHOLD, "M_MMAP_THRESHOLD"),
        ("IMAGEWAM_M_TOP_PAD", M_TOP_PAD, "M_TOP_PAD"),
        ("IMAGEWAM_M_ARENA_MAX", M_ARENA_MAX, "M_ARENA_MAX"),
    ]
    applied = {}
    for env, opt, name in spec:
        raw = os.environ.get(env)
        if raw is None or raw == "":
            continue
        try:
            val = int(raw)
        except ValueError:
            if verbose:
                print(f"[mem_tools] ignored {env}={raw!r} (not int)")
            continue
        ok = mallopt(opt, val)
        applied[name] = ok
        if verbose:
            print(f"[mem_tools] mallopt({name}, {val}) -> {ok}")
    return applied


class PeriodicTrim:
    __slots__ = ("every", "_n", "do_gc")

    def __init__(self, every: int = 0, do_gc: bool = True) -> None:
        self.every = int(every)
        self._n = 0
        self.do_gc = bool(do_gc)

    def tick(self) -> bool:
        if self.every <= 0:
            return False
        self._n += 1
        if self._n % self.every == 0:
            trim_now(do_gc=self.do_gc)
            return True
        return False
