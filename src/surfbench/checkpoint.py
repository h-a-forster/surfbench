"""Resumable unit store: one atomic JSON file per finished unit, manifest and progress log.

Specification: ``docs/DESIGN.md`` section 7.3.

Layout of a run directory::

    <run_dir>/manifest.json        config/filter/snapshot hashes and package version
    <run_dir>/units/<unit_id>.json one file per finished unit (status "ok" or "error")
    <run_dir>/progress.log         one line per unit: UTC time, unit, status, elapsed seconds

Units whose file has ``status == "ok"`` are skipped on rerun. Units recorded as ``"error"``
are retried on the next run, up to ``MAX_ATTEMPTS`` attempts in total.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "MAX_ATTEMPTS",
    "MAX_WORKERS",
    "RunMismatchError",
    "UnitStore",
    "atomic_write_bytes",
    "atomic_write_json",
    "limit_blas_threads",
    "run_units",
    "scrub_paths",
]

MAX_WORKERS = 3
MAX_ATTEMPTS = 3
_BLAS_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_UNIT_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_WIN_PATH = re.compile(r"[A-Za-z]:[\\/][^\s'\"<>|]*")
_POSIX_PATH = re.compile(r"(?<![\w.])/(?:[^\s/'\"<>|]+/)+([^\s/'\"<>|]*)")


class RunMismatchError(RuntimeError):
    """The run directory was created with a different config, filter or snapshot set."""


def scrub_paths(text: str) -> str:
    """Replace absolute file-system paths by ``<path>/basename`` (for logs and unit files)."""

    def _win(m: re.Match[str]) -> str:
        base = re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1]
        return f"<path>/{base}"

    text = _WIN_PATH.sub(_win, text)
    return _POSIX_PATH.sub(lambda m: f"<path>/{m.group(1)}", text)


def limit_blas_threads() -> None:
    """Set the BLAS thread variables to 1 (inherited by worker processes)."""
    for var in _BLAS_VARS:
        os.environ[var] = "1"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


@dataclass
class UnitStore:
    """Checkpoint store for one run directory."""

    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.units_dir = self.run_dir / "units"
        self.manifest_path = self.run_dir / "manifest.json"
        self.log_path = self.run_dir / "progress.log"

    # ----------------------------------------------------------------- manifest
    def open(self, manifest: Mapping[str, Any], *, force_new: bool = False) -> None:
        """Create the run directory or check that it belongs to the same configuration.

        ``manifest`` must be JSON-serialisable. Resuming is refused when any key differs
        from the stored manifest. With ``force_new`` the finished units are always deleted
        and the run starts over, whether or not the manifest changed (an input outside the
        manifest, such as a snapshot file, may have changed).
        """
        current = json.loads(json.dumps(dict(manifest), sort_keys=True))
        if self.manifest_path.exists() and not force_new:
            stored = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if stored != current:
                diff = sorted(
                    key for key in set(stored) | set(current) if stored.get(key) != current.get(key)
                )
                raise RunMismatchError(
                    "run directory was created with a different configuration "
                    f"(changed: {', '.join(diff)}); use --force-new to start over"
                )
        if force_new and (self.manifest_path.exists() or self.units_dir.exists()):
            shutil.rmtree(self.units_dir, ignore_errors=True)
            self.log("-", "reset", 0.0, "--force-new: finished units deleted; started over")
        self.units_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, current)

    # -------------------------------------------------------------------- units
    def path(self, unit_id: str) -> Path:
        if not _UNIT_ID.match(unit_id):
            raise ValueError(f"invalid unit id: {unit_id!r}")
        return self.units_dir / f"{unit_id}.json"

    def read(self, unit_id: str) -> dict[str, Any] | None:
        p = self.path(unit_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write(self, unit_id: str, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.path(unit_id), dict(payload))

    def attempts(self, unit_id: str) -> int:
        rec = self.read(unit_id)
        return int(rec.get("attempts", 0)) if rec else 0

    def is_final(self, unit_id: str) -> bool:
        """True if the unit is done or has exhausted its attempts."""
        rec = self.read(unit_id)
        if rec is None:
            return False
        if rec.get("status") == "ok":
            return True
        return int(rec.get("attempts", 0)) >= MAX_ATTEMPTS

    def load_all(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.units_dir.exists():
            return out
        for p in sorted(self.units_dir.glob("*.json")):
            try:
                out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    # ---------------------------------------------------------------------- log
    def log(self, unit_id: str, status: str, elapsed_s: float, note: str = "") -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"{stamp}\t{unit_id}\t{status}\t{elapsed_s:.2f}s"
        if note:
            line += "\t" + scrub_paths(note).replace("\n", " ")[:300]
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _call(
    fn: Callable[[Any], dict[str, Any]], arg: Any
) -> tuple[dict[str, Any] | None, str, float]:
    t = time.perf_counter()
    try:
        result = fn(arg)
        return result, "", time.perf_counter() - t
    except Exception as exc:  # noqa: BLE001 - recorded in the unit file, retried next run
        tb = traceback.format_exception_only(type(exc), exc)
        return None, scrub_paths("".join(tb).strip()), time.perf_counter() - t


def run_units(
    store: UnitStore,
    units: Iterable[tuple[str, Any]],
    fn: Callable[[Any], dict[str, Any]],
    *,
    workers: int = 1,
    on_done: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """Run ``fn(arg)`` for every unfinished unit and store each result as it completes.

    ``fn`` must be a picklable top-level function returning a JSON-serialisable dict. At
    most ``MAX_WORKERS`` processes are used; ``workers <= 1`` runs in-process. Returns the
    counts of ``ok``, ``error``, ``skipped`` units.
    """
    limit_blas_threads()
    all_units = list(units)
    todo = [(uid, arg) for uid, arg in all_units if not store.is_final(uid)]
    counts = {"ok": 0, "error": 0, "skipped": len(all_units) - len(todo)}

    def finish(uid: str, result: dict[str, Any] | None, err: str, elapsed: float) -> None:
        attempts = store.attempts(uid) + 1
        if result is not None:
            payload = dict(result)
            payload.update(unit_id=uid, status="ok", attempts=attempts, elapsed_s=elapsed)
            counts["ok"] += 1
            store.log(uid, "ok", elapsed)
        else:
            payload = {
                "unit_id": uid,
                "status": "error",
                "attempts": attempts,
                "error": err,
                "elapsed_s": elapsed,
            }
            counts["error"] += 1
            store.log(uid, "error", elapsed, err)
        store.write(uid, payload)
        if on_done is not None:
            on_done(uid, payload)

    n_workers = max(1, min(int(workers), MAX_WORKERS))
    if n_workers == 1 or len(todo) <= 1:
        for uid, arg in todo:
            finish(uid, *_call(fn, arg))
        return counts
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        pending = {pool.submit(_call, fn, arg): uid for uid, arg in todo}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                uid = pending.pop(fut)
                try:
                    finish(uid, *fut.result())
                except Exception as exc:  # noqa: BLE001 - worker crash (e.g. killed process)
                    finish(uid, None, scrub_paths(f"{type(exc).__name__}: {exc}"), 0.0)
    return counts
