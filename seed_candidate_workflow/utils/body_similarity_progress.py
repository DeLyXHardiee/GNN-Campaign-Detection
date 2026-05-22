"""
Line-based progress reporting for body-similarity candidate generation.

Uses ``tqdm.write`` so messages do not fight nested tqdm bars from other pipeline stages.
"""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _emit(line: str) -> None:
    if _tqdm is not None:
        _tqdm.write(line, file=sys.stderr)
    else:
        print(line, file=sys.stderr, flush=True)


class BodySimilarityProgress:
    """Periodic single-line status updates (no nested progress bars)."""

    def __init__(
        self,
        enabled: bool,
        *,
        graph_id: str = "",
        log_interval_seconds: float = 10.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.graph_id = str(graph_id or "").strip()
        self.log_interval_seconds = max(1.0, float(log_interval_seconds))
        self._phase_t0 = 0.0
        self._loop_t0 = 0.0
        self._loop_last_log = 0.0
        self._loop_done = 0
        self._loop_total = 0
        self._loop_label = ""
        self._last_pct_bucket = -1

    def _prefix(self) -> str:
        return f"[body_jaccard/{self.graph_id}]" if self.graph_id else "[body_jaccard]"

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        _emit(f"{self._prefix()} {text}")

    def phase_start(self, phase: str, *, detail: str = "") -> None:
        self._phase_t0 = time.perf_counter()
        extra = f" ({detail})" if detail else ""
        self.message(f">> {phase}{extra}")

    def phase_done(self, phase: str, **stats: Any) -> None:
        elapsed = time.perf_counter() - self._phase_t0
        parts: list[str] = []
        for key, val in stats.items():
            if isinstance(val, float):
                parts.append(f"{key}={val:.2f}")
            elif isinstance(val, int):
                parts.append(f"{key}={val:,}")
            else:
                parts.append(f"{key}={val}")
        suffix = f" [{', '.join(parts)}]" if parts else ""
        self.message(f"<< {phase} done in {elapsed:.1f}s{suffix}")

    def loop_start(self, label: str, total: int) -> None:
        self._loop_label = label
        self._loop_total = max(int(total), 0)
        self._loop_done = 0
        now = time.perf_counter()
        self._loop_t0 = now
        self._loop_last_log = now
        self._last_pct_bucket = -1
        if self._loop_total > 0:
            self.message(f"   {label}: starting ({self._loop_total:,} steps)")

    def loop_tick(self, n: int = 1, **extra: Any) -> None:
        if not self.enabled or self._loop_total <= 0:
            return
        self._loop_done += int(n)
        now = time.perf_counter()
        pct = min(100.0, 100.0 * self._loop_done / self._loop_total)
        pct_bucket = int(pct // 10)
        due_time = (now - self._loop_last_log) >= self.log_interval_seconds
        due_done = self._loop_done >= self._loop_total
        due_bucket = pct_bucket > self._last_pct_bucket
        if not (due_time or due_done or due_bucket):
            return
        self._loop_last_log = now
        self._last_pct_bucket = pct_bucket
        elapsed = max(now - self._loop_t0, 1e-9)
        rate = self._loop_done / elapsed
        extra_parts = []
        for key, val in extra.items():
            if isinstance(val, int):
                extra_parts.append(f"{key}={val:,}")
            elif isinstance(val, float):
                extra_parts.append(f"{key}={val:.2f}")
            else:
                extra_parts.append(f"{key}={val}")
        extra_s = f" | {' '.join(extra_parts)}" if extra_parts else ""
        eta_s = ""
        if rate > 0 and self._loop_done < self._loop_total:
            eta_s = f" | ETA {((self._loop_total - self._loop_done) / rate):.0f}s"
        self.message(
            f"   {self._loop_label}: {self._loop_done:,}/{self._loop_total:,} "
            f"({pct:.0f}%) | {rate:,.0f}/s{eta_s}{extra_s}"
        )

    def loop_done(self, **extra: Any) -> None:
        if not self.enabled:
            return
        elapsed = max(time.perf_counter() - self._loop_t0, 0.0)
        parts: list[str] = []
        for key, val in extra.items():
            if isinstance(val, int):
                parts.append(f"{key}={val:,}")
            elif isinstance(val, float):
                parts.append(f"{key}={val:.2f}")
            else:
                parts.append(f"{key}={val}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        total_s = f"{self._loop_done:,}/{self._loop_total:,}" if self._loop_total else str(self._loop_done)
        self.message(f"   {self._loop_label}: finished {total_s} in {elapsed:.1f}s{suffix}")


def progress_from_cfg(generator_cfg: dict[str, Any], *, graph_id: str = "") -> BodySimilarityProgress:
    return BodySimilarityProgress(
        bool(generator_cfg.get("show_progress", True)),
        graph_id=graph_id,
        log_interval_seconds=float(generator_cfg.get("progress_log_interval_seconds", 10.0)),
    )
