"""Behavior-neutral progress displays for the BrainBodyFM notebooks.

The display is deliberately observational: it measures elapsed wall-clock time
and renders notebook output, but it never changes scientific inputs, random
state, return values, or cache decisions.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from html import escape
from typing import Any, Callable, Iterator


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating…"
    rounded = int(round(seconds))
    if rounded < 60:
        return f"{rounded}s"
    minutes, remaining_seconds = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


class NotebookTaskProgress:
    """Render named notebook stages in one updating output.

    ETA uses the median duration of completed, non-cached units. Optional inner
    step counts make a long active stage visibly advance without treating its
    iterations as independent scientific jobs.
    """

    def __init__(
        self,
        title: str,
        unit_name: str,
        *,
        refresh_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        initial_seconds_per_unit: float | None = None,
    ) -> None:
        self.title = str(title)
        self.unit_name = str(unit_name)
        self.refresh_seconds = max(float(refresh_seconds), 0.0)
        self.clock = clock
        self.initial_seconds_per_unit = initial_seconds_per_unit
        self.handle = None
        self.last_rendered_at = -math.inf
        self.started_at: float | None = None
        self.total_units = 0
        self.completed_units = 0
        self.cached_candidate_units = 0
        self.new_candidate_units = 0
        self.reused_units = 0
        self.computed_units = 0
        self.skipped_units = 0
        self.failed_units = 0
        self.retry_events = 0
        self.profile = ""
        self.status = "Waiting to start"
        self.note = ""
        self.active: dict[str, Any] | None = None
        self.last_completed: dict[str, Any] | None = None
        self.unit_seconds: list[float] = []
        self.error: str | None = None
        self.blocked_reason: str | None = None
        self.skipped_reason: str | None = None
        self.finished = False

    def start(
        self,
        total_units: int,
        *,
        profile: str = "",
        cached_candidate_units: int = 0,
        note: str = "",
    ) -> None:
        self.started_at = self.clock()
        self.total_units = max(int(total_units), 0)
        self.completed_units = 0
        self.cached_candidate_units = max(int(cached_candidate_units), 0)
        self.new_candidate_units = max(
            self.total_units - self.cached_candidate_units, 0
        )
        self.reused_units = 0
        self.computed_units = 0
        self.skipped_units = 0
        self.failed_units = 0
        self.retry_events = 0
        self.profile = str(profile)
        self.status = "Preparing and validating inputs"
        self.note = str(note)
        self.active = None
        self.last_completed = None
        self.unit_seconds.clear()
        self.error = None
        self.blocked_reason = None
        self.skipped_reason = None
        self.finished = False
        self._publish(force=True)

    def revise_plan(
        self,
        total_units: int,
        *,
        cached_candidate_units: int | None = None,
        note: str | None = None,
    ) -> None:
        """Refine a plan after discovery without resetting completed work.

        Notebook setup often has to validate a manifest before it knows the
        number of per-source or per-sequence units.  This method lets the live
        display adopt that discovered total while preserving elapsed time and
        already-completed units.
        """

        requested_total = int(total_units)
        if requested_total < self.completed_units:
            raise ValueError(
                "A revised progress total cannot be smaller than completed work"
            )
        self.total_units = max(requested_total, 0)
        if cached_candidate_units is not None:
            self.cached_candidate_units = min(
                max(int(cached_candidate_units), 0), self.total_units
            )
        self.new_candidate_units = max(
            self.total_units - self.cached_candidate_units, 0
        )
        if note is not None:
            self.note = str(note)
        self._publish(force=True)

    def start_unit(
        self,
        index: int,
        label: str,
        *,
        detail: str = "",
        candidate_cached: bool = False,
        total_steps: int = 0,
    ) -> None:
        self.total_units = max(self.total_units, int(index))
        self.status = "Validating cached result" if candidate_cached else "Running"
        self.active = {
            "index": int(index),
            "label": str(label),
            "detail": str(detail),
            "candidate_cached": bool(candidate_cached),
            "started_at": self.clock(),
            "completed_steps": 0,
            "total_steps": max(int(total_steps), 0),
        }
        self._publish(force=True)

    def update_unit(
        self,
        *,
        detail: str | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        if self.active is None:
            return
        if detail is not None:
            self.active["detail"] = str(detail)
        if total_steps is not None:
            self.active["total_steps"] = max(int(total_steps), 0)
        if completed_steps is not None:
            self.active["completed_steps"] = max(int(completed_steps), 0)
        self._publish(force=False)

    def retry_unit(
        self,
        *,
        attempt: int,
        max_attempts: int,
        detail: str,
    ) -> None:
        """Record a bounded retry without pretending the unit completed."""

        if self.active is None:
            return
        self.retry_events += 1
        self.status = f"Retrying (attempt {attempt + 1}/{max_attempts})"
        self.active["detail"] = str(detail)
        self._publish(force=True)

    @contextmanager
    def unit(
        self,
        index: int,
        label: str,
        *,
        detail: str = "",
        candidate_cached: bool = False,
        total_steps: int = 0,
    ) -> Iterator[NotebookTaskProgress]:
        """Time one unit and turn exceptions into an explicit failed state."""
        self.start_unit(
            index,
            label,
            detail=detail,
            candidate_cached=candidate_cached,
            total_steps=total_steps,
        )
        started = self.clock()
        try:
            yield self
        except BaseException as error:
            self.fail(error)
            raise
        else:
            self.complete_unit(
                reused=candidate_cached,
                duration_seconds=self.clock() - started,
            )

    def complete_unit(
        self,
        *,
        reused: bool = False,
        outcome: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        now = self.clock()
        active = dict(self.active or {})
        if duration_seconds is None and self.active is not None:
            duration_seconds = now - float(self.active["started_at"])
        duration = (
            float(duration_seconds) if duration_seconds is not None else math.nan
        )
        if outcome is None:
            outcome = "reused" if reused else "computed"
        outcome = str(outcome).lower()
        if outcome not in {"computed", "reused", "skipped", "failed"}:
            raise ValueError(f"Unknown progress-unit outcome {outcome!r}")
        reused = outcome == "reused"
        self.completed_units = min(self.completed_units + 1, self.total_units)
        if outcome == "reused":
            self.reused_units += 1
            self.status = "Validated and reused cached result"
        elif outcome == "computed":
            self.computed_units += 1
            self.status = "Stage complete"
            if math.isfinite(duration) and duration > 0:
                self.unit_seconds.append(duration)
                self.unit_seconds = self.unit_seconds[-60:]
        elif outcome == "skipped":
            self.skipped_units += 1
            self.status = "Optional unit not run"
        else:
            self.failed_units += 1
            self.status = "Unit failed; continuing audit"
        active["reused"] = bool(reused)
        active["outcome"] = outcome
        active["duration_seconds"] = duration
        self.last_completed = active
        self.active = None
        self._publish(force=True)

    def complete(self, *, status: str = "Complete") -> None:
        self.completed_units = self.total_units
        self.status = str(status)
        self.active = None
        self.finished = True
        self._publish(force=True)

    def fail(self, error: BaseException | str) -> None:
        self.status = "Stopped because a stage failed"
        self.error = (
            str(error)
            if isinstance(error, str)
            else f"{type(error).__name__}: {error}"
        )
        self.active = None
        self.finished = True
        self._publish(force=True)

    def finish_skipped(
        self,
        reason: str,
        *,
        status: str = "Not configured / not run",
    ) -> None:
        """Account for optional remaining stages without claiming computation."""
        remaining = max(self.total_units - self.completed_units, 0)
        self.skipped_units += remaining
        self.completed_units = self.total_units
        self.status = str(status)
        self.skipped_reason = str(reason)
        self.active = None
        self.finished = True
        self._publish(force=True)

    def block(self, reason: str, *, account_for_remaining: bool = False) -> None:
        if account_for_remaining:
            remaining = max(self.total_units - self.completed_units, 0)
            self.skipped_units += remaining
            self.completed_units = self.total_units
        self.status = "Blocked / not run"
        self.blocked_reason = str(reason)
        self.active = None
        self.finished = True
        self._publish(force=True)

    def _seconds_per_unit(self) -> float | None:
        if self.unit_seconds:
            return float(statistics.median(self.unit_seconds[-20:]))
        hint = self.initial_seconds_per_unit
        if hint is not None and math.isfinite(float(hint)) and float(hint) > 0:
            return float(hint)
        return None

    def _active_fraction(self) -> float:
        if self.active is None:
            return 0.0
        completed_steps = int(self.active.get("completed_steps", 0))
        total_steps = int(self.active.get("total_steps", 0))
        if total_steps <= 0:
            return 0.0
        return min(max(completed_steps / total_steps, 0.0), 1.0)

    def _progress_fraction(self) -> float:
        if self.total_units <= 0:
            return 1.0 if self.finished and not self.blocked_reason else 0.0
        amount = self.completed_units + self._active_fraction()
        return min(max(amount / self.total_units, 0.0), 1.0)

    def _remaining_seconds(self) -> float | None:
        if self.completed_units >= self.total_units and self.total_units:
            return 0.0
        seconds_per_unit = self._seconds_per_unit()
        if seconds_per_unit is None:
            return None
        active_new = bool(
            self.active and not self.active.get("candidate_cached", False)
        )
        active_remaining = 0.0
        if active_new:
            active_remaining = 1.0 - self._active_fraction()
        future_new = max(
            self.new_candidate_units
            - self.computed_units
            - self.failed_units
            - self.skipped_units
            - int(active_new),
            0,
        )
        return seconds_per_unit * (active_remaining + future_new)

    def _elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(self.clock() - self.started_at, 0.0)

    def as_html(self) -> str:
        fraction = self._progress_fraction()
        percent = 100.0 * fraction
        terminal = self.finished and self.completed_units >= self.total_units
        remaining = (
            0.0
            if terminal
            else None
            if self.blocked_reason or self.error
            else self._remaining_seconds()
        )
        completion = "unknown"
        if remaining is not None:
            completion = (datetime.now() + timedelta(seconds=remaining)).strftime(
                "%I:%M %p"
            ).lstrip("0")
        bar_color = (
            "#c44536"
            if self.blocked_reason or self.error
            else "#b7791f"
            if self.skipped_reason or self.failed_units
            else "#2a9d8f"
        )

        active_html = ""
        if self.active:
            total_steps = int(self.active.get("total_steps", 0))
            step_html = ""
            if total_steps:
                step_html = (
                    f" &nbsp;·&nbsp; steps {int(self.active.get('completed_steps', 0))}"
                    f"/{total_steps}"
                )
            active_elapsed = self.clock() - float(self.active["started_at"])
            active_html = f"""
              <div style="margin-top:10px;padding:10px 12px;background:#f5f7fa;border-radius:6px">
                <strong>Active {escape(self.unit_name)} {int(self.active['index'])}/{self.total_units}</strong>
                &nbsp;·&nbsp; {escape(str(self.active['label']))}{step_html}<br>
                {escape(str(self.active.get('detail', '')))}
                <span style="color:#64748b"> &nbsp;·&nbsp; active for {_duration(active_elapsed)}</span>
              </div>
            """

        last_html = ""
        if self.last_completed and not self.active:
            action = {
                "reused": "validated and reused",
                "computed": "completed",
                "skipped": "not run",
                "failed": "failed and recorded",
            }.get(str(self.last_completed.get("outcome", "computed")), "completed")
            last_html = (
                "<div style='margin-top:8px;color:#475569'>Last "
                f"{escape(self.unit_name)}: {escape(str(self.last_completed.get('label', '')))} "
                f"— {action} in {_duration(self.last_completed.get('duration_seconds'))}.</div>"
            )

        message_html = ""
        if self.error:
            message_html = (
                "<div style='margin-top:10px;color:#991b1b'><strong>Error:</strong> "
                f"{escape(self.error)}</div>"
            )
        elif self.blocked_reason:
            message_html = (
                "<div style='margin-top:10px;color:#991b1b'><strong>Reason:</strong> "
                f"{escape(self.blocked_reason)}</div>"
            )
        elif self.skipped_reason:
            message_html = (
                "<div style='margin-top:10px;color:#92400e'><strong>Not run:</strong> "
                f"{escape(self.skipped_reason)}</div>"
            )

        cache_note = ""
        if self.cached_candidate_units:
            cache_note = (
                f" {self.cached_candidate_units} cached candidates were present at start; "
                "each counts only after validation."
            )
        profile_note = (
            f"profile {escape(self.profile)} &nbsp;·&nbsp; " if self.profile else ""
        )
        accounted = "accounted for" if self.skipped_units else "complete"
        return f"""
        <div style="font-family:system-ui,-apple-system,sans-serif;border:1px solid #cbd5e1;
                    border-radius:8px;padding:14px;max-width:920px">
          <div style="display:flex;justify-content:space-between;gap:16px;align-items:baseline">
            <strong>{escape(self.title)}</strong><span>{escape(self.status)}</span>
          </div>
          <div role="progressbar" aria-valuemin="0" aria-valuemax="100"
               aria-valuenow="{percent:.1f}" aria-label="{escape(self.title)}"
               style="height:14px;background:#e2e8f0;border-radius:7px;margin:10px 0 6px;overflow:hidden">
            <div style="height:100%;width:{percent:.3f}%;background:{bar_color}"></div>
          </div>
          <div><strong>{percent:.1f}%</strong> {accounted} &nbsp;·&nbsp;
            {escape(self.unit_name)}s {self.completed_units}/{self.total_units} &nbsp;·&nbsp;
            elapsed {_duration(self._elapsed_seconds())} &nbsp;·&nbsp;
            estimated remaining {_duration(remaining)} &nbsp;·&nbsp;
            estimated finish {completion}</div>
          <div style="margin-top:5px;color:#475569">
            {profile_note}reused {self.reused_units} &nbsp;·&nbsp;
            newly computed {self.computed_units} &nbsp;·&nbsp; skipped {self.skipped_units}
            &nbsp;·&nbsp; failed {self.failed_units} &nbsp;·&nbsp; retries {self.retry_events}
          </div>
          {active_html}{last_html}{message_html}
          <div style="margin-top:10px;font-size:0.88em;color:#64748b">
            ETA is adaptive, not a deadline. It uses the median duration of completed,
            non-cached {escape(self.unit_name)}s; differently sized sources, device load,
            network service, or suspended execution can change it.{escape(cache_note)}
            {escape(self.note)}
          </div>
        </div>
        """

    def _text_summary(self) -> str:
        return (
            f"[{self.status}] {self.unit_name}s={self.completed_units}/{self.total_units} "
            f"progress={100.0 * self._progress_fraction():.1f}% "
            f"elapsed={_duration(self._elapsed_seconds())} "
            f"remaining={_duration(self._remaining_seconds())}"
        )

    def _publish(self, *, force: bool) -> None:
        now = self.clock()
        if not force and now - self.last_rendered_at < self.refresh_seconds:
            return
        self.last_rendered_at = now
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            if get_ipython() is None:
                print(self._text_summary(), flush=True)
                return
            content = HTML(self.as_html())
            if self.handle is None:
                self.handle = display(content, display_id=True)
            elif hasattr(self.handle, "update"):
                self.handle.update(content)
            else:
                display(content)
        except Exception as error:  # A display must never discard scientific work.
            warnings.warn(
                f"Notebook progress display failed and was disabled: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
