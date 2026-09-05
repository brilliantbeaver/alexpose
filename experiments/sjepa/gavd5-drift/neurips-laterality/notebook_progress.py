"""Small progress displays and behavior-neutral notebook workflow wrappers."""

from __future__ import annotations

import math
import statistics
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from html import escape
from typing import Any, Callable


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


class NotebookTrainingProgress:
    """Render structured training events in one updating notebook output.

    The estimate uses the median duration of compatible cached checkpoints until
    enough epochs from the active job have completed. It then follows the median
    recent epoch duration. Medians make the display less sensitive to a suspended
    laptop, one-time compilation, or a temporarily busy device.
    """

    def __init__(
        self,
        *,
        refresh_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.refresh_seconds = max(float(refresh_seconds), 0.0)
        self.clock = clock
        self.handle = None
        self.started_at: float | None = None
        self.last_rendered_at = -math.inf
        self.status = "Waiting to start"
        self.profile = ""
        self.device = ""
        self.total_jobs = 0
        self.epochs_per_job = 0
        self.updates_per_epoch = 0
        self.cached_candidate_jobs = 0
        self.new_candidate_jobs = 0
        self.completed_jobs = 0
        self.reused_jobs = 0
        self.trained_jobs = 0
        self.completed_epochs: dict[int, int] = {}
        self.active: dict[str, Any] | None = None
        self.recent_epoch_seconds: list[float] = []
        self.historical_epoch_seconds: list[float] = []
        self.last_completed: dict[str, Any] | None = None
        self.error: str | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        force = kind != "epoch_completed"

        if kind == "run_started":
            # A notebook author may reuse the same display object when re-running
            # the cell. Start with clean counters instead of carrying progress or
            # timing samples forward from the preceding run.
            self.completed_jobs = 0
            self.reused_jobs = 0
            self.trained_jobs = 0
            self.completed_epochs.clear()
            self.active = None
            self.recent_epoch_seconds.clear()
            self.historical_epoch_seconds.clear()
            self.last_completed = None
            self.error = None
            self.started_at = self.clock()
            self.status = "Preparing jobs and validating existing checkpoints"
            self.profile = str(event.get("profile", ""))
            self.device = str(event.get("device", ""))
            self.total_jobs = int(event.get("total_jobs", 0))
            self.epochs_per_job = int(event.get("epochs_per_job", 0))
            self.updates_per_epoch = int(event.get("updates_per_epoch", 0))
            self.cached_candidate_jobs = int(event.get("cached_candidate_jobs", 0))
            self.new_candidate_jobs = int(event.get("new_candidate_jobs", 0))

        elif kind == "job_started":
            self.status = "Training"
            self.active = dict(event)
            self.active["epoch"] = 0
            self.active["mean_total_loss"] = None
            self.recent_epoch_seconds = []

        elif kind == "epoch_completed":
            self.status = "Training"
            self.active = dict(event)
            job_index = int(event["job_index"])
            self.completed_epochs[job_index] = int(event["epoch"])
            epoch_seconds = float(event.get("epoch_seconds", math.nan))
            if math.isfinite(epoch_seconds) and epoch_seconds > 0:
                self.recent_epoch_seconds.append(epoch_seconds)
                self.recent_epoch_seconds = self.recent_epoch_seconds[-60:]

        elif kind == "checkpoint_saving":
            self.status = "Saving and validating checkpoint"
            if self.active is None:
                self.active = dict(event)
            else:
                self.active.update(event)

        elif kind == "job_completed":
            job_index = int(event["job_index"])
            epochs = int(event.get("epochs", self.epochs_per_job))
            self.completed_epochs[job_index] = epochs
            self.completed_jobs = int(event.get("completed_jobs", job_index))
            self.reused_jobs = int(event.get("reused_jobs", self.reused_jobs))
            self.trained_jobs = int(event.get("trained_jobs", self.trained_jobs))
            historical_seconds = float(
                event.get("historical_training_seconds", math.nan)
            )
            event_device = str(event.get("device", ""))
            if (
                epochs > 0
                and math.isfinite(historical_seconds)
                and historical_seconds > 0
                and (not self.device or event_device == self.device)
            ):
                self.historical_epoch_seconds.append(historical_seconds / epochs)
            self.last_completed = dict(event)
            self.active = None
            self.status = (
                "Validated and reused checkpoint"
                if event.get("checkpoint_reused")
                else "Checkpoint complete"
            )

        elif kind == "job_failed":
            self.status = "Stopped because a job failed"
            self.active = dict(event)
            self.error = (
                f"{event.get('error_type', 'Error')}: {event.get('error', '')}"
            )

        elif kind == "run_completed":
            self.status = "Training selection complete"
            self.completed_jobs = int(event.get("completed_jobs", self.total_jobs))
            self.reused_jobs = int(event.get("reused_jobs", self.reused_jobs))
            self.trained_jobs = int(event.get("trained_jobs", self.trained_jobs))
            self.active = None

        now = self.clock()
        if force or now - self.last_rendered_at >= self.refresh_seconds:
            self._publish()
            self.last_rendered_at = now

    def _seconds_per_epoch(self) -> float | None:
        if len(self.recent_epoch_seconds) >= 5:
            return float(statistics.median(self.recent_epoch_seconds[-30:]))
        if self.historical_epoch_seconds:
            return float(statistics.median(self.historical_epoch_seconds))
        if self.recent_epoch_seconds:
            return float(statistics.median(self.recent_epoch_seconds))
        return None

    def _remaining_seconds(self) -> float | None:
        if self.completed_jobs >= self.total_jobs and self.total_jobs:
            return 0.0
        seconds_per_epoch = self._seconds_per_epoch()
        if seconds_per_epoch is None:
            return None

        active_new_job = (
            1
            if self.active and self.status != "Stopped because a job failed"
            else 0
        )
        current_epoch = int(self.active.get("epoch", 0)) if self.active else 0
        current_remaining = (
            max(self.epochs_per_job - current_epoch, 0) if active_new_job else 0
        )
        future_new_jobs = max(
            self.new_candidate_jobs - self.trained_jobs - active_new_job,
            0,
        )
        return seconds_per_epoch * (
            current_remaining + future_new_jobs * self.epochs_per_job
        )

    def _progress_fraction(self) -> float:
        total_epochs = self.total_jobs * self.epochs_per_job
        if total_epochs <= 0:
            return 0.0
        completed = sum(self.completed_epochs.values())
        return min(max(completed / total_epochs, 0.0), 1.0)

    def _elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(self.clock() - self.started_at, 0.0)

    def as_html(self) -> str:
        fraction = self._progress_fraction()
        percent = 100.0 * fraction
        remaining = self._remaining_seconds()
        completion = "unknown"
        if remaining is not None:
            completion = (datetime.now() + timedelta(seconds=remaining)).strftime(
                "%I:%M %p"
            ).lstrip("0")

        active_html = ""
        if self.active:
            loss = self.active.get("mean_total_loss")
            loss_text = "not available yet" if loss is None else f"{float(loss):.5f}"
            job_index = int(self.active.get("job_index", 0))
            epoch = int(self.active.get("epoch", 0))
            epochs = int(self.active.get("epochs", self.epochs_per_job))
            updates = int(self.active.get("optimizer_updates", 0))
            total_updates = int(self.active.get("total_optimizer_updates", 0))
            job_elapsed = _duration(
                float(self.active.get("job_elapsed_seconds", 0.0))
            )
            per_epoch = self._seconds_per_epoch()
            job_remaining = (
                None if per_epoch is None else per_epoch * max(epochs - epoch, 0)
            )
            active_html = f"""
              <div style="margin-top:10px;padding:10px 12px;background:#f5f7fa;border-radius:6px">
                <strong>Active job {job_index}/{self.total_jobs}</strong> &nbsp;·&nbsp;
                {escape(str(self.active.get('variant', '')))} &nbsp;·&nbsp;
                outer fold {int(self.active.get('fold', 0))} &nbsp;·&nbsp;
                seed {int(self.active.get('seed', 0))}<br>
                Epoch <strong>{epoch}/{epochs}</strong> &nbsp;·&nbsp;
                optimizer updates {updates}/{total_updates} &nbsp;·&nbsp;
                loss {loss_text} &nbsp;·&nbsp; elapsed {job_elapsed}
                &nbsp;·&nbsp; job ETA {_duration(job_remaining)}
              </div>
            """

        last_html = ""
        if self.last_completed and not self.active:
            action = "reused" if self.last_completed.get("checkpoint_reused") else "trained"
            last_html = (
                "<div style='margin-top:8px;color:#475569'>Last job: "
                f"{escape(str(self.last_completed.get('variant', '')))}, "
                f"fold {int(self.last_completed.get('fold', 0))}, "
                f"seed {int(self.last_completed.get('seed', 0))} — {action}.</div>"
            )

        error_html = ""
        if self.error:
            error_html = (
                "<div style='margin-top:10px;color:#991b1b'><strong>Error:</strong> "
                f"{escape(self.error)}</div>"
            )

        candidate_note = ""
        if self.cached_candidate_jobs:
            candidate_note = (
                f" {self.cached_candidate_jobs} checkpoint files were present at start; "
                "each is validated before it counts as reused."
            )

        return f"""
        <div style="font-family:system-ui,-apple-system,sans-serif;border:1px solid #cbd5e1;
                    border-radius:8px;padding:14px;max-width:920px">
          <div style="display:flex;justify-content:space-between;gap:16px;align-items:baseline">
            <strong>Fold-local training progress</strong>
            <span>{escape(self.status)}</span>
          </div>
          <div role="progressbar" aria-valuemin="0" aria-valuemax="100"
               aria-valuenow="{percent:.1f}" aria-label="Fold-local training progress"
               style="height:14px;background:#e2e8f0;border-radius:7px;margin:10px 0 6px;overflow:hidden">
            <div style="height:100%;width:{percent:.3f}%;background:#2a9d8f"></div>
          </div>
          <div><strong>{percent:.1f}%</strong> epoch-equivalent work complete &nbsp;·&nbsp;
            jobs {self.completed_jobs}/{self.total_jobs} &nbsp;·&nbsp;
            elapsed {_duration(self._elapsed_seconds())} &nbsp;·&nbsp;
            estimated remaining {_duration(remaining)} &nbsp;·&nbsp;
            estimated finish {completion}</div>
          <div style="margin-top:5px;color:#475569">
            profile {escape(self.profile)} &nbsp;·&nbsp; device {escape(self.device)} &nbsp;·&nbsp;
            reused {self.reused_jobs} &nbsp;·&nbsp; newly trained {self.trained_jobs}
          </div>
          {active_html}{last_html}{error_html}
          <div style="margin-top:10px;font-size:0.88em;color:#64748b">
            ETA is adaptive, not a deadline. It uses robust median timing from compatible
            checkpoints and recent epochs on the current device; thermal throttling, a busy
            machine, or suspending the computer can change it.{escape(candidate_note)}
          </div>
        </div>
        """

    def _text_summary(self) -> str:
        percent = 100.0 * self._progress_fraction()
        return (
            f"[{self.status}] jobs={self.completed_jobs}/{self.total_jobs} "
            f"progress={percent:.1f}% elapsed={_duration(self._elapsed_seconds())} "
            f"remaining={_duration(self._remaining_seconds())}"
        )

    def _publish(self) -> None:
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
        except ImportError:
            print(self._text_summary(), flush=True)


class NotebookTaskProgress:
    """Render progress for evaluation, reporting, and validation workflows.

    A unit can be an evaluation job, a statistical-reporting stage, or a
    manifest-validation pass. ETA uses the median duration of completed,
    non-cached units. Cached candidates count as complete only after the
    wrapped scientific function has validated and returned them.
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
            self.total_units - self.cached_candidate_units,
            0,
        )
        self.reused_units = 0
        self.computed_units = 0
        self.skipped_units = 0
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

    @contextmanager
    def unit(
        self,
        index: int,
        label: str,
        *,
        detail: str = "",
        candidate_cached: bool = False,
        total_steps: int = 0,
    ):
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
        duration_seconds: float | None = None,
    ) -> None:
        now = self.clock()
        active = dict(self.active or {})
        if duration_seconds is None and self.active is not None:
            duration_seconds = now - float(self.active["started_at"])
        duration = (
            float(duration_seconds)
            if duration_seconds is not None
            else math.nan
        )
        self.completed_units = min(self.completed_units + 1, self.total_units)
        if reused:
            self.reused_units += 1
            self.status = "Validated and reused cached result"
        else:
            self.computed_units += 1
            self.status = "Stage complete"
            if math.isfinite(duration) and duration > 0:
                self.unit_seconds.append(duration)
                self.unit_seconds = self.unit_seconds[-60:]
        active["reused"] = bool(reused)
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
        self.finished = True
        self._publish(force=True)

    def finish_skipped(
        self,
        reason: str,
        *,
        status: str = "Not configured / not run",
    ) -> None:
        """Finish a deliberately optional workflow without presenting an error.

        Skipped units are accounted for separately from computed units. This
        allows the progress bar to show that preflight reached a terminal
        decision while the status still makes clear that scientific work did
        not run.
        """

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
            self.new_candidate_units - self.computed_units - int(active_new),
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
        terminal_accounted_for = self.finished and self.completed_units >= self.total_units
        remaining = (
            0.0
            if terminal_accounted_for
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
            if self.skipped_reason
            else "#2a9d8f"
        )

        active_html = ""
        if self.active:
            step_html = ""
            total_steps = int(self.active.get("total_steps", 0))
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
            action = "validated and reused" if self.last_completed.get("reused") else "completed"
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
          <div><strong>{percent:.1f}%</strong> {"accounted for" if self.skipped_units else "complete"} &nbsp;·&nbsp;
            {escape(self.unit_name)}s {self.completed_units}/{self.total_units} &nbsp;·&nbsp;
            elapsed {_duration(self._elapsed_seconds())} &nbsp;·&nbsp;
            estimated remaining {_duration(remaining)} &nbsp;·&nbsp;
            estimated finish {completion}</div>
          <div style="margin-top:5px;color:#475569">
            {profile_note}reused {self.reused_units} &nbsp;·&nbsp; newly computed {self.computed_units}
            &nbsp;·&nbsp; skipped {self.skipped_units}
          </div>
          {active_html}{last_html}{message_html}
          <div style="margin-top:10px;font-size:0.88em;color:#64748b">
            ETA is adaptive, not a deadline. It uses the median duration of completed,
            non-cached {escape(self.unit_name)}s; differently sized sources, bootstrap
            stages, device load, or suspended execution can change it.{escape(cache_note)}
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


def evaluate_selected_with_progress(
    context,
    cohort,
    splits,
    *,
    progress: NotebookTaskProgress,
):
    """Evaluate selected checkpoints with job-level progress and unchanged math."""
    import pandas as pd

    from laterality.artifacts import evaluation_path
    from laterality.evaluation import evaluate_fold, evaluation_metadata_path

    jobs = [
        (str(variant), int(fold), int(seed))
        for variant in context.variants
        for fold in context.folds
        for seed in context.seeds
    ]
    candidates = {
        (variant, fold, seed): (
            (path := evaluation_path(context.artifact_root, variant, fold, seed)).exists()
            and evaluation_metadata_path(path).exists()
        )
        for variant, fold, seed in jobs
    }
    progress.start(
        len(jobs),
        profile=context.profile,
        cached_candidate_units=sum(candidates.values()),
        note=(
            " A fresh job encodes original and mirrored poses with learned and paired "
            "initial encoders, then tunes all registered read-out lanes."
        ),
    )
    def run_jobs():
        outputs = []
        for index, (variant, fold, seed) in enumerate(jobs, start=1):
            cached = bool(candidates[(variant, fold, seed)])
            progress.start_unit(
                index,
                f"{variant} · outer fold {fold} · seed {seed}",
                detail=(
                    "Checking lineage, file digest, row coverage, and result digest"
                    if cached
                    else "Encoding four pose passes and tuning 16 read-out lanes"
                ),
                candidate_cached=cached,
            )
            started = time.monotonic()
            frame = evaluate_fold(
                context,
                cohort,
                splits,
                fold,
                seed,
                variant,
            )
            outputs.append(frame)
            progress.complete_unit(
                reused=cached,
                duration_seconds=time.monotonic() - started,
            )
        return outputs

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "once",
                message="enable_nested_tensor is True.*",
                category=UserWarning,
            )
            outputs = run_jobs()
    except BaseException as error:
        progress.fail(error)
        raise
    progress.complete(status="Held-out evaluation complete")
    return pd.concat(outputs, ignore_index=True)


def aggregate_and_save_with_progress(
    context,
    cohort,
    splits,
    *,
    progress: NotebookTaskProgress,
):
    """Run the registered report while exposing its existing top-level stages.

    The wrapper temporarily decorates reporting functions with timers, calls the
    unchanged ``aggregate_and_save`` implementation, and restores every function
    in ``finally``. It neither changes arguments nor intercepts return values.
    """
    from laterality import reporting

    phase_labels = {
        "load_selected_evaluations": ["Load and validate every evaluation artifact"],
        "seed_average_predictions": ["Build the secondary seed-mean prediction ensemble"],
        "metric_table": [
            "Compute all-cohort ensemble metrics",
            "Compute the high-pose-coverage sensitivity metrics",
        ],
        "optimization_seed_table": ["Compute single-checkpoint seed diagnostics"],
        "bootstrap_table": ["Bootstrap secondary ensemble contrasts by source"],
        "checkpoint_bootstrap_table": ["Bootstrap primary checkpoint estimands by source"],
        "native_symmetry_bootstrap_table": ["Bootstrap native output-symmetry errors"],
        "native_symmetry_seed_table": ["Check output symmetry separately for every seed"],
        "representation_equivariance_bootstrap_table": ["Bootstrap strict representation-equivariance errors"],
        "representation_equivariance_seed_table": ["Check representation equivariance for every seed"],
        "_write_overview_figure": ["Write the compact metric overview"],
        "atomic_write_json": ["Evaluate gates and write the final report summary"],
    }
    total_phases = sum(len(labels) for labels in phase_labels.values())
    progress.start(
        total_phases,
        profile=context.profile,
        note=(
            " Reporting stages differ in cost; the 2,000-repeat source bootstraps "
            "usually dominate, so early ETA can move substantially."
        ),
    )
    originals: dict[str, Callable[..., Any]] = {}
    call_counts = {name: 0 for name in phase_labels}
    phase_index = 0

    def decorate(name: str, original: Callable[..., Any]):
        def wrapped(*args, **kwargs):
            nonlocal phase_index
            offset = call_counts[name]
            labels = phase_labels[name]
            label = labels[offset] if offset < len(labels) else f"{name} stage {offset + 1}"
            call_counts[name] += 1
            phase_index += 1
            progress.start_unit(
                phase_index,
                label,
                detail="Running the registered calculation without changing its inputs",
            )
            started = time.monotonic()
            result = original(*args, **kwargs)
            progress.complete_unit(duration_seconds=time.monotonic() - started)
            return result

        return wrapped

    for function_name in phase_labels:
        original = getattr(reporting, function_name)
        originals[function_name] = original
        setattr(reporting, function_name, decorate(function_name, original))
    try:
        report = reporting.aggregate_and_save(context, cohort, splits)
    except BaseException as error:
        progress.fail(error)
        raise
    finally:
        for function_name, original in originals.items():
            setattr(reporting, function_name, original)
    if phase_index != total_phases:
        error = RuntimeError(
            f"Reporting progress plan expected {total_phases} stages but observed {phase_index}"
        )
        progress.fail(error)
        raise error
    progress.complete(status="Statistical report complete")
    return report
