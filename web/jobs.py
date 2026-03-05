"""Background job runner using threading with stdout capture."""

import io
import sys
import threading
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime


class Job:
    __slots__ = ("id", "type", "status", "started_at", "finished_at", "log", "error")

    def __init__(self, job_type: str):
        self.id = str(uuid.uuid4())[:8]
        self.type = job_type
        self.status = "running"
        self.started_at = datetime.now().isoformat()
        self.finished_at = None
        self.log = ""
        self.error = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": self.log,
            "error": self.error,
        }


class JobRunner:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _active_job_of_type(self, job_type: str) -> Job | None:
        for job in self._jobs.values():
            if job.type == job_type and job.status == "running":
                return job
        return None

    def start(self, job_type: str, func, *args, **kwargs) -> Job | None:
        """Start a background job. Returns None if a job of this type is already running."""
        with self._lock:
            if self._active_job_of_type(job_type):
                return None
            job = Job(job_type)
            self._jobs[job.id] = job

        def _run():
            buf = io.StringIO()
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    func(*args, **kwargs)
                job.status = "completed"
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                buf.write(f"\n--- ERROR ---\n{traceback.format_exc()}")
            finally:
                job.finished_at = datetime.now().isoformat()
                job.log = buf.getvalue()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all_jobs(self) -> list[dict]:
        return [j.to_dict() for j in sorted(
            self._jobs.values(),
            key=lambda j: j.started_at,
            reverse=True,
        )]


# Singleton
runner = JobRunner()
