"""Shared fal client fake. Records calls, returns canned responses, never
touches the network. Real ``fal_client`` types are used so the queue-state
mapping is checked against the actual library."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fal_client

COMPLETED_OK = fal_client.Completed(logs=[], metrics={}, error=None, error_type=None)

SUCCESS_PAYLOAD = {
    "video": {
        "url": "https://v3.fal.media/files/out.mp4",
        "content_type": "video/mp4",
        "file_name": "out.mp4",
        "file_size": 1234,
    }
}


@dataclass
class FakeHandle:
    request_id: str


@dataclass
class FakeFalClient:
    """Records calls; returns canned responses. Never touches the network."""

    status_response: fal_client.Status = field(
        default_factory=lambda: fal_client.Queued(position=0)
    )
    result_response: Any = field(default_factory=lambda: SUCCESS_PAYLOAD)
    error: fal_client.FalClientError | None = None
    result_error: fal_client.FalClientError | None = None

    uploads: list[str] = field(default_factory=list)
    submissions: list[tuple[str, Any]] = field(default_factory=list)
    status_calls: list[tuple[str, str]] = field(default_factory=list)
    result_calls: list[tuple[str, str]] = field(default_factory=list)

    def upload_file(self, path: os.PathLike[str]) -> str:
        if self.error is not None:
            raise self.error
        self.uploads.append(os.fspath(path))
        return f"https://v3.fal.media/files/{Path(path).name}"

    def submit(self, application: str, arguments: Any) -> FakeHandle:
        if self.error is not None:
            raise self.error
        self.submissions.append((application, arguments))
        return FakeHandle(request_id="req-1")

    def status(
        self, application: str, request_id: str, *, with_logs: bool = False
    ) -> fal_client.Status:
        if self.error is not None:
            raise self.error
        self.status_calls.append((application, request_id))
        return self.status_response

    def result(self, application: str, request_id: str) -> Any:
        if self.result_error is not None:
            raise self.result_error
        self.result_calls.append((application, request_id))
        return self.result_response
