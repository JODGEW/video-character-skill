"""Shared plumbing for fal.ai queue-backed providers.

Every fal endpoint we target has the same lifecycle — upload local files, POST
to the queue, poll a status, fetch a result — and differs only in its
application id, request body and result shape.

:class:`FalQueueBase` owns the part that is the same everywhere. On top of it,
:class:`FalQueueProvider` serves character-transfer endpoints (one video in,
one video out); segmentation endpoints, whose results are not videos at all,
build on the base directly.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Protocol

import fal_client

from video_character_skill.providers.base import CharacterTransferProvider, ProviderError
from video_character_skill.schemas import (
    Job,
    JobStatus,
    MediaAsset,
    ResultVideo,
    TransferRequest,
)


class _RequestHandle(Protocol):
    @property
    def request_id(self) -> str: ...


class FalQueueClient(Protocol):
    """The slice of ``fal_client.SyncClient`` these providers use."""

    def upload_file(self, path: os.PathLike[str]) -> str: ...

    def submit(self, application: str, arguments: dict[str, Any]) -> _RequestHandle: ...

    def status(
        self, application: str, request_id: str, *, with_logs: bool = False
    ) -> fal_client.Status: ...

    def result(self, application: str, request_id: str) -> Any: ...


class FalQueueBase:
    """Submit / poll / fetch against one fal application.

    Endpoint-agnostic: it knows the queue lifecycle and how to turn a local
    path into a fal URL, but nothing about any particular request body or
    result shape. Subclasses add the typed ``submit``/``get_result`` pair for
    the job kind they serve.
    """

    name: ClassVar[str]
    app_id: ClassVar[str]

    def __init__(self, client: FalQueueClient | None = None) -> None:
        # SyncClient resolves FAL_KEY lazily, at request time.
        self._client: FalQueueClient = client if client is not None else fal_client.SyncClient()

    # -- queue lifecycle -----------------------------------------------

    def _enqueue(self, arguments: dict[str, Any]) -> Job:
        try:
            handle = self._client.submit(self.app_id, arguments)
        except fal_client.FalClientError as exc:
            raise ProviderError(f"fal submit failed: {exc}") from exc
        return Job(job_id=handle.request_id, provider=self.name, status=JobStatus.QUEUED)

    def get_status(self, job_id: str) -> Job:
        try:
            status = self._client.status(self.app_id, job_id)
        except fal_client.FalClientError as exc:
            raise ProviderError(f"fal status failed for {job_id}: {exc}") from exc
        return self._to_job(job_id, status)

    def _result_payload(self, job_id: str) -> Any:
        """Fetch a succeeded job's raw result body.

        A fal job that the queue reports as ``Completed`` can still fail when
        its result is fetched, so the status check happens here rather than in
        the caller.
        """
        job = self.get_status(job_id)
        if job.status is not JobStatus.SUCCEEDED:
            detail = f": {job.error}" if job.error else ""
            raise ProviderError(f"job {job_id} is {job.status.value}, not succeeded{detail}")
        try:
            return self._client.result(self.app_id, job_id)
        except fal_client.FalClientError as exc:
            raise ProviderError(f"fal result failed for {job_id}: {exc}") from exc

    # -- uploads -------------------------------------------------------

    def _resolve_url(self, asset: MediaAsset) -> str:
        """Return a URL fal can fetch, uploading the file if it is local."""
        if asset.is_remote:
            return asset.uri
        path = Path(asset.uri)
        if not path.is_file():
            raise ProviderError(f"local file not found: {asset.uri}")
        try:
            return self._client.upload_file(path)
        except fal_client.FalClientError as exc:
            raise ProviderError(f"fal upload failed for {asset.uri}: {exc}") from exc
        except OSError as exc:
            raise ProviderError(f"could not read {asset.uri}: {exc}") from exc

    # -- response parsing ----------------------------------------------

    def _to_job(self, job_id: str, status: fal_client.Status) -> Job:
        """Map a fal queue state onto our normalized status."""
        if isinstance(status, fal_client.Queued):
            return Job(job_id=job_id, provider=self.name, status=JobStatus.QUEUED)
        if isinstance(status, fal_client.InProgress):
            return Job(job_id=job_id, provider=self.name, status=JobStatus.RUNNING)
        if isinstance(status, fal_client.Completed):
            if status.error is None:
                return Job(job_id=job_id, provider=self.name, status=JobStatus.SUCCEEDED)
            return Job(
                job_id=job_id,
                provider=self.name,
                status=JobStatus.FAILED,
                error=self._error_text(status),
            )
        raise ProviderError(f"unrecognized fal status for {job_id}: {status!r}")

    @staticmethod
    def _error_text(status: fal_client.Completed) -> str:
        parts = [str(part) for part in (status.error_type, status.error) if part]
        return ": ".join(parts) or "unknown error"


class FalQueueProvider(FalQueueBase, CharacterTransferProvider):
    """A character-transfer endpoint on the fal queue."""

    @abstractmethod
    def build_arguments(self, request: TransferRequest) -> dict[str, Any]:
        """Build the endpoint's request body, uploading any local files first."""

    def submit(self, request: TransferRequest) -> Job:
        return self._enqueue(self.build_arguments(request))

    def get_result(self, job_id: str) -> ResultVideo:
        return self._to_result_video(job_id, self._result_payload(job_id))

    def _to_result_video(self, job_id: str, payload: Any) -> ResultVideo:
        video = payload.get("video") if isinstance(payload, dict) else None
        if not isinstance(video, dict):
            raise ProviderError(f"fal result for {job_id} has no video object: {payload!r}")
        url = video.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ProviderError(f"fal result for {job_id} has no video url: {video!r}")
        content_type = video.get("content_type")
        if isinstance(content_type, str) and content_type.strip():
            return ResultVideo(uri=url, content_type=content_type)
        return ResultVideo(uri=url)
