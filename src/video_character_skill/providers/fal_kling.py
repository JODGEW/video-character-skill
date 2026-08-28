"""fal.ai provider for Kling V3 Standard Motion Control.

Endpoint: ``fal-ai/kling-video/v3/standard/motion-control``

Input (as documented by fal):
    image_url               required  reference image
    video_url               required  driving video
    character_orientation   required  "image" | "video"
    keep_original_sound     optional  default true
    prompt                  optional  max 2500 chars
    elements                optional  list<KlingV3ImageElementInput>, at most 1

``KlingV3ImageElementInput`` (both fields nullable, per fal's OpenAPI spec)::

    frontal_image_url       "The frontal image of the element (main view)."
    reference_image_urls    "Additional reference images from different angles.
                             1-3 images supported. At least one image is required."

fal documents element binding as supported only when ``character_orientation``
is ``"video"``; that combination is rejected here before anything is uploaded.
The element is referenced from the prompt as ``@Element1`` — writing that into
the prompt is left to the caller.

Output::

    {"video": {"url": ..., "content_type": ..., "file_name": ..., "file_size": ...}}

Local file paths are uploaded to fal storage first; ``http(s)`` URIs are passed
through untouched. This module submits and polls on demand — the caller owns
any polling loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Protocol

import fal_client

from video_character_skill.providers.base import CharacterTransferProvider, ProviderError
from video_character_skill.schemas import (
    CharacterOrientation,
    IdentityElement,
    Job,
    JobStatus,
    MediaAsset,
    ResultVideo,
    TransferRequest,
)

APP_ID = "fal-ai/kling-video/v3/standard/motion-control"


class _RequestHandle(Protocol):
    @property
    def request_id(self) -> str: ...


class FalQueueClient(Protocol):
    """The slice of ``fal_client.SyncClient`` this provider uses."""

    def upload_file(self, path: os.PathLike[str]) -> str: ...

    def submit(self, application: str, arguments: dict[str, Any]) -> _RequestHandle: ...

    def status(
        self, application: str, request_id: str, *, with_logs: bool = False
    ) -> fal_client.Status: ...

    def result(self, application: str, request_id: str) -> Any: ...


class FalKlingProvider(CharacterTransferProvider):
    """Kling V3 Standard Motion Control, via the fal queue API."""

    name: ClassVar[str] = "fal-kling-v3-standard-motion-control"
    app_id: ClassVar[str] = APP_ID

    def __init__(self, client: FalQueueClient | None = None) -> None:
        # SyncClient resolves FAL_KEY lazily, at request time.
        self._client: FalQueueClient = client if client is not None else fal_client.SyncClient()

    # -- interface -----------------------------------------------------

    def submit(self, request: TransferRequest) -> Job:
        arguments = self.build_arguments(request)
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

    def get_result(self, job_id: str) -> ResultVideo:
        job = self.get_status(job_id)
        if job.status is not JobStatus.SUCCEEDED:
            detail = f": {job.error}" if job.error else ""
            raise ProviderError(f"job {job_id} is {job.status.value}, not succeeded{detail}")
        try:
            payload = self._client.result(self.app_id, job_id)
        except fal_client.FalClientError as exc:
            raise ProviderError(f"fal result failed for {job_id}: {exc}") from exc
        return self._to_result_video(job_id, payload)

    # -- payload -------------------------------------------------------

    def build_arguments(self, request: TransferRequest) -> dict[str, Any]:
        """Build the fal request body, uploading any local files first.

        Raises:
            ProviderError: on a combination fal rejects, checked before upload.
        """
        if (
            request.identity_element is not None
            and request.character_orientation is not CharacterOrientation.VIDEO
        ):
            raise ProviderError(
                "element binding requires character_orientation='video', got "
                f"'{request.character_orientation.value}'"
            )
        arguments: dict[str, Any] = {
            "image_url": self._resolve_url(request.reference_image),
            "video_url": self._resolve_url(request.driving_video),
            "character_orientation": request.character_orientation.value,
            "keep_original_sound": request.keep_original_sound,
        }
        if request.prompt is not None:
            arguments["prompt"] = request.prompt
        if request.identity_element is not None:
            arguments["elements"] = [self._element_payload(request.identity_element)]
        return arguments

    def _element_payload(self, element: IdentityElement) -> dict[str, Any]:
        """One ``KlingV3ImageElementInput``; null-valued keys are omitted."""
        payload: dict[str, Any] = {}
        if element.frontal_image is not None:
            payload["frontal_image_url"] = self._resolve_url(element.frontal_image)
        if element.additional_images:
            payload["reference_image_urls"] = [
                self._resolve_url(image) for image in element.additional_images
            ]
        return payload

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
