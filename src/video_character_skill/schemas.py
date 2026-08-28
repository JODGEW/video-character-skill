"""Typed inputs and outputs for a character-transfer job.

A job takes one reference image (who the person looks like) and one driving
video (how the person moves), and produces one result video.

Media is referenced by URI only. For this POC a URI is either an ``http(s)://``
URL or a local filesystem path; nothing here uploads, downloads or inspects the
bytes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MediaAsset(BaseModel):
    """A single media file addressed by URI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(description="http(s) URL or local filesystem path")

    @field_validator("uri")
    @classmethod
    def _uri_must_be_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("uri must not be blank")
        return stripped

    @property
    def is_remote(self) -> bool:
        return self.uri.startswith(("http://", "https://"))


class ReferenceImage(MediaAsset):
    """Still image of the person whose identity/outfit should be preserved."""


class DrivingVideo(MediaAsset):
    """Video whose motion the generated video should follow."""

    duration_seconds: float | None = Field(default=None, gt=0)


class CharacterOrientation(str, Enum):
    """Which input dictates the character's framing/orientation in the output."""

    IMAGE = "image"
    VIDEO = "video"


class IdentityElement(BaseModel):
    """Extra face imagery bound to the character, for identity consistency.

    ``frontal_image`` is the main head-on view; ``additional_images`` are the
    same face from other angles (1-3). At least one of the two must be given.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    frontal_image: ReferenceImage | None = None
    additional_images: tuple[ReferenceImage, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def _needs_at_least_one_image(self) -> IdentityElement:
        if self.frontal_image is None and not self.additional_images:
            raise ValueError(
                "identity element needs a frontal_image or at least one additional_images entry"
            )
        return self


class TransferRequest(BaseModel):
    """Everything a provider needs to start one job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_image: ReferenceImage
    driving_video: DrivingVideo
    prompt: str | None = Field(
        default=None,
        description="Optional text guidance, e.g. clothing or scene notes.",
    )
    character_orientation: CharacterOrientation = CharacterOrientation.VIDEO
    keep_original_sound: bool = True
    identity_element: IdentityElement | None = Field(
        default=None,
        description=(
            "Optional face binding for identity consistency. At most one element; "
            "providers may require character_orientation to be VIDEO."
        ),
    )


class JobStatus(str, Enum):
    """Lifecycle of a submitted job, normalized across providers."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset(
    {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}
)


class Job(BaseModel):
    """A submitted job and its last known status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


class ResultVideo(MediaAsset):
    """The generated video for a succeeded job."""

    duration_seconds: float | None = Field(default=None, gt=0)
    content_type: str = "video/mp4"
