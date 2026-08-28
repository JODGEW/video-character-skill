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


class VideoEditRequest(TransferRequest):
    """A transfer performed as an *edit* of the source video.

    Same inputs as :class:`TransferRequest`, plus extra appearance references.
    The source video's background, framing and motion are meant to survive; only
    the person is replaced. ``character_orientation`` is inherited but has no
    meaning for edit endpoints.
    """

    style_images: tuple[ReferenceImage, ...] = Field(
        default=(),
        description=(
            "Extra images showing clothing/appearance to imitate, distinct from "
            "the identity binding. Providers cap how many they accept."
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


class SegmentationRequest(BaseModel):
    """Track one concept through a video and return its per-frame masks.

    ``prompt`` names a single concept, not one instance: SAM 3 tracks every
    instance it detects and gives each its own track id. For our pipeline the
    concept is ``"person"`` and we expect exactly one track.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    driving_video: DrivingVideo
    prompt: str = Field(default="person", min_length=1)
    detection_threshold: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Lower finds more instances but less precisely.",
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_must_be_non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must not be blank")
        return stripped


class ObjectMask(BaseModel):
    """One tracked object's mask in one frame, run-length encoded.

    ``rle`` is decoded against the :class:`VideoMaskTrack` dimensions, not
    against anything carried here.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    track_id: int
    rle: str = Field(min_length=1)


class MaskFrame(BaseModel):
    """The masks present in a single frame of the source video."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    frame_index: int = Field(ge=0)
    objects: tuple[ObjectMask, ...] = ()


class VideoMaskTrack(BaseModel):
    """Per-frame, per-object masks for one segmented video.

    The temporal mask a masked edit needs: every frame carries the mask of each
    tracked object, and a track id stays with the same object across the whole
    clip. ``width``/``height`` are the dimensions the RLE decodes to.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    frames: tuple[MaskFrame, ...]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_frames: int = Field(ge=0)

    @property
    def track_ids(self) -> tuple[int, ...]:
        """Every track id seen anywhere in the clip, ascending.

        One id means one tracked subject — what a single-person edit needs.
        """
        return tuple(sorted({mask.track_id for frame in self.frames for mask in frame.objects}))


class SourceVideo(MediaAsset):
    """A video to be processed as-is, e.g. matted.

    Distinct from :class:`DrivingVideo`, which is a *motion reference*. The
    matting input is whatever footage the matte must line up with — for this
    pipeline, the edited video the compositor will cut the person out of.
    """


class MatteCodec(str, Enum):
    """How a matting endpoint should encode its output.

    ``VP9`` returns one WebM carrying a real alpha channel. ``H264`` returns
    two separate videos (rgb and alpha); fal does not document which comes
    first, so the provider refuses to guess. See
    :mod:`video_character_skill.providers.fal_veed_matting`.
    """

    VP9 = "vp9"
    H264 = "h264"


class MattingRequest(BaseModel):
    """Everything a matting provider needs to cut a subject out of a video."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_video: SourceVideo
    subject_is_person: bool = True
    refine_foreground_edges: bool = True
    output_codec: MatteCodec = MatteCodec.VP9


class MatteVideo(MediaAsset):
    """A video whose alpha channel is the subject matte.

    ``content_type`` is whatever the provider reported, unchanged — for VP9
    that is ``video/webm``. The alpha lives in the file's own alpha channel,
    so a decoder must be asked for it explicitly (e.g. ffmpeg ``-pix_fmt
    rgba``); decoding to a 3-channel format silently drops the matte.
    """

    content_type: str | None = None
