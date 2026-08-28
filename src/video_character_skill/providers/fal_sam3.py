"""fal.ai provider for SAM 3 per-object video segmentation.

Endpoint: ``fal-ai/sam-3/video-rle-objects``.

Why this endpoint and not one of the other SAM 3 video endpoints
----------------------------------------------------------------
fal publishes four SAM 3 endpoints that could plausibly produce a mask. Only
one of them returns mask *data*; the rest return a rendered video, which is a
visualization, not a matte:

``fal-ai/sam-3/video``
    Output ``SAM3VideoOutput`` — ``video: File`` (X264 .mp4 or VP9 .webm) plus
    an optional ``boundingbox_frames_zip``. ``apply_mask`` (default ``true``)
    only changes what is drawn into that video; fal's own model page describes
    ``apply_mask: false`` as returning "raw segmentation data without visual
    overlays". Either way the result is a lossily-compressed video, so mask
    edges come back re-sampled and anti-aliased. No mask data in the response.

``fal-ai/sam-3/video-rle``
    Despite the name, its advertised output is the same ``SAM3VideoOutput``.
    The underlying app declares a union response
    (``SAM3VideoOutput | SAM3RLEOutput | SAM3RLEFileOutput``) with no
    documented rule for which branch is returned, and fal's description of it
    says it "collapses every tracked object into a single union mask per
    frame" — so even the RLE branch would not separate instances.

``fal-ai/sam-3/image-rle``
    Returns real RLE (``rle: string | list<string>``), but per image: no
    temporal tracking and no stable ids, so per-frame calls would give an
    unstable, flickering mask.

``fal-ai/sam-3/video-rle-objects`` (this one)
    fal's description, verbatim: "Per-object video segmentation: for each
    frame, one RLE mask per tracked object plus its stable track id
    (``out_obj_ids``). ``/video-rle`` collapses every tracked object into a
    single union mask per frame; this endpoint instead keeps each instance
    separable [...] One text-prompt tracking session runs over the whole video
    (no chunking), so track ids stay stable across all frames."

Input (``SAM3VideoObjectsInput``; required: ``video_url``)::

    video_url            required  "The URL of the video to segment into
                                    per-object tracked masks."
    prompt               optional  default "". "Text prompt describing the
                                    concept to track (e.g. 'person'). Treated
                                    as a single concept; every instance SAM-3
                                    detects is tracked with its own stable
                                    track id across frames."
    detection_threshold  optional  default 0.5, range 0.01-1.0.

There is no ``apply_mask`` and no ``video_output_type``: this endpoint has no
video output to apply anything to.

Output (``SAM3VideoObjectsOutput``; all four fields required)::

    frames      list<SAM3VideoObjectFrame>  per-frame, per-object RLE masks
    width       integer                     mask width  (RLE decode dimension)
    height      integer                     mask height (RLE decode dimension)
    num_frames  integer                     number of frames processed

    SAM3VideoObjectFrame: frame_index (int, required)
                          objects     (list<SAM3ObjectMask>, empty when none)
    SAM3ObjectMask:       track_id    (int, "stable object/track id")
                          rle         (str, "run-length encoding
                                       (Kaggle/COCO order) of the mask")

Note this endpoint is absent from fal's public model index and from the
generated ``@fal-ai/client`` typings; it is published only through its queue
OpenAPI schema and its own playground page. Its pricing is therefore not
listed alongside the other SAM 3 endpoints.

Queue plumbing (submit/status, uploads, status mapping) is inherited from
:class:`~video_character_skill.providers._fal_queue.FalQueueBase`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import ValidationError

from video_character_skill.providers._fal_queue import FalQueueBase, FalQueueClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.schemas import Job, SegmentationRequest, VideoMaskTrack

__all__ = [
    "APP_ID",
    "FalQueueClient",
    "FalSam3VideoMaskProvider",
]

APP_ID = "fal-ai/sam-3/video-rle-objects"


class FalSam3VideoMaskProvider(FalQueueBase):
    """SAM 3 video segmentation: one text concept in, per-frame masks out."""

    name: ClassVar[str] = "fal-sam-3-video-rle-objects"
    app_id: ClassVar[str] = APP_ID

    def build_arguments(self, request: SegmentationRequest) -> dict[str, Any]:
        """Build the fal request body, uploading the video if it is local."""
        return {
            "video_url": self._resolve_url(request.driving_video),
            "prompt": request.prompt,
            "detection_threshold": request.detection_threshold,
        }

    def submit(self, request: SegmentationRequest) -> Job:
        """Start a segmentation job. The returned job carries its id."""
        return self._enqueue(self.build_arguments(request))

    def get_result(self, job_id: str) -> VideoMaskTrack:
        """Return the per-frame masks of a succeeded job.

        Raises:
            ProviderError: if the job has not succeeded, or if fal's result
                does not match the documented output schema.
        """
        payload = self._result_payload(job_id)
        try:
            return VideoMaskTrack.model_validate(payload)
        except ValidationError as exc:
            raise ProviderError(f"fal result for {job_id} is not a mask track: {exc}") from exc
