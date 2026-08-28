"""fal.ai provider for VEED video background removal (person matting).

Endpoint: ``veed/video-background-removal``

This is the matte source for the masked pipeline. SAM 3's
``video-rle-objects`` was tried first and rejected: across two real runs its
masks covered 53-82 % of the frame with a bounding box pinned to three frame
borders, which is not a person matte. VEED returns an alpha channel directly,
so there is no RLE to decode and no mask semantics to infer.

Input (``GeneralRembgInput``; required: ``video_url``)::

    video_url                required  string, format uri, 1-2083 chars
    output_codec             optional  default "vp9", enum {"vp9", "h264"}.
                                       "Single VP9 video with alpha channel or
                                       two videos (rgb and alpha) in H264
                                       format. H264 is recommended for better
                                       RGB quality."
    refine_foreground_edges  optional  default true. "Improves the quality of
                                       the extracted object's edges."
    subject_is_person        optional  default true. "Set to False if the
                                       subject is not a person."

Output (``GeneralRembgOutput``; required: ``video``)::

    video   list<File>

Why VP9 and not H264
--------------------
``video`` is a *list*, and its ordering is the whole question. Under VP9 the
list holds exactly one file — a WebM carrying the alpha channel — so
``video[0]`` is unambiguous. Under H264 it holds two, "rgb and alpha", and
**which one comes first is documented nowhere**: not in the endpoint's queue
OpenAPI, not on its API docs page, not on its model page, and not in the
generated ``@fal-ai/client`` typings, which type it as a bare ``File[]``. All
three VEED variants (standard, ``/fast``, ``/green-screen``) publish exactly
one output example, and all three show the single-element VP9 case.

Guessing that order would risk feeding RGB to the compositor as a matte — a
failure that produces plausible-looking output rather than an error. fal's
stated reason to prefer H264 is "better RGB quality", and we do not use VEED's
RGB at all: the footage we composite is the same clip we send in. So H264's
only documented advantage does not apply to us, while its ambiguity does.

Accordingly :meth:`FalVeedMattingProvider.get_result` requires exactly one
returned file. Requesting ``MatteCodec.H264`` will submit fine and then fail at
result time, by design — supporting it needs the ordering established
empirically first, not assumed.

Note the alpha rides in the file's own alpha channel: a decoder must be asked
for it (ffmpeg ``-pix_fmt rgba``), or the matte is silently dropped.

Pricing: fal lists $0.0225 per 30 frames with ``refine_foreground_edges`` on,
$0.015 with it off.

Queue plumbing (submit/status, uploads, status mapping) is inherited from
:class:`~video_character_skill.providers._fal_queue.FalQueueBase`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from video_character_skill.providers._fal_queue import FalQueueBase, FalQueueClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.schemas import Job, MatteVideo, MattingRequest

__all__ = [
    "APP_ID",
    "MAX_VIDEO_URL_CHARS",
    "FalQueueClient",
    "FalVeedMattingProvider",
]

APP_ID = "veed/video-background-removal"

# From the endpoint's OpenAPI spec, enforced locally so a bad request fails
# before it is billed.
MAX_VIDEO_URL_CHARS = 2083


class FalVeedMattingProvider(FalQueueBase):
    """VEED background removal: one video in, one alpha-bearing video out."""

    name: ClassVar[str] = "fal-veed-video-background-removal"
    app_id: ClassVar[str] = APP_ID

    def build_arguments(self, request: MattingRequest) -> dict[str, Any]:
        """Build the VEED request body, uploading the video if it is local.

        Every field is sent explicitly rather than left to the endpoint's
        defaults, so a change on fal's side cannot silently alter our matte.
        """
        video_url = self._resolve_url(request.source_video)
        if len(video_url) > MAX_VIDEO_URL_CHARS:
            raise ProviderError(
                f"video_url is {len(video_url)} chars; the maximum is {MAX_VIDEO_URL_CHARS}"
            )
        return {
            "video_url": video_url,
            "output_codec": request.output_codec.value,
            "refine_foreground_edges": request.refine_foreground_edges,
            "subject_is_person": request.subject_is_person,
        }

    def submit(self, request: MattingRequest) -> Job:
        """Start a matting job. The returned job carries its id."""
        return self._enqueue(self.build_arguments(request))

    def get_result(self, job_id: str) -> MatteVideo:
        """Return the alpha-bearing video of a succeeded job.

        Raises:
            ProviderError: if the job has not succeeded, if fal's result does
                not match the documented output schema, or if it holds more
                than one file (the H264 case, whose rgb/alpha order fal does
                not document — see the module docstring).
        """
        payload = self._result_payload(job_id)
        videos = payload.get("video") if isinstance(payload, dict) else None
        if not isinstance(videos, list) or not videos:
            raise ProviderError(f"fal result for {job_id} has no video list: {payload!r}")
        if len(videos) != 1:
            raise ProviderError(
                f"fal result for {job_id} holds {len(videos)} files; only the "
                "single-file vp9 output is supported, because fal does not document "
                "which of the two h264 files is rgb and which is alpha"
            )
        return self._to_matte_video(job_id, videos[0])

    @staticmethod
    def _to_matte_video(job_id: str, entry: Any) -> MatteVideo:
        if not isinstance(entry, dict):
            raise ProviderError(f"fal result for {job_id} is not a file object: {entry!r}")
        url = entry.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ProviderError(f"fal result for {job_id} has no video url: {entry!r}")
        content_type = entry.get("content_type")
        if isinstance(content_type, str) and content_type.strip():
            return MatteVideo(uri=url, content_type=content_type)
        return MatteVideo(uri=url)
