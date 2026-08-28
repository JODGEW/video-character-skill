"""fal.ai provider for Kling O1 video-to-video edit.

Endpoint: ``fal-ai/kling-video/o1/video-to-video/edit`` ("Modify a video based
on the prompt"). Unlike Motion Control, this edits the *source* video in place,
so its background, lighting, camera framing and motion survive; the prompt says
what to change. That is what we want: swap the person, keep the scene.

Input (verbatim from fal's OpenAPI spec; required: ``prompt``, ``video_url``)::

    prompt      required  max 2500 chars. "Use @Element1, @Element2 to reference
                          elements and @Image1, @Image2 to reference images in
                          order."
    video_url   required  ".mp4/.mov formats supported, 3-10 seconds duration,
                          720-2160px resolution, max 200MB" (min fps 24, max 60)
    keep_audio  optional  default false. "Whether to keep the original audio."
    image_urls  optional  list<str> | null. "Reference images for style/
                          appearance. Reference in prompt as @Image1, @Image2,
                          etc. Maximum 4 total (elements + reference images)."
    elements    optional  list<OmniVideoElementInput> | null. "Elements
                          (characters/objects) to include. Reference in prompt
                          as @Element1, @Element2, etc. Maximum 4 total."

``OmniVideoElementInput`` — note ``frontal_image_url`` is **required** here,
unlike the Motion Control variant of this type::

    frontal_image_url     required  "The frontal image of the element (main view)."
    reference_image_urls  optional  "Additional reference images from different
                                     angles. 1-3 images supported."

Observed backend constraint (NOT in the published OpenAPI schema)
-----------------------------------------------------------------
The spec marks ``reference_image_urls`` optional, but the backend rejects an
element without it. Job ``01a046a5-c69b-7272-a6e8-a924a8c36b36`` was reported
``Completed`` by the queue, and fetching its result returned HTTP 422::

    elementReferList: size must be between 1 and 3

That job's element had been submitted with ``frontal_image_url`` set and
``reference_image_urls`` *omitted* — so omitting the key is not a workaround;
the backend materializes it as an empty list and then enforces size 1-3.
:meth:`FalKlingO1EditProvider._element_payload` therefore always emits 1-3
entries, falling back to the frontal image itself when no other angle is given.

Note also that the failure surfaced only at ``result()``: the queue still
reported the job as completed, so a caller must not treat "completed" as
"succeeded" without fetching the result.

The reference person is bound as ``elements[0]`` and referenced from the prompt
as ``@Element1``; any :attr:`VideoEditRequest.style_images` become
``image_urls`` and are referenced as ``@Image1`` onward.
"""

from __future__ import annotations

from typing import Any, ClassVar

from video_character_skill.providers._fal_queue import FalQueueClient, FalQueueProvider
from video_character_skill.providers.base import ProviderError
from video_character_skill.schemas import (
    ReferenceImage,
    TransferRequest,
    VideoEditRequest,
)

__all__ = [
    "APP_ID",
    "MAX_DURATION_SECONDS",
    "MAX_ELEMENT_REFERENCE_IMAGES",
    "MIN_ELEMENT_REFERENCE_IMAGES",
    "MAX_PROMPT_CHARS",
    "MAX_REFERENCES",
    "MIN_DURATION_SECONDS",
    "FalKlingO1EditProvider",
    "FalQueueClient",
]

APP_ID = "fal-ai/kling-video/o1/video-to-video/edit"

# Limits taken from the endpoint's OpenAPI spec, enforced locally so a bad
# request fails before it is billed.
MIN_DURATION_SECONDS = 3.0
MAX_DURATION_SECONDS = 10.05
MAX_PROMPT_CHARS = 2500
MAX_REFERENCES = 4  # elements + image_urls combined

# Observed backend rule, absent from the published schema: an element's
# reference_image_urls must hold 1-3 entries. See the module docstring.
MIN_ELEMENT_REFERENCE_IMAGES = 1
MAX_ELEMENT_REFERENCE_IMAGES = 3


class FalKlingO1EditProvider(FalQueueProvider):
    """Kling O1 video edit: replace the person, keep the original scene."""

    name: ClassVar[str] = "fal-kling-o1-video-to-video-edit"
    app_id: ClassVar[str] = APP_ID

    def build_arguments(self, request: TransferRequest) -> dict[str, Any]:
        """Build the O1 request body, uploading any local files first.

        Raises:
            ProviderError: on a request fal would reject, checked before upload.
        """
        style_images = (
            request.style_images if isinstance(request, VideoEditRequest) else ()
        )
        self._check_duration(request)
        self._check_reference_count(len(style_images))
        prompt = request.prompt if request.prompt is not None else self.default_prompt(
            len(style_images)
        )
        self._check_prompt(prompt)

        arguments: dict[str, Any] = {
            "prompt": prompt,
            "video_url": self._resolve_url(request.driving_video),
            "keep_audio": request.keep_original_sound,
            "elements": [self._element_payload(request)],
        }
        if style_images:
            arguments["image_urls"] = [self._resolve_url(image) for image in style_images]
        return arguments

    # -- the reference person ------------------------------------------

    def _element_payload(self, request: TransferRequest) -> dict[str, Any]:
        """One ``OmniVideoElementInput`` describing the reference person.

        ``reference_image`` is the frontal (main) view unless an explicit
        identity element overrides it; the element's other angles ride along.

        ``reference_image_urls`` is always populated with 1-3 entries — see the
        observed backend constraint in the module docstring. With only one
        reference image available, the frontal image doubles as the single
        entry, reusing its already-resolved URL rather than uploading twice.
        """
        element = request.identity_element
        frontal: ReferenceImage = request.reference_image
        additional: tuple[ReferenceImage, ...] = ()
        if element is not None:
            if element.frontal_image is not None:
                frontal = element.frontal_image
            additional = element.additional_images

        self._check_reference_images(len(additional))
        frontal_url = self._resolve_url(frontal)
        reference_urls = (
            [self._resolve_url(image) for image in additional]
            if additional
            else [frontal_url]
        )
        return {
            "frontal_image_url": frontal_url,
            "reference_image_urls": reference_urls,
        }

    # -- the instruction -----------------------------------------------

    @staticmethod
    def default_prompt(style_image_count: int = 0) -> str:
        """The edit we actually want, phrased in fal's ``@`` reference syntax."""
        prompt = (
            "Replace the person in the video with @Element1. Keep @Element1's face, "
            "hairstyle, accessories and clothing style. Preserve the original video's "
            "background, scene, lighting, camera framing, camera movement and the "
            "original body motion exactly; change nothing except the person."
        )
        if style_image_count:
            references = ", ".join(f"@Image{i + 1}" for i in range(style_image_count))
            prompt += f" Match the clothing style shown in {references}."
        return prompt

    # -- local validation ----------------------------------------------

    @staticmethod
    def _check_duration(request: TransferRequest) -> None:
        duration = request.driving_video.duration_seconds
        if duration is None:
            return
        if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
            raise ProviderError(
                f"driving video is {duration}s; this endpoint accepts "
                f"{MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS}s"
            )

    @staticmethod
    def _check_reference_count(style_image_count: int) -> None:
        total = 1 + style_image_count  # always exactly one element
        if total > MAX_REFERENCES:
            raise ProviderError(
                f"{total} references (1 element + {style_image_count} images) exceeds "
                f"the maximum of {MAX_REFERENCES}"
            )

    @staticmethod
    def _check_reference_images(count: int) -> None:
        """Enforce the backend's ``elementReferList`` size rule.

        ``count`` is the number of *explicit* extra angles; zero is fine because
        the frontal image is used as the single entry in that case.
        """
        if count > MAX_ELEMENT_REFERENCE_IMAGES:
            raise ProviderError(
                f"{count} element reference images; the backend accepts "
                f"{MIN_ELEMENT_REFERENCE_IMAGES}-{MAX_ELEMENT_REFERENCE_IMAGES} "
                "(elementReferList)"
            )

    @staticmethod
    def _check_prompt(prompt: str) -> None:
        if not prompt.strip():
            raise ProviderError("prompt is required and must not be blank")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ProviderError(
                f"prompt is {len(prompt)} chars; the maximum is {MAX_PROMPT_CHARS}"
            )
