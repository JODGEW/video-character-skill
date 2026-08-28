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

Note: this endpoint regenerates the scene from the reference image. For edits
that must keep the source video's own background and framing, see
:mod:`video_character_skill.providers.fal_kling_o1`.

Queue plumbing (submit/status/result, uploads, status mapping) is inherited
from :class:`~video_character_skill.providers._fal_queue.FalQueueProvider`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from video_character_skill.providers._fal_queue import FalQueueClient, FalQueueProvider
from video_character_skill.providers.base import ProviderError
from video_character_skill.schemas import (
    CharacterOrientation,
    IdentityElement,
    TransferRequest,
)

__all__ = ["APP_ID", "FalKlingProvider", "FalQueueClient"]

APP_ID = "fal-ai/kling-video/v3/standard/motion-control"


class FalKlingProvider(FalQueueProvider):
    """Kling V3 Standard Motion Control, via the fal queue API."""

    name: ClassVar[str] = "fal-kling-v3-standard-motion-control"
    app_id: ClassVar[str] = APP_ID

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
