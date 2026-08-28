"""Unit tests for the Kling O1 video-edit provider. The fal client is faked, so
no network request occurs, but the real ``fal_client`` status/error types are
used so the queue mapping is checked against the actual library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fal_client
import pytest
from pydantic import ValidationError

from fal_fakes import COMPLETED_OK, SUCCESS_PAYLOAD, FakeFalClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.providers.fal_kling_o1 import (
    APP_ID,
    MAX_ELEMENT_REFERENCE_IMAGES,
    MAX_PROMPT_CHARS,
    FalKlingO1EditProvider,
)
from video_character_skill.schemas import (
    DrivingVideo,
    IdentityElement,
    JobStatus,
    ReferenceImage,
    TransferRequest,
    VideoEditRequest,
)

DEFAULT_PROMPT = FalKlingO1EditProvider.default_prompt()


def edit_request(**overrides: Any) -> VideoEditRequest:
    fields: dict[str, Any] = {
        "reference_image": ReferenceImage(uri="https://example.com/ref.png"),
        "driving_video": DrivingVideo(uri="https://example.com/source.mp4"),
    }
    fields.update(overrides)
    return VideoEditRequest(**fields)


# -- request payload ----------------------------------------------------


def test_default_payload_matches_o1_schema() -> None:
    fake = FakeFalClient()
    provider = FalKlingO1EditProvider(client=fake)

    provider.submit(edit_request(driving_video=DrivingVideo(
        uri="https://example.com/source.mp4", duration_seconds=9.5
    )))

    application, arguments = fake.submissions[0]
    assert application == APP_ID == "fal-ai/kling-video/o1/video-to-video/edit"
    assert arguments == {
        "prompt": DEFAULT_PROMPT,
        "video_url": "https://example.com/source.mp4",
        "keep_audio": True,
        "elements": [
            {
                "frontal_image_url": "https://example.com/ref.png",
                "reference_image_urls": ["https://example.com/ref.png"],
            }
        ],
    }
    assert "image_urls" not in arguments
    assert "image_url" not in arguments


def test_default_prompt_states_the_intended_edit() -> None:
    assert "@Element1" in DEFAULT_PROMPT
    assert "Replace the person" in DEFAULT_PROMPT
    for preserved in ("background", "camera framing", "camera movement", "motion"):
        assert preserved in DEFAULT_PROMPT
    assert len(DEFAULT_PROMPT) <= MAX_PROMPT_CHARS


def test_explicit_prompt_overrides_the_default() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(edit_request(prompt="swap to @Element1"))

    _, arguments = fake.submissions[0]
    assert arguments["prompt"] == "swap to @Element1"


def test_keep_original_sound_maps_to_keep_audio() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(edit_request(keep_original_sound=False))

    _, arguments = fake.submissions[0]
    assert arguments["keep_audio"] is False


def test_style_images_become_image_urls_and_are_referenced_in_the_prompt() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            style_images=(
                ReferenceImage(uri="https://example.com/outfit-a.png"),
                ReferenceImage(uri="https://example.com/outfit-b.png"),
            )
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["image_urls"] == [
        "https://example.com/outfit-a.png",
        "https://example.com/outfit-b.png",
    ]
    assert "@Image1, @Image2" in arguments["prompt"]


def test_plain_transfer_request_is_accepted_without_style_images() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        TransferRequest(
            reference_image=ReferenceImage(uri="https://example.com/ref.png"),
            driving_video=DrivingVideo(uri="https://example.com/source.mp4"),
        )
    )

    _, arguments = fake.submissions[0]
    assert "image_urls" not in arguments
    assert arguments["prompt"] == DEFAULT_PROMPT


# -- binding the reference person --------------------------------------


def test_identity_element_angles_ride_along_with_the_reference_image() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            identity_element=IdentityElement(
                additional_images=(
                    ReferenceImage(uri="https://example.com/face-left.png"),
                    ReferenceImage(uri="https://example.com/face-right.png"),
                )
            )
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["elements"] == [
        {
            "frontal_image_url": "https://example.com/ref.png",
            "reference_image_urls": [
                "https://example.com/face-left.png",
                "https://example.com/face-right.png",
            ],
        }
    ]


def test_explicit_frontal_image_overrides_the_reference_image() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            identity_element=IdentityElement(
                frontal_image=ReferenceImage(uri="https://example.com/face.png")
            )
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["elements"] == [
        {
            "frontal_image_url": "https://example.com/face.png",
            "reference_image_urls": ["https://example.com/face.png"],
        }
    ]


# -- uploads ------------------------------------------------------------


@pytest.fixture
def local_media(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "reference_image.png"
    image.write_bytes(b"\x89PNG fake")
    video = tmp_path / "driving_video_o1.mp4"
    video.write_bytes(b"fake mp4")
    return image, video


def test_local_video_and_reference_image_are_uploaded(local_media: tuple[Path, Path]) -> None:
    image, video = local_media
    fake = FakeFalClient()

    arguments = FalKlingO1EditProvider(client=fake).build_arguments(
        edit_request(
            reference_image=ReferenceImage(uri=str(image)),
            driving_video=DrivingVideo(uri=str(video), duration_seconds=9.5),
        )
    )

    assert fake.uploads == [str(video), str(image)]
    assert arguments["video_url"] == "https://v3.fal.media/files/driving_video_o1.mp4"
    assert arguments["elements"][0]["frontal_image_url"] == (
        "https://v3.fal.media/files/reference_image.png"
    )
    # the frontal image doubles as the single reference entry, uploaded once
    assert arguments["elements"][0]["reference_image_urls"] == [
        "https://v3.fal.media/files/reference_image.png"
    ]
    assert fake.uploads.count(str(image)) == 1


def test_local_element_and_style_images_are_uploaded(
    local_media: tuple[Path, Path], tmp_path: Path
) -> None:
    image, video = local_media
    angle = tmp_path / "face_left.png"
    angle.write_bytes(b"\x89PNG angle")
    outfit = tmp_path / "outfit.png"
    outfit.write_bytes(b"\x89PNG outfit")

    fake = FakeFalClient()
    arguments = FalKlingO1EditProvider(client=fake).build_arguments(
        edit_request(
            reference_image=ReferenceImage(uri=str(image)),
            driving_video=DrivingVideo(uri=str(video)),
            identity_element=IdentityElement(additional_images=(ReferenceImage(uri=str(angle)),)),
            style_images=(ReferenceImage(uri=str(outfit)),),
        )
    )

    assert fake.uploads == [str(video), str(image), str(angle), str(outfit)]
    assert arguments["elements"][0]["reference_image_urls"] == [
        "https://v3.fal.media/files/face_left.png"
    ]
    assert arguments["image_urls"] == ["https://v3.fal.media/files/outfit.png"]


def test_remote_urls_are_passed_through_without_upload() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).build_arguments(
        edit_request(style_images=(ReferenceImage(uri="https://example.com/outfit.png"),))
    )

    assert fake.uploads == []


def test_missing_local_file_raises_provider_error(tmp_path: Path) -> None:
    fake = FakeFalClient()
    with pytest.raises(ProviderError, match="local file not found"):
        FalKlingO1EditProvider(client=fake).build_arguments(
            edit_request(driving_video=DrivingVideo(uri=str(tmp_path / "nope.mp4")))
        )


# -- local validation ---------------------------------------------------


@pytest.mark.parametrize("duration", [2.9, 10.06, 30.0])
def test_out_of_range_duration_fails_before_upload(duration: float) -> None:
    fake = FakeFalClient()
    provider = FalKlingO1EditProvider(client=fake)
    request = edit_request(
        driving_video=DrivingVideo(uri="https://example.com/source.mp4", duration_seconds=duration)
    )

    with pytest.raises(ProviderError, match="3.0-10.05s"):
        provider.submit(request)

    assert fake.uploads == []
    assert fake.submissions == []


@pytest.mark.parametrize("duration", [3.0, 9.5, 10.05])
def test_in_range_duration_is_accepted(duration: float) -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            driving_video=DrivingVideo(
                uri="https://example.com/source.mp4", duration_seconds=duration
            )
        )
    )
    assert len(fake.submissions) == 1


def test_unknown_duration_is_not_blocked() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(edit_request())
    assert len(fake.submissions) == 1


def test_too_many_references_fails_before_upload() -> None:
    fake = FakeFalClient()
    provider = FalKlingO1EditProvider(client=fake)
    request = edit_request(
        style_images=tuple(
            ReferenceImage(uri=f"https://example.com/{i}.png") for i in range(4)
        )
    )

    with pytest.raises(ProviderError, match="exceeds the maximum of 4"):
        provider.submit(request)

    assert fake.uploads == []
    assert fake.submissions == []


def test_three_style_images_plus_one_element_is_the_limit() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            style_images=tuple(
                ReferenceImage(uri=f"https://example.com/{i}.png") for i in range(3)
            )
        )
    )

    _, arguments = fake.submissions[0]
    assert len(arguments["elements"]) + len(arguments["image_urls"]) == 4


@pytest.mark.parametrize("prompt", ["x" * (MAX_PROMPT_CHARS + 1), "   "])
def test_invalid_prompt_fails_before_upload(prompt: str) -> None:
    fake = FakeFalClient()
    provider = FalKlingO1EditProvider(client=fake)

    with pytest.raises(ProviderError):
        provider.submit(edit_request(prompt=prompt))

    assert fake.uploads == []
    assert fake.submissions == []


# -- queue status and result -------------------------------------------


@pytest.mark.parametrize(
    ("fal_status", "expected"),
    [
        (fal_client.Queued(position=2), JobStatus.QUEUED),
        (fal_client.InProgress(logs=[]), JobStatus.RUNNING),
        (COMPLETED_OK, JobStatus.SUCCEEDED),
    ],
)
def test_queue_state_mapping(fal_status: fal_client.Status, expected: JobStatus) -> None:
    job = FalKlingO1EditProvider(client=FakeFalClient(status_response=fal_status)).get_status("r1")

    assert job.status is expected
    assert job.provider == "fal-kling-o1-video-to-video-edit"


def test_completed_with_error_maps_to_failed() -> None:
    fake = FakeFalClient(
        status_response=fal_client.Completed(
            logs=[], metrics={}, error="video too long", error_type="ValidationError"
        )
    )
    job = FalKlingO1EditProvider(client=fake).get_status("r1")

    assert job.status is JobStatus.FAILED
    assert job.error == "ValidationError: video too long"


def test_submit_returns_queued_job() -> None:
    job = FalKlingO1EditProvider(client=FakeFalClient()).submit(edit_request())

    assert job.job_id == "req-1"
    assert job.status is JobStatus.QUEUED


def test_result_video_is_parsed() -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK)
    video = FalKlingO1EditProvider(client=fake).get_result("r1")

    assert fake.result_calls == [(APP_ID, "r1")]
    assert video.uri == SUCCESS_PAYLOAD["video"]["url"]
    assert video.content_type == "video/mp4"


def test_malformed_result_raises_provider_error() -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response={"video": {}})
    with pytest.raises(ProviderError, match="no video url"):
        FalKlingO1EditProvider(client=fake).get_result("r1")


def test_get_result_refuses_unfinished_job() -> None:
    fake = FakeFalClient(status_response=fal_client.InProgress(logs=[]))
    with pytest.raises(ProviderError, match="not succeeded"):
        FalKlingO1EditProvider(client=fake).get_result("r1")
    assert fake.result_calls == []


# -- API failures -------------------------------------------------------


def test_submit_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("401 unauthorized"))
    with pytest.raises(ProviderError, match="fal submit failed"):
        FalKlingO1EditProvider(client=fake).submit(edit_request())


def test_status_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("boom"))
    with pytest.raises(ProviderError, match="fal status failed"):
        FalKlingO1EditProvider(client=fake).get_status("r1")


def test_result_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(
        status_response=COMPLETED_OK,
        result_error=fal_client.FalClientError("504 gateway timeout"),
    )
    with pytest.raises(ProviderError, match="fal result failed"):
        FalKlingO1EditProvider(client=fake).get_result("r1")


def test_upload_failure_becomes_provider_error(local_media: tuple[Path, Path]) -> None:
    image, video = local_media
    fake = FakeFalClient(error=fal_client.FalClientError("storage down"))
    with pytest.raises(ProviderError, match="fal upload failed"):
        FalKlingO1EditProvider(client=fake).build_arguments(
            edit_request(
                reference_image=ReferenceImage(uri=str(image)),
                driving_video=DrivingVideo(uri=str(video)),
            )
        )


# -- regression: elementReferList size must be between 1 and 3 ----------
#
# Job 01a046a5-c69b-7272-a6e8-a924a8c36b36 was reported Completed by the queue,
# then failed at result() with HTTP 422 "elementReferList: size must be between
# 1 and 3". Its element carried a frontal_image_url with reference_image_urls
# omitted. The published OpenAPI schema marks that field optional; the backend
# does not.


def test_single_reference_image_never_yields_an_empty_reference_list() -> None:
    """The exact request shape that produced the 422."""
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        VideoEditRequest(
            reference_image=ReferenceImage(uri="https://example.com/reference_image.png"),
            driving_video=DrivingVideo(
                uri="https://example.com/driving_video_o1.mp4", duration_seconds=9.5
            ),
        )
    )

    _, arguments = fake.submissions[0]
    element = arguments["elements"][0]
    assert element["reference_image_urls"] == ["https://example.com/reference_image.png"]
    assert element["reference_image_urls"] != []
    assert len(element["reference_image_urls"]) == 1


@pytest.mark.parametrize(
    "request_builder",
    [
        pytest.param(lambda: edit_request(), id="no-element"),
        pytest.param(
            lambda: edit_request(
                identity_element=IdentityElement(
                    frontal_image=ReferenceImage(uri="https://example.com/face.png")
                )
            ),
            id="element-frontal-only",
        ),
        pytest.param(
            lambda: edit_request(
                identity_element=IdentityElement(
                    additional_images=(ReferenceImage(uri="https://example.com/a.png"),)
                )
            ),
            id="element-angles-only",
        ),
        pytest.param(
            lambda: edit_request(
                style_images=(ReferenceImage(uri="https://example.com/outfit.png"),)
            ),
            id="with-style-images",
        ),
    ],
)
def test_reference_image_urls_is_always_present_and_sized_one_to_three(
    request_builder: Any,
) -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(request_builder())

    _, arguments = fake.submissions[0]
    element = arguments["elements"][0]
    assert "reference_image_urls" in element
    assert 1 <= len(element["reference_image_urls"]) <= 3


def test_explicit_additional_images_are_used_as_the_reference_list() -> None:
    fake = FakeFalClient()
    FalKlingO1EditProvider(client=fake).submit(
        edit_request(
            identity_element=IdentityElement(
                additional_images=(
                    ReferenceImage(uri="https://example.com/face-left.png"),
                    ReferenceImage(uri="https://example.com/face-right.png"),
                    ReferenceImage(uri="https://example.com/face-up.png"),
                )
            )
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["elements"][0] == {
        "frontal_image_url": "https://example.com/ref.png",
        "reference_image_urls": [
            "https://example.com/face-left.png",
            "https://example.com/face-right.png",
            "https://example.com/face-up.png",
        ],
    }


def test_more_than_three_reference_images_fails_before_upload_or_submit() -> None:
    fake = FakeFalClient()
    angles = tuple(ReferenceImage(uri=f"https://example.com/face-{i}.png") for i in range(4))

    with pytest.raises(ValidationError):
        edit_request(identity_element=IdentityElement(additional_images=angles))

    assert fake.uploads == []
    assert fake.submissions == []


def test_provider_also_guards_the_reference_list_size() -> None:
    """Defence in depth: the provider enforces the rule independently of the model."""
    with pytest.raises(ProviderError, match="elementReferList"):
        FalKlingO1EditProvider._check_reference_images(MAX_ELEMENT_REFERENCE_IMAGES + 1)

    FalKlingO1EditProvider._check_reference_images(MAX_ELEMENT_REFERENCE_IMAGES)
    FalKlingO1EditProvider._check_reference_images(0)  # frontal-image fallback
