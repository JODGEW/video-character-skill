"""Unit tests for the fal provider. No network call is ever made: the fal client
is replaced by an in-memory fake, but the real ``fal_client`` status/error types
are used so the mapping is checked against the actual library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fal_client
import pytest

from fal_fakes import COMPLETED_OK, FakeFalClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.providers.fal_kling import APP_ID, FalKlingProvider
from video_character_skill.schemas import (
    CharacterOrientation,
    DrivingVideo,
    IdentityElement,
    JobStatus,
    ReferenceImage,
    TransferRequest,
)


@pytest.fixture
def local_files(tmp_path: Path) -> tuple[Path, Path]:
    image = tmp_path / "reference_image.png"
    image.write_bytes(b"\x89PNG fake")
    video = tmp_path / "driving_video.mov"
    video.write_bytes(b"fake mov")
    return image, video


def remote_request(**overrides: Any) -> TransferRequest:
    fields: dict[str, Any] = {
        "reference_image": ReferenceImage(uri="https://example.com/ref.png"),
        "driving_video": DrivingVideo(uri="https://example.com/drive.mp4"),
    }
    fields.update(overrides)
    return TransferRequest(**fields)


# -- uploads ------------------------------------------------------------


def test_local_image_and_video_are_uploaded(local_files: tuple[Path, Path]) -> None:
    image, video = local_files
    fake = FakeFalClient()
    provider = FalKlingProvider(client=fake)

    arguments = provider.build_arguments(
        TransferRequest(
            reference_image=ReferenceImage(uri=str(image)),
            driving_video=DrivingVideo(uri=str(video)),
        )
    )

    assert fake.uploads == [str(image), str(video)]
    assert arguments["image_url"] == "https://v3.fal.media/files/reference_image.png"
    assert arguments["video_url"] == "https://v3.fal.media/files/driving_video.mov"


def test_remote_urls_are_passed_through_without_upload() -> None:
    fake = FakeFalClient()
    arguments = FalKlingProvider(client=fake).build_arguments(remote_request())

    assert fake.uploads == []
    assert arguments["image_url"] == "https://example.com/ref.png"
    assert arguments["video_url"] == "https://example.com/drive.mp4"


def test_mixed_local_image_and_remote_video(local_files: tuple[Path, Path]) -> None:
    image, _ = local_files
    fake = FakeFalClient()

    arguments = FalKlingProvider(client=fake).build_arguments(
        TransferRequest(
            reference_image=ReferenceImage(uri=str(image)),
            driving_video=DrivingVideo(uri="https://example.com/drive.mp4"),
        )
    )

    assert fake.uploads == [str(image)]
    assert arguments["video_url"] == "https://example.com/drive.mp4"


def test_missing_local_file_raises_provider_error(tmp_path: Path) -> None:
    provider = FalKlingProvider(client=FakeFalClient())
    request = TransferRequest(
        reference_image=ReferenceImage(uri=str(tmp_path / "nope.png")),
        driving_video=DrivingVideo(uri="https://example.com/drive.mp4"),
    )
    with pytest.raises(ProviderError, match="local file not found"):
        provider.build_arguments(request)


# -- request payload ----------------------------------------------------


def test_payload_matches_kling_schema() -> None:
    fake = FakeFalClient()
    provider = FalKlingProvider(client=fake)

    provider.submit(
        remote_request(
            prompt="same outfit, studio lighting",
            character_orientation=CharacterOrientation.IMAGE,
            keep_original_sound=False,
        )
    )

    application, arguments = fake.submissions[0]
    assert application == APP_ID == "fal-ai/kling-video/v3/standard/motion-control"
    assert arguments == {
        "image_url": "https://example.com/ref.png",
        "video_url": "https://example.com/drive.mp4",
        "character_orientation": "image",
        "keep_original_sound": False,
        "prompt": "same outfit, studio lighting",
    }


def test_payload_defaults_and_omits_absent_prompt() -> None:
    fake = FakeFalClient()
    FalKlingProvider(client=fake).submit(remote_request())

    _, arguments = fake.submissions[0]
    assert arguments == {
        "image_url": "https://example.com/ref.png",
        "video_url": "https://example.com/drive.mp4",
        "character_orientation": "video",
        "keep_original_sound": True,
    }
    assert "prompt" not in arguments


def test_submit_returns_queued_job_with_fal_request_id() -> None:
    job = FalKlingProvider(client=FakeFalClient()).submit(remote_request())

    assert job.job_id == "req-1"
    assert job.provider == "fal-kling-v3-standard-motion-control"
    assert job.status is JobStatus.QUEUED


# -- status mapping -----------------------------------------------------


@pytest.mark.parametrize(
    ("fal_status", "expected"),
    [
        (fal_client.Queued(position=3), JobStatus.QUEUED),
        (fal_client.InProgress(logs=[]), JobStatus.RUNNING),
        (COMPLETED_OK, JobStatus.SUCCEEDED),
    ],
)
def test_queue_state_mapping(fal_status: fal_client.Status, expected: JobStatus) -> None:
    provider = FalKlingProvider(client=FakeFalClient(status_response=fal_status))
    job = provider.get_status("req-1")

    assert job.status is expected
    assert job.job_id == "req-1"
    assert job.error is None


def test_completed_with_error_maps_to_failed() -> None:
    completed = fal_client.Completed(
        logs=[], metrics={}, error="content policy", error_type="ValidationError"
    )
    job = FalKlingProvider(client=FakeFalClient(status_response=completed)).get_status("req-1")

    assert job.status is JobStatus.FAILED
    assert job.is_terminal
    assert job.error == "ValidationError: content policy"


def test_unrecognized_status_raises_provider_error() -> None:
    provider = FalKlingProvider(client=FakeFalClient(status_response=fal_client.Status()))
    with pytest.raises(ProviderError, match="unrecognized fal status"):
        provider.get_status("req-1")


# -- result parsing -----------------------------------------------------


def test_result_video_is_parsed_from_fal_payload() -> None:
    fake = FakeFalClient(
        status_response=COMPLETED_OK
    )
    video = FalKlingProvider(client=fake).get_result("req-1")

    assert fake.result_calls == [(APP_ID, "req-1")]
    assert video.uri == "https://v3.fal.media/files/out.mp4"
    assert video.content_type == "video/mp4"
    assert video.is_remote


def test_result_falls_back_to_default_content_type() -> None:
    fake = FakeFalClient(
        status_response=COMPLETED_OK,
        result_response={"video": {"url": "https://v3.fal.media/files/out.mp4"}},
    )
    assert FalKlingProvider(client=fake).get_result("req-1").content_type == "video/mp4"


@pytest.mark.parametrize(
    "payload",
    [{}, {"video": None}, {"video": {}}, {"video": {"url": ""}}, "not-a-dict"],
)
def test_malformed_result_raises_provider_error(payload: Any) -> None:
    fake = FakeFalClient(
        status_response=COMPLETED_OK,
        result_response=payload,
    )
    with pytest.raises(ProviderError):
        FalKlingProvider(client=fake).get_result("req-1")


def test_get_result_refuses_unfinished_job() -> None:
    fake = FakeFalClient(status_response=fal_client.InProgress(logs=[]))
    with pytest.raises(ProviderError, match="not succeeded"):
        FalKlingProvider(client=fake).get_result("req-1")
    assert fake.result_calls == []


def test_get_result_on_failed_job_reports_upstream_error() -> None:
    fake = FakeFalClient(
        status_response=fal_client.Completed(
            logs=[], metrics={}, error="content policy", error_type="ValidationError"
        )
    )
    with pytest.raises(ProviderError, match="content policy"):
        FalKlingProvider(client=fake).get_result("req-1")


# -- API failures -------------------------------------------------------


def test_submit_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("401 unauthorized"))
    with pytest.raises(ProviderError, match="fal submit failed"):
        FalKlingProvider(client=fake).submit(remote_request())


def test_result_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(
        status_response=COMPLETED_OK,
        result_error=fal_client.FalClientError("504 gateway timeout"),
    )
    with pytest.raises(ProviderError, match="fal result failed"):
        FalKlingProvider(client=fake).get_result("req-1")


def test_status_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("boom"))
    with pytest.raises(ProviderError, match="fal status failed"):
        FalKlingProvider(client=fake).get_status("req-1")


def test_upload_failure_becomes_provider_error(local_files: tuple[Path, Path]) -> None:
    image, video = local_files
    fake = FakeFalClient(error=fal_client.FalClientError("storage down"))
    provider = FalKlingProvider(client=fake)
    request = TransferRequest(
        reference_image=ReferenceImage(uri=str(image)),
        driving_video=DrivingVideo(uri=str(video)),
    )
    with pytest.raises(ProviderError, match="fal upload failed"):
        provider.build_arguments(request)


# -- element binding ----------------------------------------------------


def element_request(element: IdentityElement, **overrides: Any) -> TransferRequest:
    return remote_request(identity_element=element, **overrides)


def test_element_payload_matches_kling_schema() -> None:
    fake = FakeFalClient()
    provider = FalKlingProvider(client=fake)

    provider.submit(
        element_request(
            IdentityElement(
                frontal_image=ReferenceImage(uri="https://example.com/face-front.png"),
                additional_images=(
                    ReferenceImage(uri="https://example.com/face-left.png"),
                    ReferenceImage(uri="https://example.com/face-right.png"),
                ),
            ),
            prompt="@Element1 walking through a market",
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["elements"] == [
        {
            "frontal_image_url": "https://example.com/face-front.png",
            "reference_image_urls": [
                "https://example.com/face-left.png",
                "https://example.com/face-right.png",
            ],
        }
    ]
    assert arguments["character_orientation"] == "video"
    assert arguments["prompt"] == "@Element1 walking through a market"


def test_element_omits_absent_keys() -> None:
    fake = FakeFalClient()
    FalKlingProvider(client=fake).submit(
        element_request(
            IdentityElement(frontal_image=ReferenceImage(uri="https://example.com/face.png"))
        )
    )

    _, arguments = fake.submissions[0]
    assert arguments["elements"] == [{"frontal_image_url": "https://example.com/face.png"}]

    fake2 = FakeFalClient()
    FalKlingProvider(client=fake2).submit(
        element_request(
            IdentityElement(additional_images=(ReferenceImage(uri="https://example.com/a.png"),))
        )
    )
    _, arguments2 = fake2.submissions[0]
    assert arguments2["elements"] == [{"reference_image_urls": ["https://example.com/a.png"]}]


def test_local_element_images_are_uploaded(local_files: tuple[Path, Path], tmp_path: Path) -> None:
    image, _ = local_files
    face = tmp_path / "face_front.png"
    face.write_bytes(b"\x89PNG face")
    angle = tmp_path / "face_left.png"
    angle.write_bytes(b"\x89PNG angle")

    fake = FakeFalClient()
    arguments = FalKlingProvider(client=fake).build_arguments(
        TransferRequest(
            reference_image=ReferenceImage(uri=str(image)),
            driving_video=DrivingVideo(uri="https://example.com/drive.mp4"),
            identity_element=IdentityElement(
                frontal_image=ReferenceImage(uri=str(face)),
                additional_images=(ReferenceImage(uri=str(angle)),),
            ),
        )
    )

    assert fake.uploads == [str(image), str(face), str(angle)]
    assert arguments["elements"] == [
        {
            "frontal_image_url": "https://v3.fal.media/files/face_front.png",
            "reference_image_urls": ["https://v3.fal.media/files/face_left.png"],
        }
    ]


def test_remote_element_images_are_not_uploaded() -> None:
    fake = FakeFalClient()
    arguments = FalKlingProvider(client=fake).build_arguments(
        element_request(
            IdentityElement(
                frontal_image=ReferenceImage(uri="https://example.com/face.png"),
                additional_images=(ReferenceImage(uri="https://example.com/face-left.png"),),
            )
        )
    )

    assert fake.uploads == []
    assert arguments["elements"] == [
        {
            "frontal_image_url": "https://example.com/face.png",
            "reference_image_urls": ["https://example.com/face-left.png"],
        }
    ]


def test_element_with_image_orientation_fails_before_any_upload(tmp_path: Path) -> None:
    face = tmp_path / "face.png"
    face.write_bytes(b"\x89PNG face")
    fake = FakeFalClient()
    provider = FalKlingProvider(client=fake)
    request = element_request(
        IdentityElement(frontal_image=ReferenceImage(uri=str(face))),
        character_orientation=CharacterOrientation.IMAGE,
    )

    with pytest.raises(ProviderError, match="requires character_orientation='video'"):
        provider.submit(request)

    assert fake.uploads == []
    assert fake.submissions == []


def test_request_without_element_omits_the_key() -> None:
    fake = FakeFalClient()
    FalKlingProvider(client=fake).submit(remote_request())

    _, arguments = fake.submissions[0]
    assert "elements" not in arguments


def test_image_orientation_without_element_is_still_allowed() -> None:
    fake = FakeFalClient()
    FalKlingProvider(client=fake).submit(
        remote_request(character_orientation=CharacterOrientation.IMAGE)
    )

    _, arguments = fake.submissions[0]
    assert arguments["character_orientation"] == "image"
    assert "elements" not in arguments
