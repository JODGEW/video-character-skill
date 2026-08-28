from __future__ import annotations

import pytest
from pydantic import ValidationError

from video_character_skill.schemas import (
    CharacterOrientation,
    DrivingVideo,
    IdentityElement,
    Job,
    JobStatus,
    MaskFrame,
    ObjectMask,
    ReferenceImage,
    ResultVideo,
    SegmentationRequest,
    TransferRequest,
    VideoMaskTrack,
)


def test_uri_is_stripped_and_blank_is_rejected() -> None:
    assert ReferenceImage(uri="  ./reference_image.png  ").uri == "./reference_image.png"
    with pytest.raises(ValidationError):
        ReferenceImage(uri="   ")


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://example.com/a.png", True),
        ("http://example.com/a.png", True),
        ("./reference_image.png", False),
        ("/abs/path/reference_image.png", False),
    ],
)
def test_is_remote(uri: str, expected: bool) -> None:
    assert ReferenceImage(uri=uri).is_remote is expected


def test_driving_video_duration_must_be_positive() -> None:
    assert DrivingVideo(uri="./driving_video.mov", duration_seconds=5.0).duration_seconds == 5.0
    with pytest.raises(ValidationError):
        DrivingVideo(uri="./driving_video.mov", duration_seconds=0)


def test_assets_are_frozen_and_reject_unknown_fields() -> None:
    asset = ReferenceImage(uri="./reference_image.png")
    with pytest.raises(ValidationError):
        asset.uri = "./other.png"
    with pytest.raises(ValidationError):
        ReferenceImage(uri="./a.png", height=512)  # type: ignore[call-arg]


def test_transfer_request_defaults() -> None:
    request = TransferRequest(
        reference_image=ReferenceImage(uri="./reference_image.png"),
        driving_video=DrivingVideo(uri="./driving_video.mov"),
    )
    assert request.prompt is None
    assert request.character_orientation is CharacterOrientation.VIDEO
    assert request.keep_original_sound is True


def test_transfer_request_round_trips_through_json() -> None:
    request = TransferRequest(
        reference_image=ReferenceImage(uri="https://example.com/ref.png"),
        driving_video=DrivingVideo(uri="https://example.com/drive.mp4", duration_seconds=5.0),
        prompt="same outfit, studio lighting",
        character_orientation=CharacterOrientation.IMAGE,
        keep_original_sound=False,
    )
    assert TransferRequest.model_validate_json(request.model_dump_json()) == request


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (JobStatus.QUEUED, False),
        (JobStatus.RUNNING, False),
        (JobStatus.SUCCEEDED, True),
        (JobStatus.FAILED, True),
        (JobStatus.CANCELED, True),
    ],
)
def test_job_status_terminality(status: JobStatus, terminal: bool) -> None:
    assert status.is_terminal is terminal
    assert Job(job_id="j1", provider="fake", status=status).is_terminal is terminal


def test_job_defaults_to_queued_and_requires_ids() -> None:
    assert Job(job_id="j1", provider="fake").status is JobStatus.QUEUED
    with pytest.raises(ValidationError):
        Job(job_id="", provider="fake")


def test_result_video_defaults_to_mp4() -> None:
    assert ResultVideo(uri="https://example.com/out.mp4").content_type == "video/mp4"


def test_character_orientation_rejects_other_values() -> None:
    with pytest.raises(ValidationError):
        TransferRequest(
            reference_image=ReferenceImage(uri="./reference_image.png"),
            driving_video=DrivingVideo(uri="./driving_video.mov"),
            character_orientation="both",  # type: ignore[arg-type]
        )


def test_identity_element_requires_at_least_one_image() -> None:
    with pytest.raises(ValidationError):
        IdentityElement()


def test_identity_element_caps_additional_images_at_three() -> None:
    angles = tuple(ReferenceImage(uri=f"https://example.com/{i}.png") for i in range(4))
    with pytest.raises(ValidationError):
        IdentityElement(additional_images=angles)
    assert len(IdentityElement(additional_images=angles[:3]).additional_images) == 3


def test_transfer_request_has_no_element_by_default() -> None:
    request = TransferRequest(
        reference_image=ReferenceImage(uri="./reference_image.png"),
        driving_video=DrivingVideo(uri="./driving_video.mov"),
    )
    assert request.identity_element is None


def test_segmentation_request_defaults_to_one_person_at_half_threshold() -> None:
    request = SegmentationRequest(driving_video=DrivingVideo(uri="./driving_video_o1.mp4"))
    assert request.prompt == "person"
    assert request.detection_threshold == 0.5


def test_segmentation_request_strips_its_prompt_and_rejects_blanks() -> None:
    video = DrivingVideo(uri="./driving_video_o1.mp4")
    assert SegmentationRequest(driving_video=video, prompt="  person  ").prompt == "person"
    with pytest.raises(ValidationError):
        SegmentationRequest(driving_video=video, prompt=" ")


def test_mask_track_reports_track_ids_once_ascending() -> None:
    track = VideoMaskTrack(
        frames=(
            MaskFrame(frame_index=0, objects=(ObjectMask(track_id=2, rle="1 2"),)),
            MaskFrame(frame_index=1, objects=()),
            MaskFrame(
                frame_index=2,
                objects=(ObjectMask(track_id=2, rle="3 4"), ObjectMask(track_id=1, rle="5 6")),
            ),
        ),
        width=1080,
        height=1920,
        num_frames=3,
    )
    assert track.track_ids == (1, 2)


def test_mask_track_requires_positive_decode_dimensions() -> None:
    with pytest.raises(ValidationError):
        VideoMaskTrack(frames=(), width=0, height=1920, num_frames=0)


def test_empty_mask_track_has_no_track_ids() -> None:
    assert VideoMaskTrack(frames=(), width=8, height=8, num_frames=0).track_ids == ()
