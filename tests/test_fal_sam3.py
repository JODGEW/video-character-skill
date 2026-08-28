"""Unit tests for the SAM 3 segmentation provider. No network call is ever
made: the fal client is replaced by an in-memory fake, but the real
``fal_client`` status/error types are used so the mapping is checked against
the actual library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fal_client
import pytest
from pydantic import ValidationError

from fal_fakes import COMPLETED_OK, FakeFalClient
from video_character_skill.providers.base import ProviderError
from video_character_skill.providers.fal_sam3 import APP_ID, FalSam3VideoMaskProvider
from video_character_skill.schemas import (
    DrivingVideo,
    JobStatus,
    SegmentationRequest,
    VideoMaskTrack,
)

# One person tracked across three frames, briefly occluded in the middle one.
# Shaped exactly like fal's documented SAM3VideoObjectsOutput.
MASK_PAYLOAD: dict[str, Any] = {
    "frames": [
        {"frame_index": 0, "objects": [{"track_id": 1, "rle": "12 4 20 6"}]},
        {"frame_index": 1, "objects": []},
        {"frame_index": 2, "objects": [{"track_id": 1, "rle": "13 5 21 7"}]},
    ],
    "width": 1080,
    "height": 1920,
    "num_frames": 3,
}


@pytest.fixture
def local_video(tmp_path: Path) -> Path:
    video = tmp_path / "driving_video_o1.mp4"
    video.write_bytes(b"fake mp4")
    return video


def remote_request(**overrides: Any) -> SegmentationRequest:
    fields: dict[str, Any] = {
        "driving_video": DrivingVideo(uri="https://example.com/drive.mp4")
    }
    fields.update(overrides)
    return SegmentationRequest(**fields)


def mask_client(**overrides: Any) -> FakeFalClient:
    return FakeFalClient(result_response=MASK_PAYLOAD, **overrides)


# -- uploads ------------------------------------------------------------


def test_local_video_is_uploaded(local_video: Path) -> None:
    fake = FakeFalClient()
    provider = FalSam3VideoMaskProvider(client=fake)

    arguments = provider.build_arguments(
        SegmentationRequest(driving_video=DrivingVideo(uri=str(local_video)))
    )

    assert fake.uploads == [str(local_video)]
    assert arguments["video_url"] == "https://v3.fal.media/files/driving_video_o1.mp4"


def test_remote_video_is_passed_through_untouched() -> None:
    fake = FakeFalClient()
    provider = FalSam3VideoMaskProvider(client=fake)

    arguments = provider.build_arguments(remote_request())

    assert fake.uploads == []
    assert arguments["video_url"] == "https://example.com/drive.mp4"


def test_missing_local_video_fails_before_upload() -> None:
    fake = FakeFalClient()
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="local file not found"):
        provider.build_arguments(
            SegmentationRequest(driving_video=DrivingVideo(uri="./nope.mp4"))
        )
    assert fake.uploads == []


def test_upload_failure_becomes_provider_error(local_video: Path) -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("upload exploded"))
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="fal upload failed"):
        provider.build_arguments(
            SegmentationRequest(driving_video=DrivingVideo(uri=str(local_video)))
        )


# -- payload ------------------------------------------------------------


def test_person_payload_is_exactly_the_documented_three_fields() -> None:
    provider = FalSam3VideoMaskProvider(client=FakeFalClient())

    assert provider.build_arguments(remote_request()) == {
        "video_url": "https://example.com/drive.mp4",
        "prompt": "person",
        "detection_threshold": 0.5,
    }


def test_prompt_and_threshold_are_forwarded() -> None:
    provider = FalSam3VideoMaskProvider(client=FakeFalClient())

    arguments = provider.build_arguments(
        remote_request(prompt="person, cloth", detection_threshold=0.3)
    )

    assert arguments["prompt"] == "person, cloth"
    assert arguments["detection_threshold"] == 0.3


def test_submit_posts_to_the_sam3_app_and_returns_a_queued_job() -> None:
    fake = FakeFalClient()
    provider = FalSam3VideoMaskProvider(client=fake)

    job = provider.submit(remote_request())

    assert APP_ID == "fal-ai/sam-3/video-rle-objects"
    application, arguments = fake.submissions[0]
    assert application == APP_ID
    assert arguments["prompt"] == "person"
    assert job.job_id == "req-1"
    assert job.provider == "fal-sam-3-video-rle-objects"
    assert job.status is JobStatus.QUEUED


def test_blank_prompt_is_rejected_by_the_request() -> None:
    with pytest.raises(ValidationError):
        remote_request(prompt="   ")


@pytest.mark.parametrize("threshold", [0.0, 1.5])
def test_out_of_range_threshold_is_rejected(threshold: float) -> None:
    with pytest.raises(ValidationError):
        remote_request(detection_threshold=threshold)


# -- queue status mapping ----------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (fal_client.Queued(position=3), JobStatus.QUEUED),
        (fal_client.InProgress(logs=[]), JobStatus.RUNNING),
        (COMPLETED_OK, JobStatus.SUCCEEDED),
    ],
)
def test_queue_states_map_onto_job_statuses(
    status: fal_client.Status, expected: JobStatus
) -> None:
    provider = FalSam3VideoMaskProvider(client=FakeFalClient(status_response=status))

    job = provider.get_status("req-1")

    assert job.status is expected
    assert job.provider == "fal-sam-3-video-rle-objects"
    assert job.error is None


def test_completed_with_error_maps_to_failed() -> None:
    failed = fal_client.Completed(
        logs=[], metrics={}, error="no object matched", error_type="ValueError"
    )
    provider = FalSam3VideoMaskProvider(client=FakeFalClient(status_response=failed))

    job = provider.get_status("req-1")

    assert job.status is JobStatus.FAILED
    assert job.error == "ValueError: no object matched"


def test_status_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("status exploded"))
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="fal status failed"):
        provider.get_status("req-1")


# -- result parsing -----------------------------------------------------


def test_result_parses_into_a_mask_track() -> None:
    fake = mask_client(status_response=COMPLETED_OK)
    provider = FalSam3VideoMaskProvider(client=fake)

    track = provider.get_result("req-1")

    assert isinstance(track, VideoMaskTrack)
    assert (track.width, track.height, track.num_frames) == (1080, 1920, 3)
    assert len(track.frames) == 3
    assert track.frames[0].frame_index == 0
    assert track.frames[0].objects[0].track_id == 1
    assert track.frames[0].objects[0].rle == "12 4 20 6"
    assert track.frames[1].objects == ()
    assert fake.result_calls == [(APP_ID, "req-1")]


def test_a_single_tracked_person_yields_one_track_id() -> None:
    provider = FalSam3VideoMaskProvider(client=mask_client(status_response=COMPLETED_OK))

    assert provider.get_result("req-1").track_ids == (1,)


def test_two_people_yield_two_track_ids() -> None:
    payload = {
        "frames": [
            {
                "frame_index": 0,
                "objects": [
                    {"track_id": 2, "rle": "1 2"},
                    {"track_id": 1, "rle": "3 4"},
                ],
            }
        ],
        "width": 640,
        "height": 480,
        "num_frames": 1,
    }
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)

    assert FalSam3VideoMaskProvider(client=fake).get_result("req-1").track_ids == (1, 2)


def test_unknown_result_fields_are_ignored() -> None:
    payload = dict(MASK_PAYLOAD, fps=30, video={"url": "https://example.com/x.mp4"})
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)

    assert FalSam3VideoMaskProvider(client=fake).get_result("req-1").num_frames == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"video": {"url": "https://example.com/out.mp4"}},
        {"frames": [], "width": 1080, "height": 1920},
        {"frames": [{"objects": []}], "width": 1080, "height": 1920, "num_frames": 1},
        {"frames": [], "width": 0, "height": 1920, "num_frames": 0},
        "not a dict",
    ],
)
def test_malformed_results_become_provider_errors(payload: Any) -> None:
    fake = FakeFalClient(status_response=COMPLETED_OK, result_response=payload)
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="is not a mask track"):
        provider.get_result("req-1")


# -- result failures ----------------------------------------------------


def test_result_before_success_is_refused() -> None:
    fake = mask_client(status_response=fal_client.InProgress(logs=[]))
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="is running, not succeeded"):
        provider.get_result("req-1")
    assert fake.result_calls == []


def test_failed_job_reports_its_error_from_get_result() -> None:
    failed = fal_client.Completed(
        logs=[], metrics={}, error="out of memory", error_type="RuntimeError"
    )
    provider = FalSam3VideoMaskProvider(client=mask_client(status_response=failed))

    with pytest.raises(ProviderError, match="RuntimeError: out of memory"):
        provider.get_result("req-1")


def test_result_fetch_failure_becomes_provider_error() -> None:
    fake = mask_client(
        status_response=COMPLETED_OK,
        result_error=fal_client.FalClientError("422 elementReferList"),
    )
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="fal result failed"):
        provider.get_result("req-1")


def test_submit_failure_becomes_provider_error() -> None:
    fake = FakeFalClient(error=fal_client.FalClientError("submit exploded"))
    provider = FalSam3VideoMaskProvider(client=fake)

    with pytest.raises(ProviderError, match="fal submit failed"):
        provider.submit(remote_request())
